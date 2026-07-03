"""
agent-killswitch — Client-side API spend limit engine and local circuit breaker.

Stops runaway autonomous AI agent loops before they incur provider costs.
Runs entirely on-device (Termux / Android). No external proxies, no Docker,
no compiled extensions. Pure Python 3.11.

Usage:
    import killswitch

    client = killswitch.Client(
        budget_file="budget.json",
        model="gpt-4o",
        session_budget_usd=5.00,
    )

    response = client.complete(
        prompt="Summarise the latest RFC on post-quantum cryptography.",
        max_tokens=512,
        api_url="https://api.openai.com/v1/chat/completions",
        api_key=os.environ["OPENAI_API_KEY"],
        provider="openai",
    )
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import deque
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Deque, Dict, Optional, Tuple

import requests


# ─── Exceptions ──────────────────────────────────────────────────────────────

class OverBudgetError(Exception):
    """
    Raised when the estimated worst-case cost of a request exceeds the
    remaining session budget. The HTTP call is never fired.
    """

    def __init__(self, estimated_cost: float, remaining: float, model: str):
        self.estimated_cost = estimated_cost
        self.remaining = remaining
        self.model = model
        super().__init__(
            f"OverBudgetError: estimated cost ${estimated_cost:.6f} for "
            f"model '{model}' exceeds remaining session budget "
            f"${remaining:.6f}. Request blocked."
        )


class RunawayLoopError(Exception):
    """
    Raised when consecutive prompts are too similar (after normalisation),
    indicating the agent is stuck in a zombie loop making no real progress.
    """

    def __init__(self, similarity: float, threshold: float, window: int):
        self.similarity = similarity
        self.threshold = threshold
        self.window = window
        super().__init__(
            f"RunawayLoopError: last {window} prompts are too similar "
            f"(min pairwise similarity {similarity:.4f} >= threshold "
            f"{threshold}). Circuit breaker tripped."
        )


class UnknownModelError(Exception):
    """Raised when the requested model is absent from the pricing catalog."""
    pass


# ─── Helpers ─────────────────────────────────────────────────────────────────

_ISO_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r"(?:[Zz]|[+-]\d{2}:?\d{2})?"
)
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_DIGIT_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_prompt(text: str) -> str:
    """
    Normalise a prompt for similarity comparison.

    Removes digits, ISO timestamps, URLs, UUIDs, and collapses whitespace.
    This ensures that agent loops which vary only by counters, timestamps,
    or IDs are detected as repetitive.
    """
    text = text.lower()
    text = _ISO_TIMESTAMP_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _UUID_RE.sub("", text)
    text = _DIGIT_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _similarity(a: str, b: str) -> float:
    """Return the ratio of SequenceMatcher between two strings (0.0 – 1.0)."""
    return SequenceMatcher(None, a, b).ratio()


# ─── Core Client ─────────────────────────────────────────────────────────────

class Client:
    """
    API client wrapper with built-in spend-limit enforcement and
    runaway-loop circuit breaker.

    Lifecycle of every ``complete()`` call:

    1. Load ``budget.json`` and compute remaining session budget.
    2. Check loop detector — if the current prompt plus the last
       ``loop_window - 1`` prompts are all too similar, raise
       ``RunawayLoopError`` before any network activity.
    3. Record the prompt in the sliding-window history.
    4. Estimate worst-case token cost (input heuristic + max_tokens).
    5. If estimated cost > remaining budget → ``OverBudgetError`` (no HTTP).
    6. Fire the HTTP request via ``requests.post``.
    7. Parse actual usage from the provider response.
    8. Compute exact cost and append to the ledger, updating ``spent_usd``.
    9. Return the raw response dict.
    """

    DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
        "gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015},
        "gemini-1.5-flash": {"input_per_1k": 0.000075, "output_per_1k": 0.0003},
        "claude-3-5-sonnet": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    }

    # ── Construction ────────────────────────────────────────────────────

    def __init__(
        self,
        budget_file: str = "budget.json",
        model: str = "gpt-4o",
        session_budget_usd: Optional[float] = None,
        loop_similarity_threshold: float = 0.85,
        loop_window: int = 3,
        max_prompt_history: int = 50,
        request_timeout: int = 60,
    ) -> None:
        self.budget_file = budget_file
        self.model = model
        self.loop_similarity_threshold = loop_similarity_threshold
        self.loop_window = max(2, loop_window)
        self.request_timeout = request_timeout
        self.prompt_history: Deque[str] = deque(maxlen=max_prompt_history)

        # Last transaction metadata (populated after each successful call)
        self.last_transaction: Optional[Dict[str, Any]] = None

        if session_budget_usd is not None or not os.path.exists(budget_file):
            self._init_budget_file(session_budget_usd)

    # ── Budget file I/O ─────────────────────────────────────────────────

    def _init_budget_file(self, session_budget_usd: Optional[float]) -> None:
        if os.path.exists(self.budget_file):
            budget = self._load_budget()
            if session_budget_usd is not None:
                budget["session"]["total_budget_usd"] = float(session_budget_usd)
                budget["session"]["spent_usd"] = 0.0
                budget["session"]["updated_at"] = _now_iso()
                budget["ledger"] = []
                self._save_budget(budget)
        else:
            budget = {
                "session": {
                    "total_budget_usd": float(
                        session_budget_usd if session_budget_usd is not None else 10.0
                    ),
                    "spent_usd": 0.0,
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                },
                "pricing": dict(self.DEFAULT_PRICING),
                "ledger": [],
            }
            self._save_budget(budget)

    def _load_budget(self) -> Dict[str, Any]:
        with open(self.budget_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Coerce numeric fields to float in case of manual edits
        data["session"]["total_budget_usd"] = float(data["session"]["total_budget_usd"])
        data["session"]["spent_usd"] = float(data["session"]["spent_usd"])
        return data

    def _save_budget(self, budget: Dict[str, Any]) -> None:
        """Atomic write: temp file + os.replace (POSIX-atomic on Termux)."""
        budget["session"]["updated_at"] = _now_iso()
        dir_name = os.path.dirname(os.path.abspath(self.budget_file))
        if not dir_name:
            dir_name = "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp", prefix=".budget_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(budget, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp_path, self.budget_file)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # ── Cost computation ────────────────────────────────────────────────

    @staticmethod
    def _estimate_input_tokens(text: str) -> int:
        """
        Heuristic token estimate: ~4 characters per token for English text.
        This is the same approximation used by OpenAI's tiktoken for ASCII.
        """
        return max(1, len(text) // 4)

    def _get_rates(self, budget: Dict[str, Any]) -> Dict[str, float]:
        pricing = budget.get("pricing", {})
        # Merge with defaults so newly added models are always available
        merged = dict(self.DEFAULT_PRICING)
        merged.update(pricing)
        if self.model not in merged:
            raise UnknownModelError(
                f"Model '{self.model}' not found in pricing catalog. "
                f"Available: {sorted(merged.keys())}"
            )
        return merged[self.model]

    def _compute_cost(
        self, input_tokens: int, output_tokens: int, budget: Dict[str, Any]
    ) -> float:
        rates = self._get_rates(budget)
        input_cost = (input_tokens / 1000.0) * rates["input_per_1k"]
        output_cost = (output_tokens / 1000.0) * rates["output_per_1k"]
        return round(input_cost + output_cost, 8)

    def _remaining_budget(self, budget: Dict[str, Any]) -> float:
        return round(
            budget["session"]["total_budget_usd"] - budget["session"]["spent_usd"], 8
        )

    # ── Loop detection ──────────────────────────────────────────────────

    def _check_loop(self, current_prompt: str) -> None:
        """
        Compare the current prompt against the last ``loop_window - 1``
        prompts using normalised similarity. If *all* pairwise similarities
        in the window exceed the threshold, raise ``RunawayLoopError``.
        """
        needed = self.loop_window - 1
        if len(self.prompt_history) < needed:
            return

        current_norm = _normalize_prompt(current_prompt)
        recent = list(self.prompt_history)[-needed:]
        recent_norm = [_normalize_prompt(p) for p in recent]

        window = recent_norm + [current_norm]

        min_sim = 1.0
        for i in range(len(window)):
            for j in range(i + 1, len(window)):
                sim = _similarity(window[i], window[j])
                if sim < min_sim:
                    min_sim = sim

        if min_sim >= self.loop_similarity_threshold:
            raise RunawayLoopError(
                similarity=min_sim,
                threshold=self.loop_similarity_threshold,
                window=self.loop_window,
            )

    # ── HTTP request ────────────────────────────────────────────────────

    def _build_request(
        self,
        prompt: str,
        max_tokens: int,
        api_url: str,
        api_key: str,
        provider: str,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Construct headers and JSON body for the given provider."""
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        if provider == "anthropic":
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
        elif provider == "gemini":
            headers = {
                "Content-Type": "application/json",
            }
            # Gemini uses API key in query string; caller can embed it in api_url
            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            }
        else:
            # Default: OpenAI-compatible
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

        return headers, body

    def _make_request(
        self,
        prompt: str,
        max_tokens: int,
        api_url: str,
        api_key: str,
        provider: str,
    ) -> Dict[str, Any]:
        """Execute the HTTP POST and return parsed JSON. Raises on HTTP errors."""
        headers, body = self._build_request(
            prompt, max_tokens, api_url, api_key, provider
        )
        response = requests.post(
            api_url, headers=headers, json=body, timeout=self.request_timeout
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_usage(
        response_data: Dict[str, Any], provider: str
    ) -> Tuple[int, int]:
        """Extract (input_tokens, output_tokens) from a provider response."""
        usage = response_data.get("usage", {})

        if provider == "anthropic":
            return (
                int(usage.get("input_tokens", 0)),
                int(usage.get("output_tokens", 0)),
            )
        elif provider == "gemini":
            meta = usage.get("promptTokenCount", 0)
            cand = usage.get("candidatesTokenCount", 0)
            return int(meta), int(cand)
        else:
            # OpenAI-compatible
            return (
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
            )

    # ── Ledger ──────────────────────────────────────────────────────────

    def _record_transaction(
        self,
        budget: Dict[str, Any],
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float,
        actual_cost: float,
        endpoint: str,
    ) -> None:
        entry = {
            "timestamp": _now_iso(),
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(estimated_cost, 8),
            "actual_cost_usd": round(actual_cost, 8),
            "endpoint": endpoint,
        }
        budget["ledger"].append(entry)
        budget["session"]["spent_usd"] = round(
            budget["session"]["spent_usd"] + actual_cost, 8
        )
        self._save_budget(budget)

    # ── Public API ──────────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        max_tokens: int,
        api_url: str,
        api_key: str,
        provider: str = "openai",
    ) -> Dict[str, Any]:
        """
        Execute a completion request with full circuit-breaker protection.

        Returns the raw provider response dict.

        Raises:
            RunawayLoopError: if the last ``loop_window`` prompts are too similar.
            OverBudgetError: if the estimated cost exceeds remaining budget.
            UnknownModelError: if the model is not in the pricing catalog.
            requests.HTTPError: if the provider returns a non-2xx status.
        """
        # 1 — Load current budget state
        budget = self._load_budget()

        # 2 — Loop detection (before recording prompt)
        self._check_loop(prompt)

        # 3 — Record prompt in sliding window
        self.prompt_history.append(prompt)

        # 4 — Estimate worst-case cost
        estimated_input_tokens = self._estimate_input_tokens(prompt)
        estimated_cost = self._compute_cost(
            estimated_input_tokens, max_tokens, budget
        )

        # 5 — Budget gate
        remaining = self._remaining_budget(budget)
        if estimated_cost > remaining:
            raise OverBudgetError(
                estimated_cost=estimated_cost,
                remaining=remaining,
                model=self.model,
            )

        # 6 — Fire HTTP request
        response_data = self._make_request(
            prompt, max_tokens, api_url, api_key, provider
        )

        # 7 — Parse actual usage
        actual_input_tokens, actual_output_tokens = self._parse_usage(
            response_data, provider
        )

        # 8 — Compute exact cost
        actual_cost = self._compute_cost(
            actual_input_tokens, actual_output_tokens, budget
        )

        # 9 — Record transaction and persist
        self._record_transaction(
            budget=budget,
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            endpoint=api_url,
        )

        # Stash metadata for caller inspection
        updated_remaining = self._remaining_budget(budget)
        self.last_transaction = {
            "model": self.model,
            "input_tokens": actual_input_tokens,
            "output_tokens": actual_output_tokens,
            "estimated_cost_usd": round(estimated_cost, 8),
            "actual_cost_usd": round(actual_cost, 8),
            "remaining_budget_usd": updated_remaining,
        }

        return response_data

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the current session budget and ledger size."""
        budget = self._load_budget()
        remaining = self._remaining_budget(budget)
        return {
            "model": self.model,
            "total_budget_usd": budget["session"]["total_budget_usd"],
            "spent_usd": budget["session"]["spent_usd"],
            "remaining_usd": remaining,
            "transaction_count": len(budget["ledger"]),
            "prompt_history_count": len(self.prompt_history),
        }

    def reset_session(self, session_budget_usd: Optional[float] = None) -> None:
        """Zero out the spent amount and clear the ledger. Also clears prompt history."""
        budget = self._load_budget()
        budget["session"]["spent_usd"] = 0.0
        budget["ledger"] = []
        if session_budget_usd is not None:
            budget["session"]["total_budget_usd"] = float(session_budget_usd)
        budget["session"]["updated_at"] = _now_iso()
        self._save_budget(budget)
        self.prompt_history.clear()
        self.last_transaction = None

    def get_ledger(self) -> list:
        """Return the full transaction ledger."""
        budget = self._load_budget()
        return budget.get("ledger", [])
