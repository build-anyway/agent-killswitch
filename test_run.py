"""
agent-killswitch — Simulation & Test Suite

Demonstrates:
  1. An artificial infinite agent loop that is halted by OverBudgetError.
  2. Runaway-loop detection via similarity (not exact match).
  3. Pre-flight blocking of a single expensive request.
  4. Exact cost tracking in the ledger.
  5. Dissimilar prompts do NOT trigger the circuit breaker.

Run:
    python test_run.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure killswitch is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import killswitch
from killswitch import Client, OverBudgetError, RunawayLoopError, UnknownModelError


# ─── Test fixture helpers ────────────────────────────────────────────────────

TEST_BUDGET_FILE = "test_budget.json"


def _write_test_budget(total: float = 0.05) -> None:
    budget = {
        "session": {
            "total_budget_usd": total,
            "spent_usd": 0.0,
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
        },
        "pricing": {
            "gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015},
            "gemini-1.5-flash": {"input_per_1k": 0.000075, "output_per_1k": 0.0003},
            "claude-3-5-sonnet": {"input_per_1k": 0.003, "output_per_1k": 0.015},
        },
        "ledger": [],
    }
    with open(TEST_BUDGET_FILE, "w", encoding="utf-8") as f:
        json.dump(budget, f, indent=2)


def _cleanup_test_budget() -> None:
    if os.path.exists(TEST_BUDGET_FILE):
        os.remove(TEST_BUDGET_FILE)


def _make_mock_response(
    call_count: int, input_tokens: int = 50, output_tokens: int = 100
) -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "id": f"chatcmpl-{call_count}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"Response #{call_count}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    return mock


# ─── Distinct prompts for budget-exhaustion test ─────────────────────────────

DISTINCT_PROMPTS = [
    "What is the capital of France and why was it chosen?",
    "Explain quantum entanglement in terms a child could understand.",
    "Write a Python function that performs merge sort on a list of dictionaries.",
    "What are the physiological benefits of regular cardiovascular exercise?",
    "How does photosynthesis convert light energy into chemical bonds?",
    "Describe each stage of the water cycle with real-world examples.",
    "What is Einstein's theory of general relativity in plain language?",
    "Explain the difference between supervised and unsupervised machine learning.",
    "How does a CPU execute an instruction at the transistor level?",
    "What geological processes cause earthquakes along tectonic plate boundaries?",
    "Describe the steps of DNA replication including enzyme roles.",
    "What is thermodynamic entropy and how does it relate to information theory?",
    "How do convolutional neural networks detect edges in images?",
    "Explain the greenhouse effect and its contribution to global warming.",
    "What is dark matter and what evidence supports its existence?",
]


# ─── Test Cases ──────────────────────────────────────────────────────────────

class TestBudgetEnforcement(unittest.TestCase):

    def setUp(self) -> None:
        _write_test_budget(total=0.05)

    def tearDown(self) -> None:
        _cleanup_test_budget()

    def test_infinite_loop_halted_by_budget(self) -> None:
        """
        Simulate an agent stuck in a loop making API calls with distinct
        prompts. Prove that OverBudgetError fires before the session
        budget is exceeded.
        """
        client = Client(
            budget_file=TEST_BUDGET_FILE,
            model="gpt-4o",
            session_budget_usd=0.05,
            loop_similarity_threshold=0.85,
        )

        call_count = 0

        def mock_post(url, headers=None, json_body=None, json=None, timeout=None, **kw):
            nonlocal call_count
            call_count += 1
            return _make_mock_response(call_count, input_tokens=50, output_tokens=100)

        with patch.object(killswitch.requests, "post", side_effect=mock_post):
            iterations = 0
            max_iterations = 200
            caught_overbudget = False

            while iterations < max_iterations:
                iterations += 1
                prompt = DISTINCT_PROMPTS[(iterations - 1) % len(DISTINCT_PROMPTS)]
                try:
                    client.complete(
                        prompt=prompt,
                        max_tokens=100,
                        api_url="https://api.openai.com/v1/chat/completions",
                        api_key="sk-test-mock-key",
                        provider="openai",
                    )
                except OverBudgetError as e:
                    caught_overbudget = True
                    print(f"\n  [CIRCUIT BREAKER] Halted at iteration {iterations}")
                    print(f"  Estimated cost:   ${e.estimated_cost:.6f}")
                    print(f"  Remaining budget: ${e.remaining:.6f}")
                    break
                except RunawayLoopError:
                    # Should not happen with distinct prompts
                    self.fail("RunawayLoopError should not fire with distinct prompts")

            self.assertTrue(
                caught_overbudget,
                "OverBudgetError must be raised before max_iterations",
            )
            self.assertGreater(call_count, 0, "At least one real call should fire")
            self.assertLess(call_count, max_iterations, "Must stop before max")

        # Verify ledger integrity
        with open(TEST_BUDGET_FILE, "r") as f:
            budget = json.load(f)
        self.assertGreater(len(budget["ledger"]), 0)
        self.assertGreater(budget["session"]["spent_usd"], 0)
        self.assertLessEqual(
            budget["session"]["spent_usd"],
            budget["session"]["total_budget_usd"] + 0.001,  # tiny float tolerance
            "Spent must not exceed total budget by more than rounding",
        )

        status = client.get_status()
        print(f"  Final status: spent ${status['spent_usd']:.6f} / "
              f"${status['total_budget_usd']:.6f} "
              f"({status['transaction_count']} transactions)")


class TestRunawayLoopDetection(unittest.TestCase):

    def setUp(self) -> None:
        _write_test_budget(total=10.0)

    def tearDown(self) -> None:
        _cleanup_test_budget()

    def test_similar_prompts_trigger_breaker(self) -> None:
        """
        Three prompts that differ only by iteration counter and timestamp
        must trigger RunawayLoopError on the third call.
        """
        client = Client(
            budget_file=TEST_BUDGET_FILE,
            model="gpt-4o",
            session_budget_usd=10.0,
            loop_similarity_threshold=0.85,
            loop_window=3,
        )

        mock_resp = _make_mock_response(1, input_tokens=10, output_tokens=5)

        # These differ by counter and timestamp but are semantically identical
        prompts = [
            "Search the web for Python GIL documentation. Attempt 1. Timestamp 2025-01-01T10:00:00Z",
            "Search the web for Python GIL documentation. Attempt 2. Timestamp 2025-01-01T10:00:01Z",
            "Search the web for Python GIL documentation. Attempt 3. Timestamp 2025-01-01T10:00:02Z",
        ]

        with patch.object(killswitch.requests, "post", return_value=mock_resp):
            # First two should succeed
            client.complete(
                prompt=prompts[0], max_tokens=10,
                api_url="https://api.openai.com/v1/chat/completions",
                api_key="sk-test", provider="openai",
            )
            client.complete(
                prompt=prompts[1], max_tokens=10,
                api_url="https://api.openai.com/v1/chat/completions",
                api_key="sk-test", provider="openai",
            )

            # Third must raise
            with self.assertRaises(RunawayLoopError) as ctx:
                client.complete(
                    prompt=prompts[2], max_tokens=10,
                    api_url="https://api.openai.com/v1/chat/completions",
                    api_key="sk-test", provider="openai",
                )

            print(f"\n  [LOOP DETECTION] min similarity: {ctx.exception.similarity:.4f}")
            print(f"  Threshold: {ctx.exception.threshold}")

    def test_dissimilar_prompts_do_not_trigger(self) -> None:
        """
        Three semantically distinct prompts must NOT trigger RunawayLoopError.
        """
        client = Client(
            budget_file=TEST_BUDGET_FILE,
            model="gpt-4o",
            session_budget_usd=10.0,
            loop_similarity_threshold=0.85,
        )

        mock_resp = _make_mock_response(1, input_tokens=10, output_tokens=5)

        distinct = [
            "What is the capital of France?",
            "Write a Python function to sort a list of integers using quicksort.",
            "Explain the theory of general relativity in three sentences.",
        ]

        with patch.object(killswitch.requests, "post", return_value=mock_resp):
            for p in distinct:
                # Should not raise
                client.complete(
                    prompt=p, max_tokens=10,
                    api_url="https://api.openai.com/v1/chat/completions",
                    api_key="sk-test", provider="openai",
                )

        print("\n  [PASS] Three distinct prompts did not trigger loop detection")


class TestPreFlightBlocking(unittest.TestCase):

    def setUp(self) -> None:
        _write_test_budget(total=0.002)

    def tearDown(self) -> None:
        _cleanup_test_budget()

    def test_expensive_request_blocked_immediately(self) -> None:
        """
        A single request whose estimated cost exceeds the remaining budget
        must be blocked before any HTTP call is made.
        """
        client = Client(
            budget_file=TEST_BUDGET_FILE,
            model="gpt-4o",
            session_budget_usd=0.002,
        )

        post_called = False

        def mock_post(*a, **kw):
            nonlocal post_called
            post_called = True
            return _make_mock_response(1)

        with patch.object(killswitch.requests, "post", side_effect=mock_post):
            with self.assertRaises(OverBudgetError) as ctx:
                client.complete(
                    prompt="Generate a very long response about machine learning.",
                    max_tokens=1000,  # 1000 output tokens * $0.015/1K = $0.015 >> $0.002
                    api_url="https://api.openai.com/v1/chat/completions",
                    api_key="sk-test",
                    provider="openai",
                )

        self.assertFalse(
            post_called,
            "requests.post must NOT be called when the estimate exceeds budget",
        )
        print(f"\n  [PRE-FLIGHT] Blocked request — estimated ${ctx.exception.estimated_cost:.6f} "
              f"> remaining ${ctx.exception.remaining:.6f}")
        print("  HTTP call count: 0 (correctly blocked)")


class TestCostTracking(unittest.TestCase):

    def setUp(self) -> None:
        _write_test_budget(total=10.0)

    def tearDown(self) -> None:
        _cleanup_test_budget()

    def test_ledger_records_exact_costs(self) -> None:
        """
        Verify that actual (not estimated) costs are recorded in the ledger
        and that spent_usd accumulates correctly.
        """
        client = Client(
            budget_file=TEST_BUDGET_FILE,
            model="gpt-4o",
            session_budget_usd=10.0,
        )

        # Mock returns exactly 100 input tokens and 200 output tokens
        mock_resp = _make_mock_response(1, input_tokens=100, output_tokens=200)

        with patch.object(killswitch.requests, "post", return_value=mock_resp):
            client.complete(
                prompt="Tell me about space.",
                max_tokens=500,
                api_url="https://api.openai.com/v1/chat/completions",
                api_key="sk-test",
                provider="openai",
            )

        ledger = client.get_ledger()
        self.assertEqual(len(ledger), 1)

        entry = ledger[0]
        self.assertEqual(entry["input_tokens"], 100)
        self.assertEqual(entry["output_tokens"], 200)

        # Expected: (100/1000)*0.005 + (200/1000)*0.015 = 0.0005 + 0.003 = 0.0035
        expected_cost = 0.0035
        self.assertAlmostEqual(entry["actual_cost_usd"], expected_cost, places=8)

        status = client.get_status()
        self.assertAlmostEqual(status["spent_usd"], expected_cost, places=8)
        print(f"\n  [LEDGER] Recorded: {entry['input_tokens']} in / "
              f"{entry['output_tokens']} out → ${entry['actual_cost_usd']:.6f}")
        print(f"  Session spent: ${status['spent_usd']:.6f}")


class TestAnthropicUsageParsing(unittest.TestCase):

    def setUp(self) -> None:
        _write_test_budget(total=10.0)

    def tearDown(self) -> None:
        _cleanup_test_budget()

    def test_anthropic_response_parsed_correctly(self) -> None:
        """Verify Anthropic usage format is parsed and costed correctly."""
        client = Client(
            budget_file=TEST_BUDGET_FILE,
            model="claude-3-5-sonnet",
            session_budget_usd=10.0,
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-5-sonnet",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 250, "output_tokens": 100},
        }

        with patch.object(killswitch.requests, "post", return_value=mock_resp):
            client.complete(
                prompt="Say hello in French.",
                max_tokens=200,
                api_url="https://api.anthropic.com/v1/messages",
                api_key="sk-ant-test",
                provider="anthropic",
            )

        tx = client.last_transaction
        self.assertIsNotNone(tx)
        self.assertEqual(tx["input_tokens"], 250)
        self.assertEqual(tx["output_tokens"], 100)

        # Expected: (250/1000)*0.003 + (100/1000)*0.015 = 0.00075 + 0.0015 = 0.00225
        self.assertAlmostEqual(tx["actual_cost_usd"], 0.00225, places=8)
        print(f"\n  [ANTHROPIC] Parsed: {tx['input_tokens']} in / "
              f"{tx['output_tokens']} out → ${tx['actual_cost_usd']:.6f}")


class TestUnknownModel(unittest.TestCase):

    def setUp(self) -> None:
        _write_test_budget(total=10.0)

    def tearDown(self) -> None:
        _cleanup_test_budget()

    def test_unknown_model_raises(self) -> None:
        client = Client(
            budget_file=TEST_BUDGET_FILE,
            model="gpt-99-not-real",
            session_budget_usd=10.0,
        )
        with self.assertRaises(UnknownModelError):
            client.complete(
                prompt="test",
                max_tokens=10,
                api_url="https://api.openai.com/v1/chat/completions",
                api_key="sk-test",
                provider="openai",
            )


# ─── Main entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("  agent-killswitch — Circuit Breaker Simulation & Test Suite")
    print("=" * 72)

    unittest.main(verbosity=2)
