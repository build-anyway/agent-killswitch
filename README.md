# agent-killswitch

Client-side spend limiter and loop breaker for AI agent scripts.
Wraps OpenAI/Anthropic calls, blocks requests that would exceed a local budget,
and halts execution if it detects repeated near-identical prompts (not just exact matches).

## Setup
cp .env .env.local
# fill in your keys in .env.local

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python test_run.py

## Status
Passing test suite (7/7). Not yet validated against real production agent workloads.
