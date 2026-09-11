# Agent Delegation Lab

A platform where you choose an AI agent to make 8 money decisions for you. You can read the 8 decisions first, and you can pay to test agents on practice questions before you choose.

## Setup

1. Use Python 3.12. EDSL does not run on Python 3.14. On Streamlit Cloud, pick the Python version under Advanced settings when you deploy.

   Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your Expected Parrot API key:
   ```
   export EXPECTED_PARROT_API_KEY="your-key-here"
   ```
   On Streamlit Cloud, add `EXPECTED_PARROT_API_KEY = "your-key-here"` under the app's Settings > Secrets.

3. Run the app:
   ```
   streamlit run app.py
   ```

## How it works

You start with a budget of $10.00. You keep any budget you do not spend, plus what your agent earns.

1. You read the 8 decisions your agent will face. You cannot test any agent on them. Three are simple choices between two options. The other five have more outcomes, more options, or unknown odds, e.g. four insurance plans across three accident outcomes.
2. You can pay to test agents on practice questions. You pick 1, 10, or 100 runs per question. Each run is a new answer from the agent, so more runs show how often it picks each option. One run of one agent on one question costs:

   | Bank | Preset agent | Your own agent |
   |---|---|---|
   | Category banks (5 banks of 20 questions) | $0.01 | $0.02 |
   | Close variant banks (8 banks of 8 questions) | $0.03 | $0.06 |

   Each test shows the share of runs that picked each option, and a payout histogram for each agent from 1,000 simulated rounds. Each round takes one of the agent's runs at random and draws each lottery once.
3. You choose a control tier. More control costs more.
   - Default ($0). You get a preset agent picked at random.
   - Choose ($1). You pick which preset agent represents you.
   - Custom ($5). You write your agent's instructions from scratch.
4. Your agent answers each of the 8 decisions once, and each lottery is drawn once. You see what it chose, what came up, and what you earned.

Your own agent's instructions must describe a general approach. Before a test of your own agent and before a Custom run, a model checks the instructions against the 8 decisions. It rejects instructions that refer to a specific decision, e.g. by naming its amounts or saying what to choose in it. A rejected test is not charged.

Tests and the final run ask the model for new answers every time and skip EDSL's cache. The instruction check uses the cache, so the same text always gets the same verdict.

## Question banks

The banks are built in `banks.py` from question templates with fixed random seeds, so every session sees the same questions.

- Each category bank holds questions of one category, with different amounts and odds from the 8 decisions. It starts with the hand-written practice questions and fills the rest from templates. It leaves out any question that is a close variant of one of the 8 decisions.
- Each close variant bank holds small changes to one of the 8 decisions, written in the same words.

A question is a close variant of a decision when it has the same number of options and outcomes, every probability is within 10 percentage points, and every dollar amount is within 25% or $0.50, whichever is larger.

## Agent Presets

- **The Maximizer**: Pure expected-value optimizer, risk-neutral
- **The Protector**: Risk-averse, prefers certainty, buys insurance
- **The Adventurer**: Risk-seeking, goes for big payoffs
- **The Analyst**: Balanced, seeks best risk-adjusted returns
- **The Gut-Feeler**: Intuition-driven, unpredictable

## Decision Categories

- **Risk**: Safe vs. risky lotteries with known odds
- **Loss**: Situations involving potential losses from an endowment
- **Insurance**: Pay a premium to protect against unlikely costly events
- **Ambiguity**: Choices where exact probabilities are unknown
- **Investment**: Allocate across options with different risk-return profiles

## Files

- `app.py` is the Streamlit app.
- `decisions.py` holds the 8 decisions and the payoff simulation.
- `banks.py` builds the practice question banks.
- `agents.py` holds the 5 preset agents.
- `engine.py` runs agents on questions through EDSL.
- `plots.py` draws the charts.
- `logger.py` writes one JSON line per test, per final run, and per rejected instruction check to `logs/sessions.jsonl`.
