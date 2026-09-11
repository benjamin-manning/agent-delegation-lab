# Agent Delegation Lab

A platform where you pre-commit to an AI agent, then watch it face 8 economic decision problems. How much of your budget will you spend for control over your agent?

## Setup

1. Install dependencies:
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

- You start with **$10.00** in guaranteed budget
- You choose a **control tier** -- more control costs more from your budget:
  - **Default** ($0): A random preset agent is assigned to you
  - **Choose** ($1): Pick which preset agent represents you
  - **Edit** ($3): Pick a preset and modify its instructions
  - **Custom** ($5): Write your agent's instructions from scratch
- Your agent faces **8 decision problems** spanning risk, loss, insurance, ambiguity, and investment
- You see the **distribution of payoffs** from 1,000 Monte Carlo simulations of the outcomes

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

- `app.py` -- Streamlit frontend
- `decisions.py` -- 8 decision problems with payoff simulation
- `agents.py` -- 5 agent presets and agent factory
- `engine.py` -- EDSL survey runner and payoff simulator
- `plots.py` -- Matplotlib visualizations
- `logger.py` -- JSON-lines research logger
