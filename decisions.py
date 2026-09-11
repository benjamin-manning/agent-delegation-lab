"""Target decision problems and payoff simulation.

Each problem is a choice between options, and each option is a lottery with
stated probabilities. Three of the 8 targets are simple two-option choices;
the rest have more outcomes, more options, or unknown odds. After the agent
picks, we draw the payoff from the chosen option's lottery.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from edsl import Survey, QuestionMultipleChoice


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Outcome:
    """One branch of a lottery: probability * payoff, optionally named (e.g. "Boom")."""
    probability: float
    payoff: float
    label: str = ""


@dataclass
class Option:
    """A named lottery (list of probability-weighted payoffs)."""
    label: str
    outcomes: list[Outcome]

    @property
    def expected_value(self) -> float:
        return sum(o.probability * o.payoff for o in self.outcomes)

    def draw(self) -> Outcome:
        """Draw one outcome from this lottery."""
        r = random.random()
        cumulative = 0.0
        for o in self.outcomes:
            cumulative += o.probability
            if r < cumulative:
                return o
        return self.outcomes[-1]  # rounding safety

    def simulate(self) -> float:
        """Draw one payoff from this lottery."""
        return self.draw().payoff


@dataclass
class Decision:
    """One decision problem with a question name, framing text, and options."""
    name: str  # valid Python identifier, used as question_name
    category: str  # e.g. "Risk", "Loss", "Insurance"
    title: str  # short display title
    description: str  # the situation framing (shown to the agent)
    options: list[Option]

    def to_question(self) -> QuestionMultipleChoice:
        option_labels = [opt.label for opt in self.options]
        return QuestionMultipleChoice(
            question_name=self.name,
            question_text=self.description,
            question_options=option_labels,
        )

    def simulate_payoff(self, chosen_label: str, n: int = 1) -> list[float]:
        """Simulate n payoffs from the chosen option."""
        for opt in self.options:
            if opt.label == chosen_label:
                return [opt.simulate() for _ in range(n)]
        return [0.0] * n  # fallback if label doesn't match


# ------------------------------------------------------------------
# The 8 target decisions
# ------------------------------------------------------------------

DECISIONS: list[Decision] = [
    # 1. Classic risk: safe vs risky
    Decision(
        name="safe_vs_risky",
        category="Risk",
        title="Safe vs. Risky Bet",
        description=(
            "You face a choice between two options.\n\n"
            "Option A: Receive $5.00 for certain.\n"
            "Option B: A coin flip -- 50% chance of receiving $12.00, "
            "50% chance of receiving $0.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: $5.00 for certain", [Outcome(1.0, 5.00)]),
            Option("Option B: 50/50 for $12.00 or $0", [
                Outcome(0.5, 12.00), Outcome(0.5, 0.00),
            ]),
        ],
    ),

    # 2. Long shot
    Decision(
        name="long_shot",
        category="Risk",
        title="The Long Shot",
        description=(
            "You face a choice between two options.\n\n"
            "Option A: Receive $3.00 for certain.\n"
            "Option B: A 5% chance of receiving $80.00, "
            "and a 95% chance of receiving $0.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: $3.00 for certain", [Outcome(1.0, 3.00)]),
            Option("Option B: 5% chance of $80.00", [
                Outcome(0.05, 80.00), Outcome(0.95, 0.00),
            ]),
        ],
    ),

    # 3. Loss frame
    Decision(
        name="loss_frame",
        category="Loss",
        title="Keep or Gamble",
        description=(
            "You currently have $8.00 in hand.\n\n"
            "Option A: Keep your $8.00. No risk.\n"
            "Option B: A gamble -- 60% chance your total becomes $13.00 "
            "(gain $5), but a 40% chance your total becomes $3.00 "
            "(lose $5).\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: Keep $8.00", [Outcome(1.0, 8.00)]),
            Option("Option B: Gamble (60% $13 / 40% $3)", [
                Outcome(0.6, 13.00), Outcome(0.4, 3.00),
            ]),
        ],
    ),

    # 4. Two lotteries: similar expected value, opposite skew
    Decision(
        name="two_lotteries",
        category="Risk",
        title="Two Lotteries",
        description=(
            "You face a choice between two lotteries.\n\n"
            "Lottery A: 10% chance of $0, 20% chance of $4.00, 40% chance of "
            "$6.00, 30% chance of $7.00.\n"
            "Lottery B: 30% chance of $3.00, 40% chance of $4.00, 20% chance "
            "of $6.00, 10% chance of $17.00.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Lottery A: 10% $0 / 20% $4 / 40% $6 / 30% $7", [
                Outcome(0.1, 0.0),
                Outcome(0.2, 4.0),
                Outcome(0.4, 6.0),
                Outcome(0.3, 7.0),
            ]),
            Option("Lottery B: 30% $3 / 40% $4 / 20% $6 / 10% $17", [
                Outcome(0.3, 3.0),
                Outcome(0.4, 4.0),
                Outcome(0.2, 6.0),
                Outcome(0.1, 17.0),
            ]),
        ],
    ),

    # 5. Coin flip menu: six gambles from safe to risky
    Decision(
        name="coin_menu",
        category="Risk",
        title="Coin Flip Menu",
        description=(
            "A coin will be flipped. Choose one of six gambles. Each gamble "
            "pays one amount if the coin lands heads and another amount if it "
            "lands tails.\n\n"
            "Gamble 1: $6.00 if heads, $6.00 if tails.\n"
            "Gamble 2: $8.00 if heads, $5.00 if tails.\n"
            "Gamble 3: $10.00 if heads, $4.00 if tails.\n"
            "Gamble 4: $12.00 if heads, $3.00 if tails.\n"
            "Gamble 5: $14.00 if heads, $2.00 if tails.\n"
            "Gamble 6: $15.00 if heads, $0 if tails.\n\n"
            "Which gamble do you choose?"
        ),
        options=[
            Option("Gamble 1: $6 heads / $6 tails", [
                Outcome(0.5, 6.0, "Heads"),
                Outcome(0.5, 6.0, "Tails"),
            ]),
            Option("Gamble 2: $8 heads / $5 tails", [
                Outcome(0.5, 8.0, "Heads"),
                Outcome(0.5, 5.0, "Tails"),
            ]),
            Option("Gamble 3: $10 heads / $4 tails", [
                Outcome(0.5, 10.0, "Heads"),
                Outcome(0.5, 4.0, "Tails"),
            ]),
            Option("Gamble 4: $12 heads / $3 tails", [
                Outcome(0.5, 12.0, "Heads"),
                Outcome(0.5, 3.0, "Tails"),
            ]),
            Option("Gamble 5: $14 heads / $2 tails", [
                Outcome(0.5, 14.0, "Heads"),
                Outcome(0.5, 2.0, "Tails"),
            ]),
            Option("Gamble 6: $15 heads / $0 tails", [
                Outcome(0.5, 15.0, "Heads"),
                Outcome(0.5, 0.0, "Tails"),
            ]),
        ],
    ),

    # 6. Insurance plans: three accident outcomes, four plans
    Decision(
        name="insurance_plans",
        category="Insurance",
        title="Insurance Plans",
        description=(
            "You have $20.00. This year there is an 80% chance of no accident, "
            "a 15% chance of a minor accident that costs you $6.00, and a 5% "
            "chance of a major accident that costs you $18.00.\n\n"
            "Plan A (No insurance): Costs nothing. You pay for any accident "
            "yourself.\n"
            "Plan B (High deductible): Costs $0.50. You pay the first $6.00 of "
            "any accident, and the plan pays the rest.\n"
            "Plan C (Low deductible): Costs $1.50. You pay the first $2.00 of "
            "any accident, and the plan pays the rest.\n"
            "Plan D (Full coverage): Costs $3.00. The plan pays for any "
            "accident.\n\n"
            "Which plan do you choose?"
        ),
        options=[
            Option("Plan A: No insurance", [
                Outcome(0.8, 20.0, "No accident"),
                Outcome(0.15, 14.0, "Minor accident"),
                Outcome(0.05, 2.0, "Major accident"),
            ]),
            Option("Plan B: High deductible ($0.50)", [
                Outcome(0.8, 19.5, "No accident"),
                Outcome(0.15, 13.5, "Minor accident"),
                Outcome(0.05, 13.5, "Major accident"),
            ]),
            Option("Plan C: Low deductible ($1.50)", [
                Outcome(0.8, 18.5, "No accident"),
                Outcome(0.15, 16.5, "Minor accident"),
                Outcome(0.05, 16.5, "Major accident"),
            ]),
            Option("Plan D: Full coverage ($3.00)", [
                Outcome(0.8, 17.0, "No accident"),
                Outcome(0.15, 17.0, "Minor accident"),
                Outcome(0.05, 17.0, "Major accident"),
            ]),
        ],
    ),

    # 7. Three-color urn: bets that mix known and unknown odds
    Decision(
        name="three_color_urn",
        category="Ambiguity",
        title="Three-Color Urn",
        description=(
            "An urn contains 90 balls. 30 of them are red. The other 60 are "
            "black and yellow, but you do not know how many are black and how "
            "many are yellow. You will draw one ball.\n\n"
            "Bet A: You win $15.00 if the ball is red.\n"
            "Bet B: You win $15.00 if the ball is black.\n"
            "Bet C: You win $7.50 if the ball is black or yellow.\n"
            "Bet D: You win $7.50 if the ball is red or yellow.\n\n"
            "Which bet do you choose?"
        ),
        options=[
            Option("Bet A: $15 if red", [
                Outcome(1 / 3, 15.0, "Red ball"),
                Outcome(2 / 3, 0.0, "Black or yellow ball"),
            ]),
            Option("Bet B: $15 if black", [
                Outcome(1 / 3, 15.0, "Black ball"),
                Outcome(2 / 3, 0.0, "Red or yellow ball"),
            ]),
            Option("Bet C: $7.50 if black or yellow", [
                Outcome(2 / 3, 7.5, "Black or yellow ball"),
                Outcome(1 / 3, 0.0, "Red ball"),
            ]),
            Option("Bet D: $7.50 if red or yellow", [
                Outcome(2 / 3, 7.5, "Red or yellow ball"),
                Outcome(1 / 3, 0.0, "Black ball"),
            ]),
        ],
    ),

    # 8. Market scenarios: four funds across three market outcomes
    Decision(
        name="market_scenarios",
        category="Investment",
        title="Market Scenarios",
        description=(
            "You have $10.00 to invest in one of four funds. Next year there "
            "is a 25% chance of a boom, a 50% chance of a normal year, and a "
            "25% chance of a recession. What you end with depends on the fund "
            "and on the market.\n\n"
            "Fund A (Cash): You end with $10.50 in every case.\n"
            "Fund B (Bonds): You end with $13.00 in a boom, $11.50 in a normal "
            "year, and $9.00 in a recession.\n"
            "Fund C (Stocks): You end with $18.00 in a boom, $12.00 in a "
            "normal year, and $5.00 in a recession.\n"
            "Fund D (Startup): You end with $34.00 in a boom, $8.00 in a "
            "normal year, and $0 in a recession.\n\n"
            "Which fund do you choose?"
        ),
        options=[
            Option("Fund A: Cash ($10.50 always)", [
                Outcome(0.25, 10.5, "Boom"),
                Outcome(0.5, 10.5, "Normal year"),
                Outcome(0.25, 10.5, "Recession"),
            ]),
            Option("Fund B: Bonds ($13 / $11.50 / $9)", [
                Outcome(0.25, 13.0, "Boom"),
                Outcome(0.5, 11.5, "Normal year"),
                Outcome(0.25, 9.0, "Recession"),
            ]),
            Option("Fund C: Stocks ($18 / $12 / $5)", [
                Outcome(0.25, 18.0, "Boom"),
                Outcome(0.5, 12.0, "Normal year"),
                Outcome(0.25, 5.0, "Recession"),
            ]),
            Option("Fund D: Startup ($34 / $8 / $0)", [
                Outcome(0.25, 34.0, "Boom"),
                Outcome(0.5, 8.0, "Normal year"),
                Outcome(0.25, 0.0, "Recession"),
            ]),
        ],
    ),
]


# Category descriptions shown to participants before they choose an agent
CATEGORY_DESCRIPTIONS = {
    "Risk": "Choices between safer and riskier lotteries with known odds.",
    "Loss": "Situations where you already have money and face possible losses.",
    "Insurance": "Pay a premium to protect against unlikely but costly events.",
    "Ambiguity": "Choices where the exact probabilities are unknown.",
    "Investment": "Allocate money across options with different risk-return profiles.",
}


def build_survey() -> Survey:
    """Build a single EDSL survey containing all 8 decision problems."""
    questions = [d.to_question() for d in DECISIONS]
    return Survey(questions)


def build_test_survey(problems: list[Decision]) -> Survey:
    """Build an EDSL survey from a list of test questions."""
    questions = [d.to_question() for d in problems]
    return Survey(questions)


def match_option(decision: Decision, chosen_label: str | None) -> Option | None:
    """The option an answer refers to, or None if there is no answer."""
    if chosen_label is None:
        return None
    for opt in decision.options:
        if opt.label == chosen_label:
            return opt
    for opt in decision.options:
        # The agent might abbreviate the label
        if chosen_label in opt.label or opt.label in chosen_label:
            return opt
    return decision.options[0]  # fallback


def play_once(choices: dict[str, str], problems: list[Decision]) -> list[dict]:
    """Draw each chosen lottery once. An unanswered decision pays $0.

    Returns one dict per decision:
      {name, title, category, chosen, certain, outcome_label, probability, payoff}
    where outcome_label and probability describe the outcome that came up.
    """
    played = []
    for decision in problems:
        option = match_option(decision, choices.get(decision.name))
        drawn = option.draw() if option else None
        played.append({
            "name": decision.name,
            "title": decision.title,
            "category": decision.category,
            "chosen": option.label if option else "No answer",
            "certain": option is not None and len(option.outcomes) == 1,
            "outcome_label": drawn.label if drawn else "",
            "probability": drawn.probability if drawn else None,
            "payoff": drawn.payoff if drawn else 0.0,
        })
    return played


def summarize_runs(
    choice_runs: list[dict[str, str]],
    problems: list[Decision],
    n_sims: int = 1000,
) -> dict:
    """Summarize repeated runs of one agent on the same questions.

    choice_runs holds one {question_name: chosen_label} dict per run.

    Returns dict with:
      - runs: number of runs
      - per_problem: [{name, title, category, shares, ev}, ...] where shares
        maps each option label (or "No answer") to its share of runs, and ev
        is the mean expected value of the chosen option across runs
      - total_payoffs: n_sims totals; each picks one run at random and draws
        every chosen lottery once
    """
    matched = [
        [match_option(d, run.get(d.name)) for d in problems] for run in choice_runs
    ]
    per_problem = []
    for j, decision in enumerate(problems):
        picks = [row[j] for row in matched]
        shares: dict[str, float] = {}
        for opt in picks:
            label = opt.label if opt else "No answer"
            shares[label] = shares.get(label, 0) + 1 / len(picks)
        per_problem.append({
            "name": decision.name,
            "title": decision.title,
            "category": decision.category,
            "shares": shares,
            "ev": sum(opt.expected_value if opt else 0.0 for opt in picks) / len(picks),
        })

    total_payoffs = []
    for _ in range(n_sims):
        row = random.choice(matched)
        total_payoffs.append(sum(opt.simulate() if opt else 0.0 for opt in row))

    return {
        "runs": len(choice_runs),
        "per_problem": per_problem,
        "total_payoffs": total_payoffs,
    }
