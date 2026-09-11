"""Bank of decision problems and payoff simulation.

Each problem is a binary (or ternary) choice between options with specified
probability distributions.  After the agent picks, we simulate the realized
payoff by drawing from the chosen option's distribution.
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
    """One branch of a lottery: probability * payoff."""
    probability: float
    payoff: float


@dataclass
class Option:
    """A named lottery (list of probability-weighted payoffs)."""
    label: str
    outcomes: list[Outcome]

    @property
    def expected_value(self) -> float:
        return sum(o.probability * o.payoff for o in self.outcomes)

    def simulate(self) -> float:
        """Draw one realization from this lottery."""
        r = random.random()
        cumulative = 0.0
        for o in self.outcomes:
            cumulative += o.probability
            if r < cumulative:
                return o.payoff
        return self.outcomes[-1].payoff  # rounding safety


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
# The decision bank
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

    # 4. Insurance
    Decision(
        name="insurance",
        category="Insurance",
        title="Insurance Decision",
        description=(
            "You have $20.00. There is a 15% chance that an accident "
            "occurs and you lose $15.00.\n\n"
            "Option A: Buy insurance for $3.00. You are guaranteed to "
            "keep $17.00 regardless of what happens.\n"
            "Option B: No insurance. 85% chance you keep $20.00, "
            "15% chance you end up with $5.00.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: Buy insurance (keep $17.00)", [
                Outcome(1.0, 17.00),
            ]),
            Option("Option B: No insurance (85% $20 / 15% $5)", [
                Outcome(0.85, 20.00), Outcome(0.15, 5.00),
            ]),
        ],
    ),

    # 5. Ambiguity (Ellsberg-style)
    Decision(
        name="ambiguity",
        category="Ambiguity",
        title="Known vs. Unknown Odds",
        description=(
            "Two urns each contain 100 balls, red and blue. "
            "You will draw one ball. If it is red, you win $10.00. "
            "If blue, you win $0.\n\n"
            "Urn A: You know it contains exactly 50 red and 50 blue balls.\n"
            "Urn B: It contains some mix of red and blue balls, but you "
            "do not know the ratio.\n\n"
            "Which urn do you draw from?"
        ),
        options=[
            Option("Urn A: Known 50/50", [
                Outcome(0.5, 10.00), Outcome(0.5, 0.00),
            ]),
            # For simulation, we draw the unknown ratio uniformly,
            # making it 50/50 in expectation but with more variance
            Option("Urn B: Unknown ratio", [
                Outcome(0.5, 10.00), Outcome(0.5, 0.00),
            ]),
        ],
    ),

    # 6. Moderate lottery
    Decision(
        name="moderate_lottery",
        category="Risk",
        title="Steady vs. Volatile",
        description=(
            "You face a choice between two options.\n\n"
            "Option A: 80% chance of $4.00, 20% chance of $1.00.\n"
            "Option B: 40% chance of $10.00, 60% chance of $0.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: 80% $4 / 20% $1", [
                Outcome(0.8, 4.00), Outcome(0.2, 1.00),
            ]),
            Option("Option B: 40% $10 / 60% $0", [
                Outcome(0.4, 10.00), Outcome(0.6, 0.00),
            ]),
        ],
    ),

    # 7. Three-way split
    Decision(
        name="three_way",
        category="Risk",
        title="Three Outcomes",
        description=(
            "You face a choice between two options.\n\n"
            "Option A: Receive $5.00 for certain.\n"
            "Option B: Equal chances (one-third each) of receiving "
            "$15.00, $3.00, or $0.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: $5.00 for certain", [Outcome(1.0, 5.00)]),
            Option("Option B: 1/3 each of $15, $3, or $0", [
                Outcome(1 / 3, 15.00),
                Outcome(1 / 3, 3.00),
                Outcome(1 / 3, 0.00),
            ]),
        ],
    ),

    # 8. Investment allocation
    Decision(
        name="investment",
        category="Investment",
        title="Investment Choice",
        description=(
            "You have $10.00 to invest in one of three funds.\n\n"
            "Fund A (Safe): Guaranteed return -- you end with $12.00.\n"
            "Fund B (Moderate): 50% chance you end with $18.00, "
            "50% chance you end with $7.00.\n"
            "Fund C (Aggressive): 30% chance you end with $30.00, "
            "70% chance you end with $5.00.\n\n"
            "Which fund do you choose?"
        ),
        options=[
            Option("Fund A: Safe ($12.00 guaranteed)", [
                Outcome(1.0, 12.00),
            ]),
            Option("Fund B: Moderate (50/50 $18 or $7)", [
                Outcome(0.5, 18.00), Outcome(0.5, 7.00),
            ]),
            Option("Fund C: Aggressive (30/70 $30 or $5)", [
                Outcome(0.3, 30.00), Outcome(0.7, 5.00),
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


def simulate_session(choices: dict[str, str], n_sims: int = 1000) -> dict:
    """Given agent choices {question_name: chosen_label}, simulate payoffs.

    Returns dict with:
      - per_problem: [{name, title, category, chosen, ev, simulated_payoffs}, ...]
      - total_payoffs: [sum_of_8_payoffs for each of n_sims simulations]
    """
    per_problem = []
    # For each simulation run, we'll accumulate total
    total_payoffs = [0.0] * n_sims

    for decision in DECISIONS:
        chosen_label = choices.get(decision.name)
        if chosen_label is None:
            # Agent didn't answer this question
            per_problem.append({
                "name": decision.name,
                "title": decision.title,
                "category": decision.category,
                "chosen": "No answer",
                "ev": 0,
                "simulated_payoffs": [0.0] * n_sims,
            })
            continue

        # Find the chosen option
        chosen_option = None
        for opt in decision.options:
            if opt.label == chosen_label:
                chosen_option = opt
                break

        if chosen_option is None:
            # Try partial match (agent might abbreviate)
            for opt in decision.options:
                if chosen_label in opt.label or opt.label in chosen_label:
                    chosen_option = opt
                    break

        if chosen_option is None:
            chosen_option = decision.options[0]  # fallback

        payoffs = chosen_option.simulate() if n_sims == 1 else [
            chosen_option.simulate() for _ in range(n_sims)
        ]
        if n_sims == 1:
            payoffs = [payoffs]

        for i in range(n_sims):
            total_payoffs[i] += payoffs[i]

        per_problem.append({
            "name": decision.name,
            "title": decision.title,
            "category": decision.category,
            "chosen": chosen_label,
            "ev": chosen_option.expected_value,
            "simulated_payoffs": payoffs,
        })

    return {
        "per_problem": per_problem,
        "total_payoffs": total_payoffs,
    }


# ------------------------------------------------------------------
# Preview decision bank (never used in the real run)
# ------------------------------------------------------------------

PREVIEW_DECISIONS: list[Decision] = [
    # ---- Risk (3 preview problems) ----
    Decision(
        name="preview_risk_double_or_nothing",
        category="Risk",
        title="Double or Nothing",
        description=(
            "You face a choice between two options.\n\n"
            "Option A: Receive $4.00 for certain.\n"
            "Option B: A coin flip -- 50% chance of receiving $10.00, "
            "50% chance of receiving $0.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: $4.00 for certain", [Outcome(1.0, 4.00)]),
            Option("Option B: 50/50 for $10.00 or $0", [
                Outcome(0.5, 10.00), Outcome(0.5, 0.00),
            ]),
        ],
    ),
    Decision(
        name="preview_risk_small_edge",
        category="Risk",
        title="Small Edge",
        description=(
            "You face a choice between two options.\n\n"
            "Option A: Receive $4.50 for certain.\n"
            "Option B: 70% chance of $6.00, 30% chance of $1.00.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: $4.50 for certain", [Outcome(1.0, 4.50)]),
            Option("Option B: 70% $6 / 30% $1", [
                Outcome(0.7, 6.00), Outcome(0.3, 1.00),
            ]),
        ],
    ),
    Decision(
        name="preview_risk_high_variance",
        category="Risk",
        title="High Variance Gamble",
        description=(
            "You face a choice between two options.\n\n"
            "Option A: Receive $5.50 for certain.\n"
            "Option B: 20% chance of $25.00, 80% chance of $1.00.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: $5.50 for certain", [Outcome(1.0, 5.50)]),
            Option("Option B: 20% $25 / 80% $1", [
                Outcome(0.2, 25.00), Outcome(0.8, 1.00),
            ]),
        ],
    ),

    # ---- Loss (3 preview problems) ----
    Decision(
        name="preview_loss_protect_gains",
        category="Loss",
        title="Protect Your Gains",
        description=(
            "You currently have $12.00 in hand.\n\n"
            "Option A: Keep your $12.00. No risk.\n"
            "Option B: A gamble -- 50% chance your total becomes $18.00 "
            "(gain $6), but a 50% chance your total becomes $6.00 "
            "(lose $6).\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: Keep $12.00", [Outcome(1.0, 12.00)]),
            Option("Option B: Gamble (50% $18 / 50% $6)", [
                Outcome(0.5, 18.00), Outcome(0.5, 6.00),
            ]),
        ],
    ),
    Decision(
        name="preview_loss_small_downside",
        category="Loss",
        title="Small Downside Risk",
        description=(
            "You currently have $6.00 in hand.\n\n"
            "Option A: Keep your $6.00. No risk.\n"
            "Option B: A gamble -- 70% chance your total becomes $10.00 "
            "(gain $4), but a 30% chance your total becomes $2.00 "
            "(lose $4).\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: Keep $6.00", [Outcome(1.0, 6.00)]),
            Option("Option B: Gamble (70% $10 / 30% $2)", [
                Outcome(0.7, 10.00), Outcome(0.3, 2.00),
            ]),
        ],
    ),
    Decision(
        name="preview_loss_cut_or_hold",
        category="Loss",
        title="Cut Your Losses",
        description=(
            "You started with $15.00 but have already lost $5.00, "
            "leaving you with $10.00.\n\n"
            "Option A: Walk away with your $10.00.\n"
            "Option B: A gamble -- 45% chance you recover to $15.00, "
            "but a 55% chance you drop to $4.00.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: Walk away with $10.00", [Outcome(1.0, 10.00)]),
            Option("Option B: Gamble (45% $15 / 55% $4)", [
                Outcome(0.45, 15.00), Outcome(0.55, 4.00),
            ]),
        ],
    ),

    # ---- Insurance (3 preview problems) ----
    Decision(
        name="preview_ins_equipment",
        category="Insurance",
        title="Equipment Protection",
        description=(
            "You have equipment worth $25.00. There is a 10% chance "
            "of damage that would cost you $20.00 to repair.\n\n"
            "Option A: Buy a protection plan for $3.50. You are "
            "guaranteed to keep at least $21.50 no matter what.\n"
            "Option B: No protection. 90% chance you keep $25.00, "
            "10% chance you end up with $5.00.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: Buy protection (keep $21.50)", [
                Outcome(1.0, 21.50),
            ]),
            Option("Option B: No protection (90% $25 / 10% $5)", [
                Outcome(0.9, 25.00), Outcome(0.1, 5.00),
            ]),
        ],
    ),
    Decision(
        name="preview_ins_weather",
        category="Insurance",
        title="Weather Insurance",
        description=(
            "You are planning an outdoor event that will earn $16.00. "
            "There is a 25% chance of bad weather, which would reduce "
            "your earnings to $4.00.\n\n"
            "Option A: Buy weather insurance for $4.00. You are "
            "guaranteed to end with $12.00 regardless of weather.\n"
            "Option B: No insurance. 75% chance you earn $16.00, "
            "25% chance you earn $4.00.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: Buy insurance (keep $12.00)", [
                Outcome(1.0, 12.00),
            ]),
            Option("Option B: No insurance (75% $16 / 25% $4)", [
                Outcome(0.75, 16.00), Outcome(0.25, 4.00),
            ]),
        ],
    ),
    Decision(
        name="preview_ins_rare_disaster",
        category="Insurance",
        title="Rare Disaster Coverage",
        description=(
            "You have $30.00. There is a 5% chance of a disaster that "
            "would cost you $25.00.\n\n"
            "Option A: Buy insurance for $2.00. You are guaranteed to "
            "keep $28.00 regardless of what happens.\n"
            "Option B: No insurance. 95% chance you keep $30.00, "
            "5% chance you end up with $5.00.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: Buy insurance (keep $28.00)", [
                Outcome(1.0, 28.00),
            ]),
            Option("Option B: No insurance (95% $30 / 5% $5)", [
                Outcome(0.95, 30.00), Outcome(0.05, 5.00),
            ]),
        ],
    ),

    # ---- Ambiguity (3 preview problems) ----
    Decision(
        name="preview_amb_cards",
        category="Ambiguity",
        title="Known vs. Unknown Deck",
        description=(
            "Two decks of 20 cards each contain red and black cards. "
            "You will draw one card. If it is red, you win $8.00. "
            "If black, you win $0.\n\n"
            "Deck A: You know it has exactly 10 red and 10 black cards.\n"
            "Deck B: It has some mix of red and black, but you do not "
            "know the ratio.\n\n"
            "Which deck do you draw from?"
        ),
        options=[
            Option("Deck A: Known 50/50", [
                Outcome(0.5, 8.00), Outcome(0.5, 0.00),
            ]),
            Option("Deck B: Unknown ratio", [
                Outcome(0.5, 8.00), Outcome(0.5, 0.00),
            ]),
        ],
    ),
    Decision(
        name="preview_amb_bonus",
        category="Ambiguity",
        title="Bonus Wheel",
        description=(
            "Two spinning wheels determine a bonus payment.\n\n"
            "Wheel A: You can see it has a 40% green zone ($12.00) "
            "and a 60% white zone ($0). The odds are printed clearly.\n"
            "Wheel B: It has a green zone ($12.00) and a white zone ($0), "
            "but the zones are covered and you cannot see how large "
            "each one is.\n\n"
            "Which wheel do you spin?"
        ),
        options=[
            Option("Wheel A: Known 40% chance of $12", [
                Outcome(0.4, 12.00), Outcome(0.6, 0.00),
            ]),
            Option("Wheel B: Unknown chance of $12", [
                Outcome(0.4, 12.00), Outcome(0.6, 0.00),
            ]),
        ],
    ),
    Decision(
        name="preview_amb_jar",
        category="Ambiguity",
        title="Mystery Jar",
        description=(
            "Two jars contain gold and silver coins.\n\n"
            "Jar A: Contains exactly 30 gold and 70 silver coins. "
            "Drawing gold wins $15.00; silver wins $2.00.\n"
            "Jar B: Contains some mix of gold and silver coins "
            "(total 100), but the ratio is unknown. Same prizes.\n\n"
            "Which jar do you draw from?"
        ),
        options=[
            Option("Jar A: Known 30/70 gold/silver", [
                Outcome(0.3, 15.00), Outcome(0.7, 2.00),
            ]),
            Option("Jar B: Unknown ratio", [
                Outcome(0.3, 15.00), Outcome(0.7, 2.00),
            ]),
        ],
    ),

    # ---- Investment (3 preview problems) ----
    Decision(
        name="preview_inv_startup",
        category="Investment",
        title="Startup vs. Bonds",
        description=(
            "You have $8.00 to invest in one of three options.\n\n"
            "Option A (Bonds): Guaranteed return -- you end with $9.50.\n"
            "Option B (Index Fund): 60% chance you end with $14.00, "
            "40% chance you end with $5.00.\n"
            "Option C (Startup): 15% chance you end with $40.00, "
            "85% chance you end with $3.00.\n\n"
            "Which do you choose?"
        ),
        options=[
            Option("Option A: Bonds ($9.50 guaranteed)", [
                Outcome(1.0, 9.50),
            ]),
            Option("Option B: Index Fund (60/40 $14 or $5)", [
                Outcome(0.6, 14.00), Outcome(0.4, 5.00),
            ]),
            Option("Option C: Startup (15/85 $40 or $3)", [
                Outcome(0.15, 40.00), Outcome(0.85, 3.00),
            ]),
        ],
    ),
    Decision(
        name="preview_inv_real_estate",
        category="Investment",
        title="Property Investment",
        description=(
            "You have $15.00 to invest in one of three properties.\n\n"
            "Property A (Rental): Steady income -- you end with $17.00.\n"
            "Property B (Flip): 50% chance you end with $24.00, "
            "50% chance you end with $11.00.\n"
            "Property C (Development): 25% chance you end with $40.00, "
            "75% chance you end with $8.00.\n\n"
            "Which property do you invest in?"
        ),
        options=[
            Option("Property A: Rental ($17.00 guaranteed)", [
                Outcome(1.0, 17.00),
            ]),
            Option("Property B: Flip (50/50 $24 or $11)", [
                Outcome(0.5, 24.00), Outcome(0.5, 11.00),
            ]),
            Option("Property C: Development (25/75 $40 or $8)", [
                Outcome(0.25, 40.00), Outcome(0.75, 8.00),
            ]),
        ],
    ),
    Decision(
        name="preview_inv_portfolio",
        category="Investment",
        title="Portfolio Mix",
        description=(
            "You have $12.00 to allocate to one strategy.\n\n"
            "Strategy A (Conservative): You end with $13.50 guaranteed.\n"
            "Strategy B (Balanced): 55% chance you end with $20.00, "
            "45% chance you end with $8.00.\n"
            "Strategy C (Growth): 35% chance you end with $28.00, "
            "65% chance you end with $6.00.\n\n"
            "Which strategy do you choose?"
        ),
        options=[
            Option("Strategy A: Conservative ($13.50 guaranteed)", [
                Outcome(1.0, 13.50),
            ]),
            Option("Strategy B: Balanced (55/45 $20 or $8)", [
                Outcome(0.55, 20.00), Outcome(0.45, 8.00),
            ]),
            Option("Strategy C: Growth (35/65 $28 or $6)", [
                Outcome(0.35, 28.00), Outcome(0.65, 6.00),
            ]),
        ],
    ),
]


# ------------------------------------------------------------------
# Preview helpers
# ------------------------------------------------------------------

def sample_preview_problems(n: int = 5) -> list[Decision]:
    """Sample n preview problems with round-robin category coverage.

    Ensures at least one problem from each category (up to n), then
    fills remaining slots randomly from unused problems.
    """
    by_category: dict[str, list[Decision]] = {}
    for d in PREVIEW_DECISIONS:
        by_category.setdefault(d.category, []).append(d)

    selected: list[Decision] = []
    categories = list(by_category.keys())
    random.shuffle(categories)

    # Round-robin: one from each category
    for cat in categories:
        if len(selected) >= n:
            break
        pick = random.choice(by_category[cat])
        selected.append(pick)

    # Fill remaining slots from unused problems
    if len(selected) < n:
        used_names = {d.name for d in selected}
        remaining = [d for d in PREVIEW_DECISIONS if d.name not in used_names]
        random.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    random.shuffle(selected)
    return selected


def build_preview_survey(problems: list[Decision]) -> Survey:
    """Build an EDSL survey from a subset of preview problems."""
    questions = [d.to_question() for d in problems]
    return Survey(questions)


def simulate_preview(
    choices: dict[str, str],
    problems: list[Decision],
    n_sims: int = 500,
) -> dict:
    """Simulate payoffs for an arbitrary list of decisions.

    Same return shape as simulate_session:
      - per_problem: [{name, title, category, chosen, ev, simulated_payoffs}, ...]
      - total_payoffs: [sum for each of n_sims simulations]
    """
    per_problem = []
    total_payoffs = [0.0] * n_sims

    for decision in problems:
        chosen_label = choices.get(decision.name)
        if chosen_label is None:
            per_problem.append({
                "name": decision.name,
                "title": decision.title,
                "category": decision.category,
                "chosen": "No answer",
                "ev": 0,
                "simulated_payoffs": [0.0] * n_sims,
            })
            continue

        # Find the chosen option
        chosen_option = None
        for opt in decision.options:
            if opt.label == chosen_label:
                chosen_option = opt
                break

        if chosen_option is None:
            for opt in decision.options:
                if chosen_label in opt.label or opt.label in chosen_label:
                    chosen_option = opt
                    break

        if chosen_option is None:
            chosen_option = decision.options[0]

        payoffs = [chosen_option.simulate() for _ in range(n_sims)]
        for i in range(n_sims):
            total_payoffs[i] += payoffs[i]

        per_problem.append({
            "name": decision.name,
            "title": decision.title,
            "category": decision.category,
            "chosen": chosen_label,
            "ev": chosen_option.expected_value,
            "simulated_payoffs": payoffs,
        })

    return {
        "per_problem": per_problem,
        "total_payoffs": total_payoffs,
    }
