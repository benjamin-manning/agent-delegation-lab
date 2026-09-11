"""Test banks: questions participants can pay to test agents on.

Two kinds of bank:
  - Category banks hold questions of the same kinds as the targets, with
    different amounts and odds. Near copies of any target are left out.
  - Near-copy banks hold small variations of one target decision.

Questions are built from families. A family is a question type with
parameters and a render function. Each render function reproduces the
wording of the targets exactly, so TARGET_PARAMS rebuilds every target.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from decisions import CATEGORY_DESCRIPTIONS, DECISIONS, Decision, Option, Outcome

CATEGORY_BANK_SIZE = 20
NEAR_BANK_SIZE = 8
MIN_EV_RATIO = 0.6  # lowest option EV / highest option EV, category banks only


# ------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------

def _money(x: float) -> str:
    """$5.00 style; zero is written $0."""
    return "$0" if x == 0 else f"${x:.2f}"


def _short(x: float) -> str:
    """$5 for whole dollars, $4.50 otherwise."""
    return f"${x:.0f}" if x == int(x) else f"${x:.2f}"


def _pct(p: float) -> str:
    return f"{round(p * 100)}%"


def _a(p: float) -> str:
    """Article for a percentage: "an 85%", "a 15%"."""
    return "an" if _pct(p)[0] == "8" or _pct(p)[:2] in ("11", "18") else "a"


def _half(x: float) -> float:
    """Round to the nearest $0.50."""
    return round(x * 2) / 2


# ------------------------------------------------------------------
# Families: params -> (description, options, title)
# ------------------------------------------------------------------

def _sure_vs_gamble(sure, p, high, low):
    if p == 0.5:
        b_text = (
            f"Option B: A coin flip -- 50% chance of receiving {_money(high)}, "
            f"50% chance of receiving {_money(low)}."
        )
        b_label = f"Option B: 50/50 for {_money(high)} or {_money(low)}"
    else:
        b_text = (
            f"Option B: {_a(p).capitalize()} {_pct(p)} chance of receiving {_money(high)}, "
            f"and {_a(1 - p)} {_pct(1 - p)} chance of receiving {_money(low)}."
        )
        b_label = f"Option B: {_pct(p)} chance of {_money(high)}"
        if low:
            b_label += f", else {_money(low)}"
    text = (
        "You face a choice between two options.\n\n"
        f"Option A: Receive {_money(sure)} for certain.\n"
        f"{b_text}\n\n"
        "Which do you choose?"
    )
    options = [
        Option(f"Option A: {_money(sure)} for certain", [Outcome(1.0, sure)]),
        Option(b_label, [Outcome(p, high), Outcome(1 - p, low)]),
    ]
    title = f"{_money(sure)} sure or {_pct(p)} chance of {_money(high)}"
    return text, options, title


def _sure_vs_thirds(sure, x1, x2, x3):
    text = (
        "You face a choice between two options.\n\n"
        f"Option A: Receive {_money(sure)} for certain.\n"
        "Option B: Equal chances (one-third each) of receiving "
        f"{_money(x1)}, {_money(x2)}, or {_money(x3)}.\n\n"
        "Which do you choose?"
    )
    options = [
        Option(f"Option A: {_money(sure)} for certain", [Outcome(1.0, sure)]),
        Option(
            f"Option B: 1/3 each of {_short(x1)}, {_short(x2)}, or {_short(x3)}",
            [Outcome(1 / 3, x1), Outcome(1 / 3, x2), Outcome(1 / 3, x3)],
        ),
    ]
    title = f"{_money(sure)} sure or thirds of {_short(x1)}, {_short(x2)}, {_short(x3)}"
    return text, options, title


def _gamble_vs_gamble(pa, ha, la, pb, hb, lb):
    text = (
        "You face a choice between two options.\n\n"
        f"Option A: {_pct(pa)} chance of {_money(ha)}, "
        f"{_pct(1 - pa)} chance of {_money(la)}.\n"
        f"Option B: {_pct(pb)} chance of {_money(hb)}, "
        f"{_pct(1 - pb)} chance of {_money(lb)}.\n\n"
        "Which do you choose?"
    )
    options = [
        Option(
            f"Option A: {_pct(pa)} {_short(ha)} / {_pct(1 - pa)} {_short(la)}",
            [Outcome(pa, ha), Outcome(1 - pa, la)],
        ),
        Option(
            f"Option B: {_pct(pb)} {_short(hb)} / {_pct(1 - pb)} {_short(lb)}",
            [Outcome(pb, hb), Outcome(1 - pb, lb)],
        ),
    ]
    title = f"{_pct(pa)} chance of {_money(ha)} or {_pct(pb)} chance of {_money(hb)}"
    return text, options, title


def _keep_or_gamble(have, p, gain, loss):
    up, down = have + gain, have - loss
    text = (
        f"You currently have {_money(have)} in hand.\n\n"
        f"Option A: Keep your {_money(have)}. No risk.\n"
        f"Option B: A gamble -- {_pct(p)} chance your total becomes {_money(up)} "
        f"(gain {_short(gain)}), but {_a(1 - p)} {_pct(1 - p)} chance your total becomes "
        f"{_money(down)} (lose {_short(loss)}).\n\n"
        "Which do you choose?"
    )
    options = [
        Option(f"Option A: Keep {_money(have)}", [Outcome(1.0, have)]),
        Option(
            f"Option B: Gamble ({_pct(p)} {_short(up)} / {_pct(1 - p)} {_short(down)})",
            [Outcome(p, up), Outcome(1 - p, down)],
        ),
    ]
    title = f"Keep {_money(have)} or a {_pct(p)} chance to gain {_short(gain)}"
    return text, options, title


def _insurance(wealth, q, loss, premium):
    insured, hit = wealth - premium, wealth - loss
    text = (
        f"You have {_money(wealth)}. There is {_a(q)} {_pct(q)} chance that an accident "
        f"occurs and you lose {_money(loss)}.\n\n"
        f"Option A: Buy insurance for {_money(premium)}. You are guaranteed to "
        f"keep {_money(insured)} regardless of what happens.\n"
        f"Option B: No insurance. {_pct(1 - q)} chance you keep {_money(wealth)}, "
        f"{_pct(q)} chance you end up with {_money(hit)}.\n\n"
        "Which do you choose?"
    )
    options = [
        Option(f"Option A: Buy insurance (keep {_money(insured)})", [Outcome(1.0, insured)]),
        Option(
            f"Option B: No insurance ({_pct(1 - q)} {_short(wealth)} / {_pct(q)} {_short(hit)})",
            [Outcome(1 - q, wealth), Outcome(q, hit)],
        ),
    ]
    title = f"Insure {_money(wealth)} for {_money(premium)} against a {_pct(q)} loss"
    return text, options, title


def _ambiguity(prize, red):
    blue = 100 - red
    text = (
        "Two urns each contain 100 balls, red and blue. "
        f"You will draw one ball. If it is red, you win {_money(prize)}. "
        "If blue, you win $0.\n\n"
        f"Urn A: You know it contains exactly {red} red and {blue} blue balls.\n"
        "Urn B: It contains some mix of red and blue balls, but you "
        "do not know the ratio.\n\n"
        "Which urn do you draw from?"
    )
    # The unknown urn is simulated at the known urn's odds.
    options = [
        Option(f"Urn A: Known {red}/{blue}", [
            Outcome(red / 100, prize, "Red ball"), Outcome(blue / 100, 0, "Blue ball"),
        ]),
        Option("Urn B: Unknown ratio", [
            Outcome(red / 100, prize, "Red ball"), Outcome(blue / 100, 0, "Blue ball"),
        ]),
    ]
    title = f"Known {red}/{blue} urn or unknown urn for {_money(prize)}"
    return text, options, title


def _investment(stake, safe, pb, hb, lb, pc, hc, lc):
    def odds(p, h, l):
        return f"{round(p * 100)}/{round((1 - p) * 100)} {_short(h)} or {_short(l)}"

    text = (
        f"You have {_money(stake)} to invest in one of three funds.\n\n"
        f"Fund A (Safe): Guaranteed return -- you end with {_money(safe)}.\n"
        f"Fund B (Moderate): {_pct(pb)} chance you end with {_money(hb)}, "
        f"{_pct(1 - pb)} chance you end with {_money(lb)}.\n"
        f"Fund C (Aggressive): {_pct(pc)} chance you end with {_money(hc)}, "
        f"{_pct(1 - pc)} chance you end with {_money(lc)}.\n\n"
        "Which fund do you choose?"
    )
    options = [
        Option(f"Fund A: Safe ({_money(safe)} guaranteed)", [Outcome(1.0, safe)]),
        Option(f"Fund B: Moderate ({odds(pb, hb, lb)})", [Outcome(pb, hb), Outcome(1 - pb, lb)]),
        Option(f"Fund C: Aggressive ({odds(pc, hc, lc)})", [Outcome(pc, hc), Outcome(1 - pc, lc)]),
    ]
    title = f"Invest {_money(stake)}, safe fund pays {_money(safe)}"
    return text, options, title


_LETTERS = "ABCDEFGH"
_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _join(parts: list[str]) -> str:
    """'a and b', or 'a, b, and c'."""
    if len(parts) < 3:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _multi_lottery(lotteries):
    """Several lotteries, each a list of (probability, amount)."""
    lines, options = [], []
    for letter, lottery in zip(_LETTERS, lotteries):
        chances = ", ".join(f"{_pct(p)} chance of {_money(x)}" for p, x in lottery)
        lines.append(f"Lottery {letter}: {chances}.")
        options.append(Option(
            f"Lottery {letter}: " + " / ".join(f"{_pct(p)} {_short(x)}" for p, x in lottery),
            [Outcome(p, x) for p, x in lottery],
        ))
    text = (
        f"You face a choice between {_NUMBER_WORDS[len(lotteries)]} lotteries.\n\n"
        + "\n".join(lines)
        + "\n\nWhich do you choose?"
    )
    tops = " and ".join(_short(max(x for _, x in lottery)) for lottery in lotteries)
    return text, options, f"Lotteries with top prizes {tops}"


def _coin_menu(pairs):
    """A coin flip; each gamble pays (heads, tails)."""
    lines = [
        f"Gamble {i}: {_money(h)} if heads, {_money(t)} if tails."
        for i, (h, t) in enumerate(pairs, 1)
    ]
    text = (
        f"A coin will be flipped. Choose one of {_NUMBER_WORDS[len(pairs)]} gambles. "
        "Each gamble pays one amount if the coin lands heads and another amount "
        "if it lands tails.\n\n"
        + "\n".join(lines)
        + "\n\nWhich gamble do you choose?"
    )
    options = [
        Option(
            f"Gamble {i}: {_short(h)} heads / {_short(t)} tails",
            [Outcome(0.5, h, "Heads"), Outcome(0.5, t, "Tails")],
        )
        for i, (h, t) in enumerate(pairs, 1)
    ]
    top = max(h for h, _ in pairs)
    return text, options, f"{len(pairs)} coin flip gambles, up to {_money(top)}"


def _insurance_plans(wealth, states, plans):
    """states: [(label, probability, loss)], the first with no loss.
    plans: [(name, premium, deductible)]; deductible None means no insurance
    and 0 means full coverage."""
    (none_label, none_p, _), *risky = states
    chances = [f"{_a(none_p)} {_pct(none_p)} chance of {none_label.lower()}"] + [
        f"{_a(p)} {_pct(p)} chance of a {label.lower()} that costs you {_money(loss)}"
        for label, p, loss in risky
    ]
    lines, options = [], []
    for letter, (name, premium, deductible) in zip(_LETTERS, plans):
        cost = "Costs nothing." if premium == 0 else f"Costs {_money(premium)}."
        if deductible is None:
            cover = "You pay for any accident yourself."
        elif deductible == 0:
            cover = "The plan pays for any accident."
        else:
            cover = (
                f"You pay the first {_money(deductible)} of any accident, "
                "and the plan pays the rest."
            )
        lines.append(f"Plan {letter} ({name}): {cost} {cover}")
        options.append(Option(
            f"Plan {letter}: {name}" + (f" ({_money(premium)})" if premium else ""),
            [
                Outcome(p, wealth - premium - (loss if deductible is None else min(loss, deductible)), label)
                for label, p, loss in states
            ],
        ))
    text = (
        f"You have {_money(wealth)}. This year there is {_join(chances)}.\n\n"
        + "\n".join(lines)
        + "\n\nWhich plan do you choose?"
    )
    full = max(premium for _, premium, _ in plans)
    return text, options, f"Insurance plans for {_money(wealth)}, full cover {_money(full)}"


def _three_color_urn(total, red, single, pair):
    other = total - red
    p_red = red / total
    p_each = other / 2 / total  # the unknown black/yellow split is simulated as even
    text = (
        f"An urn contains {total} balls. {red} of them are red. The other {other} "
        "are black and yellow, but you do not know how many are black and how "
        "many are yellow. You will draw one ball.\n\n"
        f"Bet A: You win {_money(single)} if the ball is red.\n"
        f"Bet B: You win {_money(single)} if the ball is black.\n"
        f"Bet C: You win {_money(pair)} if the ball is black or yellow.\n"
        f"Bet D: You win {_money(pair)} if the ball is red or yellow.\n\n"
        "Which bet do you choose?"
    )
    # Known-odds bets come first in each pair, so ties in payoff spread count
    # them as the safest option.
    options = [
        Option(f"Bet A: {_short(single)} if red", [
            Outcome(p_red, single, "Red ball"), Outcome(1 - p_red, 0, "Black or yellow ball"),
        ]),
        Option(f"Bet B: {_short(single)} if black", [
            Outcome(p_each, single, "Black ball"), Outcome(1 - p_each, 0, "Red or yellow ball"),
        ]),
        Option(f"Bet C: {_short(pair)} if black or yellow", [
            Outcome(1 - p_red, pair, "Black or yellow ball"), Outcome(p_red, 0, "Red ball"),
        ]),
        Option(f"Bet D: {_short(pair)} if red or yellow", [
            Outcome(p_red + p_each, pair, "Red or yellow ball"), Outcome(p_each, 0, "Black ball"),
        ]),
    ]
    return text, options, f"Three-color urn, {red} red of {total}"


def _market_scenarios(stake, states, funds):
    """states: [(label, phrase, probability)], e.g. ("Boom", "a boom", 0.25).
    funds: [(name, [amount in each state])]."""
    chances = _join([f"{_a(p)} {_pct(p)} chance of {phrase}" for _, phrase, p in states])
    lines, options = [], []
    for letter, (name, amounts) in zip(_LETTERS, funds):
        if len(set(amounts)) == 1:
            detail = f"You end with {_money(amounts[0])} in every case."
            summary = f"{_short(amounts[0])} always"
        else:
            detail = "You end with " + _join([
                f"{_money(x)} in {phrase}" for x, (_, phrase, _) in zip(amounts, states)
            ]) + "."
            summary = " / ".join(_short(x) for x in amounts)
        lines.append(f"Fund {letter} ({name}): {detail}")
        options.append(Option(
            f"Fund {letter}: {name} ({summary})",
            [Outcome(p, x, label) for x, (label, _, p) in zip(amounts, states)],
        ))
    n = _NUMBER_WORDS[len(funds)]
    text = (
        f"You have {_money(stake)} to invest in one of {n} funds. Next year there "
        f"is {chances}. What you end with depends on the fund and on the market.\n\n"
        + "\n".join(lines)
        + "\n\nWhich fund do you choose?"
    )
    return text, options, f"{n.capitalize()} funds for {_money(stake)} across market outcomes"


FAMILIES = {
    "sure_vs_gamble": _sure_vs_gamble,
    "sure_vs_thirds": _sure_vs_thirds,
    "gamble_vs_gamble": _gamble_vs_gamble,
    "keep_or_gamble": _keep_or_gamble,
    "insurance": _insurance,
    "ambiguity": _ambiguity,
    "investment": _investment,
    "multi_lottery": _multi_lottery,
    "coin_menu": _coin_menu,
    "insurance_plans": _insurance_plans,
    "three_color_urn": _three_color_urn,
    "market_scenarios": _market_scenarios,
}

CATEGORY_FAMILIES = {
    "Risk": [
        "sure_vs_gamble", "gamble_vs_gamble", "sure_vs_thirds",
        "multi_lottery", "multi_lottery", "coin_menu", "coin_menu",
    ],
    "Loss": ["keep_or_gamble"],
    "Insurance": ["insurance", "insurance_plans", "insurance_plans"],
    "Ambiguity": ["ambiguity", "three_color_urn", "three_color_urn"],
    "Investment": ["investment", "market_scenarios", "market_scenarios"],
}

PROB_PARAMS = {"p", "q", "pa", "pb", "pc"}

# Family, parameters, and title that build each target in decisions.DECISIONS.
TARGET_PARAMS = {
    "safe_vs_risky": ("sure_vs_gamble", dict(sure=5, p=0.5, high=12, low=0)),
    "long_shot": ("sure_vs_gamble", dict(sure=3, p=0.05, high=80, low=0)),
    "loss_frame": ("keep_or_gamble", dict(have=8, p=0.6, gain=5, loss=5)),
    "two_lotteries": ("multi_lottery", dict(lotteries=[
        [(0.10, 0), (0.20, 4), (0.40, 6), (0.30, 7)],
        [(0.30, 3), (0.40, 4), (0.20, 6), (0.10, 17)],
    ])),
    "coin_menu": ("coin_menu", dict(
        pairs=[(6, 6), (8, 5), (10, 4), (12, 3), (14, 2), (15, 0)],
    )),
    "insurance_plans": ("insurance_plans", dict(
        wealth=20,
        states=[("No accident", 0.80, 0), ("Minor accident", 0.15, 6), ("Major accident", 0.05, 18)],
        plans=[
            ("No insurance", 0, None), ("High deductible", 0.5, 6),
            ("Low deductible", 1.5, 2), ("Full coverage", 3, 0),
        ],
    )),
    "three_color_urn": ("three_color_urn", dict(total=90, red=30, single=15, pair=7.5)),
    "market_scenarios": ("market_scenarios", dict(
        stake=10,
        states=[("Boom", "a boom", 0.25), ("Normal year", "a normal year", 0.50), ("Recession", "a recession", 0.25)],
        funds=[("Cash", [10.5] * 3), ("Bonds", [13, 11.5, 9]), ("Stocks", [18, 12, 5]), ("Startup", [34, 8, 0])],
    )),
}


def build_decision(
    name: str, category: str, family: str, params: dict, title: str | None = None,
) -> Decision:
    text, options, generated_title = FAMILIES[family](**params)
    return Decision(
        name=name, category=category, title=title or generated_title,
        description=text, options=options,
    )


def _valid(family: str, v: dict) -> bool:
    """Ordering checks so every generated question makes sense."""
    if family == "sure_vs_gamble":
        return v["low"] < v["sure"] < v["high"]
    if family == "sure_vs_thirds":
        return v["x3"] < v["x2"] < v["x1"] and v["x3"] < v["sure"] < v["x1"]
    if family == "gamble_vs_gamble":
        return v["la"] < v["ha"] and v["lb"] < v["hb"] and v["ha"] < v["hb"]
    if family == "keep_or_gamble":
        return v["gain"] > 0 and 0 < v["loss"] < v["have"]
    if family == "insurance":
        return 0 < v["premium"] < v["loss"] < v["wealth"]
    if family == "ambiguity":
        return v["prize"] > 0 and 0 < v["red"] < 100
    if family == "investment":
        return v["lc"] < v["lb"] < v["safe"] < v["hb"] < v["hc"]
    if family == "multi_lottery":
        return all(
            abs(sum(p for p, _ in lot) - 1) < 1e-9
            and all(p >= 0.05 - 1e-9 for p, _ in lot)
            and all(x0 < x1 for (_, x0), (_, x1) in zip(lot, lot[1:]))
            and lot[0][1] >= 0
            for lot in v["lotteries"]
        ) and len({tuple(lot) for lot in v["lotteries"]}) == len(v["lotteries"])
    if family == "coin_menu":
        pairs = v["pairs"]
        return (
            all(t >= 0 for _, t in pairs)
            and all(h0 < h1 and t0 >= t1 for (h0, t0), (h1, t1) in zip(pairs, pairs[1:]))
            and pairs[0][0] >= pairs[0][1]
        )
    if family == "insurance_plans":
        probs = [p for _, p, _ in v["states"]]
        losses = [loss for _, _, loss in v["states"]]
        premiums = [prem for _, prem, _ in v["plans"]]
        return (
            abs(sum(probs) - 1) < 1e-9 and min(probs) >= 0.05 - 1e-9
            and losses[0] == 0 and all(a < b for a, b in zip(losses, losses[1:]))
            and losses[-1] < v["wealth"]
            and premiums[0] == 0 and all(a < b for a, b in zip(premiums, premiums[1:]))
            and v["wealth"] - premiums[-1] > 0
        )
    if family == "three_color_urn":
        return 0 < v["red"] < v["total"] and 0 < v["pair"] < v["single"]
    if family == "market_scenarios":
        probs = [p for _, _, p in v["states"]]
        first = [amounts[0] for _, amounts in v["funds"]]
        last = [amounts[-1] for _, amounts in v["funds"]]
        return (
            abs(sum(probs) - 1) < 1e-9 and min(probs) >= 0.05 - 1e-9
            and all(a < b for a, b in zip(first, first[1:]))
            and all(a >= b for a, b in zip(last, last[1:]))
            and min(last) >= 0
        )
    raise ValueError(family)


# ------------------------------------------------------------------
# Comparing questions
# ------------------------------------------------------------------

def is_near_copy(a: Decision, b: Decision) -> bool:
    """Same shape, every probability within 0.10, every payoff within 25% or $0.50."""
    if len(a.options) != len(b.options):
        return False
    for oa, ob in zip(a.options, b.options):
        if len(oa.outcomes) != len(ob.outcomes):
            return False
        for xa, xb in zip(oa.outcomes, ob.outcomes):
            if abs(xa.probability - xb.probability) > 0.10 + 1e-9:
                return False
            tol = max(0.25 * max(abs(xa.payoff), abs(xb.payoff)), 0.5)
            if abs(xa.payoff - xb.payoff) > tol + 1e-9:
                return False
    return True


def same_outcomes(a: Decision, b: Decision) -> bool:
    if len(a.options) != len(b.options):
        return False
    for oa, ob in zip(a.options, b.options):
        if len(oa.outcomes) != len(ob.outcomes):
            return False
        for xa, xb in zip(oa.outcomes, ob.outcomes):
            if abs(xa.probability - xb.probability) > 1e-9 or abs(xa.payoff - xb.payoff) > 1e-9:
                return False
    return True


def _ev_ratio(d: Decision) -> float:
    evs = [opt.expected_value for opt in d.options]
    return min(evs) / max(evs)


# ------------------------------------------------------------------
# Hand-written practice questions (seed the category banks)
# ------------------------------------------------------------------

HANDWRITTEN: list[Decision] = [
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
# Generation
# ------------------------------------------------------------------

def _sample_params(family: str, rng: random.Random) -> dict:
    def pct(lo, hi):
        return rng.randrange(lo, hi + 1, 5) / 100

    def money(lo, hi):
        return _half(rng.uniform(lo, hi))

    if family == "sure_vs_gamble":
        p = pct(5, 90)
        low = rng.choice([0, 0, 0, 0.5, 1, 2])
        high = money(low + 4, min(80, max(10, 8 / p)))
        ev = p * high + (1 - p) * low
        return dict(sure=money(0.6 * ev, 1.3 * ev), p=p, high=high, low=low)
    if family == "sure_vs_thirds":
        x1 = money(8, 30)
        x2 = money(1, x1 / 2)
        x3 = rng.choice([0, 0, 0.5, 1])
        ev = (x1 + x2 + x3) / 3
        return dict(sure=money(0.6 * ev, 1.3 * ev), x1=x1, x2=x2, x3=x3)
    if family == "gamble_vs_gamble":
        ha = money(2, 10)
        return dict(
            pa=pct(55, 90), ha=ha, la=money(0, ha - 1.5),
            pb=pct(10, 50), hb=money(ha + 3, 30), lb=rng.choice([0, 0, 0.5, 1]),
        )
    if family == "keep_or_gamble":
        have = money(4, 20)
        return dict(have=have, p=pct(30, 80), gain=money(1, have), loss=money(1, have - 0.5))
    if family == "insurance":
        wealth = money(10, 40)
        q = pct(5, 30)
        loss = money(0.3 * wealth, 0.9 * wealth)
        premium = max(0.5, money(0.7 * q * loss, 1.8 * q * loss))
        return dict(wealth=wealth, q=q, loss=loss, premium=premium)
    if family == "ambiguity":
        return dict(prize=money(4, 25), red=rng.randrange(10, 91, 5))
    if family == "investment":
        s = money(5, 20)
        return dict(
            stake=s, safe=money(1.05 * s, 1.25 * s),
            pb=pct(40, 65), hb=money(1.3 * s, 2 * s), lb=money(0.4 * s, 0.9 * s),
            pc=pct(10, 40), hc=money(2.2 * s, 4.5 * s), lc=money(0.1 * s, 0.6 * s),
        )
    if family == "multi_lottery":
        def lottery():
            k = rng.choice([3, 4])
            cuts = sorted(rng.sample(range(5, 100, 5), k - 1))
            probs = [(b - a) / 100 for a, b in zip([0] + cuts, cuts + [100])]
            amounts = sorted(x / 2 for x in rng.sample(range(0, 61), k))
            return list(zip(probs, amounts))

        first, second = lottery(), lottery()
        ev_first = sum(p * x for p, x in first)
        ev_second = sum(p * x for p, x in second)
        if ev_second > 0:  # scale the second lottery to a similar expected value
            scale = ev_first / ev_second * rng.uniform(0.9, 1.1)
            second = [(p, _half(x * scale)) for p, x in second]
        return dict(lotteries=[first, second])
    if family == "coin_menu":
        n = rng.choice([4, 5, 6])
        s, dh = money(3, 8), money(1, 3)
        dt = money(0.5, max(0.5, 0.6 * dh))
        pairs = [(s + i * dh, s - i * dt) for i in range(n)]
        if rng.random() < 0.5:
            # Last gamble: same expected value as the one before, more spread.
            h, t = pairs[-2]
            pairs[-1] = (h + t, 0)
        return dict(pairs=[(_half(h), _half(t)) for h, t in pairs])
    if family == "insurance_plans":
        w = money(12, 40)
        p_minor, p_major = pct(10, 25), pct(5, 10)
        minor, major = money(0.15 * w, 0.35 * w), money(0.6 * w, 0.9 * w)
        ded_high, ded_low = money(0.6 * minor, minor), money(0.15 * minor, 0.45 * minor)

        def covered(d):
            return p_minor * max(minor - d, 0) + p_major * max(major - d, 0)

        premiums = [max(0.5, money(covered(d), 1.7 * covered(d))) for d in (ded_high, ded_low, 0)]
        return dict(
            wealth=w,
            states=[
                ("No accident", round(1 - p_minor - p_major, 2), 0),
                ("Minor accident", p_minor, minor),
                ("Major accident", p_major, major),
            ],
            plans=[
                ("No insurance", 0, None),
                ("High deductible", premiums[0], ded_high),
                ("Low deductible", premiums[1], ded_low),
                ("Full coverage", premiums[2], 0),
            ],
        )
    if family == "three_color_urn":
        red = rng.randrange(15, 46, 5)
        single = money(8, 25)
        even = single * red / (90 - red)  # pair prize with the same expected value
        return dict(total=90, red=red, single=single, pair=money(0.8 * even, 1.25 * even))
    if family == "market_scenarios":
        s = money(5, 20)
        p_boom, p_rec = pct(15, 35), pct(15, 35)
        cash = money(1.02 * s, 1.08 * s)
        return dict(
            stake=s,
            states=[
                ("Boom", "a boom", p_boom),
                ("Normal year", "a normal year", round(1 - p_boom - p_rec, 2)),
                ("Recession", "a recession", p_rec),
            ],
            funds=[
                ("Cash", [cash] * 3),
                ("Bonds", [money(1.2 * s, 1.35 * s), money(1.08 * s, 1.18 * s), money(0.85 * s, 0.97 * s)]),
                ("Stocks", [money(1.5 * s, 2.0 * s), money(1.1 * s, 1.3 * s), money(0.4 * s, 0.7 * s)]),
                ("Startup", [money(2.5 * s, 4.0 * s), money(0.6 * s, 1.0 * s), money(0, 0.2 * s)]),
            ],
        )
    raise ValueError(family)


def _nudge(x: float, rng: random.Random, spread: float) -> float:
    """Scale an amount by up to +/- spread, rounded to $0.50. Zero stays zero."""
    return _half(x * rng.uniform(1 - spread, 1 + spread))


def _perturb(family: str, params: dict, rng: random.Random) -> dict:
    """A small random change to a target's parameters, for near-copy banks."""
    if family == "multi_lottery":
        lotteries = []
        for lottery in params["lotteries"]:
            probs = [p for p, _ in lottery]
            if rng.random() < 0.5:  # move 5% from one outcome to another
                i, j = rng.sample(range(len(probs)), 2)
                probs[i], probs[j] = round(probs[i] + 0.05, 2), round(probs[j] - 0.05, 2)
            lotteries.append([(p, _nudge(x, rng, 0.2)) for p, (_, x) in zip(probs, lottery)])
        return dict(lotteries=lotteries)
    if family == "coin_menu":
        return dict(pairs=[(_nudge(h, rng, 0.15), _nudge(t, rng, 0.15)) for h, t in params["pairs"]])
    if family == "insurance_plans":
        shift = rng.choice([-0.05, 0, 0.05])
        (l0, p0, x0), (l1, p1, x1), (l2, p2, x2) = params["states"]
        return dict(
            wealth=_nudge(params["wealth"], rng, 0.08),
            states=[
                (l0, round(p0 - shift, 2), x0),
                (l1, round(p1 + shift, 2), _nudge(x1, rng, 0.08)),
                (l2, p2, _nudge(x2, rng, 0.08)),
            ],
            plans=[
                (name, _nudge(prem, rng, 0.15) if prem else 0, _nudge(ded, rng, 0.1) if ded else ded)
                for name, prem, ded in params["plans"]
            ],
        )
    if family == "three_color_urn":
        return dict(
            total=params["total"],
            red=params["red"] + rng.choice([-5, 0, 5]),
            single=_nudge(params["single"], rng, 0.15),
            pair=_nudge(params["pair"], rng, 0.15),
        )
    if family == "market_scenarios":
        shift = rng.choice([-0.05, 0, 0.05])
        (lb, fb, pb), middle, (lr, fr, pr) = params["states"]
        funds = []
        for name, amounts in params["funds"]:
            if len(set(amounts)) == 1:
                funds.append((name, [_nudge(amounts[0], rng, 0.1)] * len(amounts)))
            else:
                funds.append((name, [_nudge(x, rng, 0.15) for x in amounts]))
        return dict(
            stake=params["stake"],
            states=[(lb, fb, round(pb + shift, 2)), middle, (lr, fr, round(pr - shift, 2))],
            funds=funds,
        )
    out = {}
    for k, v in params.items():
        if k in PROB_PARAMS:
            p = v + rng.choice([-0.10, -0.05, 0, 0.05, 0.10])
            out[k] = round(min(0.95, max(0.05, p)) * 20) / 20
        elif k == "red":
            out[k] = min(90, max(10, v + rng.choice([-10, -5, 0, 5, 10])))
        else:
            out[k] = _nudge(v, rng, 0.2)
    return out


@dataclass
class Bank:
    key: str
    name: str
    kind: str  # "category" or "near"
    decisions: list[Decision]


def _category_bank(category: str) -> Bank:
    key = f"cat_{category.lower()}"
    rng = random.Random(key)
    picked = [
        d for d in HANDWRITTEN
        if d.category == category and not any(is_near_copy(d, t) for t in DECISIONS)
    ]
    for _ in range(20000):
        if len(picked) >= CATEGORY_BANK_SIZE:
            break
        family = rng.choice(CATEGORY_FAMILIES[category])
        params = _sample_params(family, rng)
        if not _valid(family, params):
            continue
        d = build_decision("", category, family, params)
        if _ev_ratio(d) < MIN_EV_RATIO:
            continue
        if any(is_near_copy(d, other) for other in list(DECISIONS) + picked):
            continue
        picked.append(d)
    if len(picked) < CATEGORY_BANK_SIZE:
        raise RuntimeError(f"Could not fill bank {key}")
    decisions = [replace(d, name=f"{key}_{i:02d}") for i, d in enumerate(picked, 1)]
    return Bank(key, f"{category} questions", "category", decisions)


def _near_bank(target: Decision) -> Bank:
    key = f"near_{target.name}"
    rng = random.Random(key)
    family, base = TARGET_PARAMS[target.name]
    picked: list[Decision] = []
    for _ in range(20000):
        if len(picked) >= NEAR_BANK_SIZE:
            break
        params = _perturb(family, base, rng)
        if not _valid(family, params):
            continue
        d = build_decision("", target.category, family, params)
        if not is_near_copy(d, target) or same_outcomes(d, target):
            continue
        if any(same_outcomes(d, other) for other in picked):
            continue
        picked.append(d)
    if len(picked) < NEAR_BANK_SIZE:
        raise RuntimeError(f"Could not fill bank {key}")
    decisions = [replace(d, name=f"{key}_{i:02d}") for i, d in enumerate(picked, 1)]
    return Bank(key, f"Close variants of {target.title}", "near", decisions)


BANKS: list[Bank] = (
    [_category_bank(c) for c in CATEGORY_DESCRIPTIONS]
    + [_near_bank(t) for t in DECISIONS]
)
BANKS_BY_KEY = {b.key: b for b in BANKS}
