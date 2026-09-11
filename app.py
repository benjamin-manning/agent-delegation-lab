"""
Agent Delegation Lab
=====================
See the 8 decisions your agent will face, pay to test agents on practice
question banks, then commit to one agent. It answers each decision once and
each lottery is drawn once.

Launch:  streamlit run app.py
"""

from __future__ import annotations

import os
import random

import numpy as np
import streamlit as st

# Streamlit Cloud stores the key in st.secrets; EDSL reads it from the environment.
# Must run before any module that imports edsl.
try:
    os.environ.setdefault(
        "EXPECTED_PARROT_API_KEY", st.secrets["EXPECTED_PARROT_API_KEY"]
    )
except Exception:
    pass

from agents import PRESETS, create_agent
from banks import BANKS, BANKS_BY_KEY
from decisions import DECISIONS
from engine import check_instructions, run_session, run_tests
from plots import plot_payoff_hists
from logger import log_session

# ---- page config ----
st.set_page_config(
    page_title="Agent Delegation Lab",
    page_icon="🎯",
    layout="wide",
)

# ---- constants ----
BASE_BUDGET = 10.00
TEST_PRICES = {"category": 0.01, "near": 0.03}  # per agent, question, and run; preset agent
OWN_AGENT_TEST_MULTIPLIER = 2
RUN_CHOICES = [1, 10, 100]
OWN_AGENT = "My agent"
TIER_COSTS = {
    "Default": 0.00,
    "Choose": 1.00,
    "Custom": 5.00,
}
TIER_DESCRIPTIONS = {
    "Default": "You get a preset agent picked at random. It costs nothing.",
    "Choose": "Pick which preset agent represents you.",
    "Custom": "Write your agent's instructions from scratch.",
}
GENERAL_APPROACH_RULE = (
    "Describe a general approach. Instructions that refer to any of the 8 "
    "decisions are rejected."
)

# ---- session state ----
DEFAULTS = {
    "budget_remaining": BASE_BUDGET,
    "test_history": [],
    "own_agent_versions": {},  # instruction text -> "My agent vN"
    "last_own_instruction": "",
    "instruction_checks": {},  # instruction text -> (allowed, reason)
    "committed": False,
    "session_results": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---- helpers ----
def esc(text: str) -> str:
    """Escape $ so Streamlit does not render it as math."""
    return text.replace("$", "\\$")


def md(text: str) -> str:
    """esc() plus hard line breaks, for decision descriptions."""
    return esc(text).replace("\n", "  \n")


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def question_label(decision) -> str:
    """Number within its bank plus title, e.g. '7. Small Edge'."""
    return f"{int(decision.name.rsplit('_', 1)[1])}. {decision.title}"


def option_name(label: str) -> str:
    """'Option A: $5.00 for certain' -> 'Option A'."""
    return label.split(":")[0]


def test_price(bank_kind: str, own_agent: bool) -> float:
    price = TEST_PRICES[bank_kind]
    return price * OWN_AGENT_TEST_MULTIPLIER if own_agent else price


def safest_label(decision) -> str:
    """Label of the option whose payoff varies least."""
    def spread(opt):
        ev = opt.expected_value
        return sum(o.probability * (o.payoff - ev) ** 2 for o in opt.outcomes)
    return min(decision.options, key=spread).label


def format_shares(decision, shares: dict[str, float], runs: int) -> str:
    """One run: the option picked. More runs: each option's share, in option order."""
    order = [o.label for o in decision.options] + ["No answer"]
    picked = [label for label in order if shares.get(label, 0) > 0]
    if runs == 1:
        return option_name(picked[0])
    return ", ".join(f"{option_name(label)} {shares[label]:.0%}" for label in picked)


def check_own_instructions(text: str) -> tuple[bool, str]:
    """Run the instruction check once per distinct text; log rejections."""
    checks = st.session_state.instruction_checks
    if text not in checks:
        checks[text] = check_instructions(text)
        if not checks[text][0]:
            try:
                log_session(
                    game_key="delegation_lab_instruction_check",
                    agent_configs=[{"instruction": text}],
                    settings={},
                    results={"allowed": False, "reason": checks[text][1]},
                )
            except Exception:
                pass
    return checks[text]


def what_came_up(outcome: dict) -> str:
    if outcome["chosen"] == "No answer":
        return "No answer, so no payoff"
    if outcome["certain"]:
        return f"${outcome['payoff']:.2f} for certain"
    label, p = outcome["outcome_label"], outcome["probability"]
    if label and outcome["category"] == "Ambiguity":
        return f"A {label.lower()}"  # no chance shown when the odds are unknown
    chance = "one in three" if abs(p - 1 / 3) < 1e-9 else f"{round(p * 100)}% chance"
    if label:
        return f"{label} ({chance})"
    return f"The {chance}"


# =====================================================================
# HEADER
# =====================================================================
st.title("Agent Delegation Lab")

with st.expander("About this experiment", expanded=True):
    st.markdown(esc(
        "You choose an AI agent to make 8 money decisions for you. "
        "You keep what the agent earns, plus any budget you do not spend.\n\n"
        f"You start with a budget of ${BASE_BUDGET:.0f}. "
        "You can spend it in two ways.\n\n"
        "- You can test agents on practice questions before you choose.\n"
        "- You can pay to pick which preset agent you get, or to write your "
        "own agent's instructions.\n\n"
        "How it works:\n\n"
        "1. Read the 8 decisions your agent will face. You can read them, "
        "but you cannot test any agent on them.\n"
        "2. Test agents on practice questions. This step is optional, and "
        "each test costs money. An agent can answer the same question "
        "differently each time, so tests show how often it picks each option.\n"
        "3. Choose a control tier and set up your agent.\n"
        "4. Your agent makes each of the 8 decisions once, and each lottery "
        "is drawn once. You see what it chose, what came up, and what you earned.\n\n"
        "We study how much people pay to test and control an AI agent, and "
        "whether testing changes which agent they pick."
    ))

# Budget display
budget = st.session_state.budget_remaining
spent = BASE_BUDGET - budget
col_b1, col_b2, col_b3 = st.columns(3)
col_b1.metric("Starting Budget", f"${BASE_BUDGET:.2f}")
col_b2.metric("Spent So Far", f"${spent:.2f}")
col_b3.metric("Budget Remaining", f"${budget:.2f}")

# =====================================================================
# STEP 1: The 8 target decisions, visible but not testable
# =====================================================================
st.subheader("Step 1. The decisions your agent will face")
st.markdown(
    "Your agent will face these 8 decisions at the end. "
    "You can read them now. You cannot test any agent on them."
)
with st.expander("Read the 8 decisions", expanded=True):
    target_cols = st.columns(2)
    for i, decision in enumerate(DECISIONS):
        with target_cols[i % 2], st.container(border=True):
            st.markdown(f"**{i + 1}. {decision.title}** ({decision.category})")
            st.markdown(md(decision.description))

st.divider()

# =====================================================================
# STEP 2: Test lab (before commitment)
# =====================================================================
if not st.session_state.committed:
    st.subheader("Step 2. Test agents on practice questions (optional)")
    cat_price, near_price = TEST_PRICES["category"], TEST_PRICES["near"]
    st.markdown(esc(
        "You can test any agent on practice questions before you choose. "
        "The practice questions are different from the 8 decisions above.\n\n"
        "Each run is a new answer from the agent. The same agent can answer "
        "the same question differently, so more runs show how consistent it "
        "is. At the end, your agent answers each of the 8 decisions only once.\n\n"
        "You pay for each agent, on each question, for each run. "
        "Here are the prices for one run:\n\n"
        "- Category banks hold questions of the same kinds as the 8 decisions, "
        f"with different amounts and odds. One run costs ${cat_price:.2f} "
        f"for a preset agent and ${test_price('category', True):.2f} for your own agent.\n"
        "- Close variant banks hold small changes to one of the 8 decisions. "
        f"One run costs ${near_price:.2f} for a preset agent and "
        f"${test_price('near', True):.2f} for your own agent."
    ))

    col_agents, col_bank = st.columns(2)

    with col_agents:
        chosen_presets = st.multiselect(
            "Preset agents to test", list(PRESETS), key="test_presets"
        )
        with st.expander("Read the preset agents' instructions"):
            for name, preset in PRESETS.items():
                st.markdown(f"**{name}**. {esc(preset['description'])}")
                st.code(preset["instruction"], language=None)
        own_text = st.text_area(
            "Your own agent's instructions (optional)",
            height=150,
            key="own_agent_text",
            placeholder=(
                "Describe how your agent should make decisions. For example, "
                "say what it should care about and how much risk it should take."
            ),
        ).strip()
        st.caption(
            f"{GENERAL_APPROACH_RULE} You are not charged for a rejected test. "
            f"If you test different instructions, we label them {OWN_AGENT} v1, "
            f"{OWN_AGENT} v2, and so on."
        )

    with col_bank:
        bank_key = st.selectbox(
            "Question bank",
            [b.key for b in BANKS],
            format_func=lambda k: (
                f"{BANKS_BY_KEY[k].name} ({len(BANKS_BY_KEY[k].decisions)} questions, "
                f"${TEST_PRICES[BANKS_BY_KEY[k].kind]:.2f} per run for a preset agent)"
            ),
            key="test_bank",
        )
        bank = BANKS_BY_KEY[bank_key]
        with st.expander(f"Browse the {len(bank.decisions)} questions in this bank for free"):
            for decision in bank.decisions:
                st.markdown(f"**{question_label(decision)}**")
                st.markdown(md(decision.description))
        use_all = st.checkbox("Use all questions in this bank", key="test_use_all")
        picked = st.multiselect(
            "Questions to test",
            [d.name for d in bank.decisions],
            format_func=lambda n: question_label(
                next(d for d in bank.decisions if d.name == n)
            ),
            key=f"test_questions_{bank.key}",
            disabled=use_all,
        )
        problems = (
            bank.decisions if use_all
            else [d for d in bank.decisions if d.name in picked]
        )
        runs = st.radio(
            "Runs per question", RUN_CHOICES, index=1, horizontal=True, key="test_runs"
        )
        st.caption("A test with 100 runs can take a few minutes.")

    n_agents = len(chosen_presets) + (1 if own_text else 0)
    cost = round(
        len(problems)
        * runs
        * (
            len(chosen_presets) * test_price(bank.kind, False)
            + (test_price(bank.kind, True) if own_text else 0)
        ),
        2,
    )
    can_afford_test = cost <= st.session_state.budget_remaining + 1e-9

    if n_agents == 0:
        st.caption("Pick at least one agent to test.")
    elif not problems:
        st.caption("Pick at least one question.")
    else:
        st.markdown(esc(
            f"This test runs {plural(n_agents, 'agent')} {plural(runs, 'time')} "
            f"on each of {plural(len(problems), 'question')} and costs ${cost:.2f}."
        ))
        if not can_afford_test:
            st.caption("You do not have enough budget for this test.")

    if st.button(
        esc(f"Run test (${cost:.2f})"),
        disabled=not (n_agents and problems and can_afford_test),
        width="stretch",
    ):
        blocked = False
        if own_text:
            try:
                with st.spinner("Checking your agent's instructions..."):
                    allowed, reason = check_own_instructions(own_text)
            except Exception as exc:
                st.error(
                    "We could not check your agent's instructions, and you were "
                    f"not charged. {exc}"
                )
                blocked = True
            else:
                if not allowed:
                    st.error(esc(
                        "Your agent's instructions refer to one of the 8 decisions, "
                        f"so we did not run this test. You were not charged. {reason}"
                    ))
                    blocked = True

        if not blocked:
            versions = st.session_state.own_agent_versions
            agent_specs = [
                {"label": name, "preset": name, "instruction": PRESETS[name]["instruction"]}
                for name in chosen_presets
            ]
            if own_text:
                own_label = versions.get(own_text, f"{OWN_AGENT} v{len(versions) + 1}")
                agent_specs.append(
                    {"label": own_label, "preset": None, "instruction": own_text}
                )
            agents = [
                create_agent(s["label"], preset_name=s["preset"])
                if s["preset"]
                else create_agent(s["label"], custom_instruction=s["instruction"])
                for s in agent_specs
            ]

            st.session_state.budget_remaining = round(
                st.session_state.budget_remaining - cost, 2
            )
            with st.spinner("Your agents are answering the practice questions..."):
                try:
                    results = run_tests(agents, problems, runs=runs)
                except Exception as exc:
                    st.session_state.budget_remaining = round(
                        st.session_state.budget_remaining + cost, 2
                    )
                    st.error(f"The test failed, and you were not charged. {exc}")
                else:
                    if own_text:
                        versions[own_text] = own_label
                        st.session_state.last_own_instruction = own_text
                    st.session_state.test_history.append({
                        "bank_key": bank.key,
                        "bank_name": bank.name,
                        "bank_kind": bank.kind,
                        "problems": problems,
                        "runs": runs,
                        "agents": agent_specs,
                        "results": results,
                        "cost": cost,
                    })
                    try:
                        log_session(
                            game_key="delegation_lab_test",
                            agent_configs=agent_specs,
                            settings={
                                "bank": bank.key,
                                "bank_kind": bank.kind,
                                "questions": [d.name for d in problems],
                                "runs": runs,
                                "cost": cost,
                                "budget_after": st.session_state.budget_remaining,
                            },
                            results={
                                label: r["choice_runs"] for label, r in results.items()
                            },
                        )
                    except Exception:
                        pass
                    st.rerun()

    # ---- Test results ----
    if st.session_state.test_history:
        st.divider()
        st.subheader("Your test results")

        for n, record in reversed(
            list(enumerate(st.session_state.test_history, 1))
        ):
            with st.container(border=True):
                st.markdown(f"##### Test {n}")
                st.caption(esc(
                    f"{record['bank_name']}. {plural(len(record['problems']), 'question')}, "
                    f"{plural(record['runs'], 'run')} per question. Cost ${record['cost']:.2f}."
                ))
                rows = []
                for j, decision in enumerate(record["problems"]):
                    row = {"Question": question_label(decision)}
                    for label, r in record["results"].items():
                        row[label] = format_shares(
                            decision, r["per_problem"][j]["shares"], record["runs"]
                        )
                    rows.append(row)
                mean_row = {"Question": "Average expected value per question"}
                for label, r in record["results"].items():
                    mean_row[label] = f"${np.mean([pp['ev'] for pp in r['per_problem']]):.2f}"
                st.dataframe(rows + [mean_row], hide_index=True, width="stretch")

                st.caption(
                    "Total payout on these questions for each agent, from 1,000 "
                    "simulated rounds. Each round takes one of the agent's runs at "
                    "random and draws each lottery once."
                )
                st.altair_chart(plot_payoff_hists(
                    {label: r["total_payoffs"] for label, r in record["results"].items()},
                    theme=st.context.theme.type,
                ))

        # ---- Summary across all tests ----
        st.markdown("#### Summary of all your tests")
        st.caption(
            "Each row sums up every test you ran with that agent. Different tests "
            "can use different questions, so compare agents within one test when "
            "you can. The safest option is the one whose payoff varies least. "
            "\"Same answer every run\" counts only questions you ran more than once."
        )
        summary: dict[str, dict] = {}
        for record in st.session_state.test_history:
            for label, r in record["results"].items():
                s = summary.setdefault(
                    label,
                    {"questions": 0, "answers": 0, "ev": 0.0, "safest": 0.0, "repeated": 0, "same": 0},
                )
                for pp, decision in zip(r["per_problem"], record["problems"]):
                    s["questions"] += 1
                    s["answers"] += record["runs"]
                    s["ev"] += pp["ev"]
                    s["safest"] += pp["shares"].get(safest_label(decision), 0) * record["runs"]
                    if record["runs"] > 1:
                        s["repeated"] += 1
                        s["same"] += len(pp["shares"]) == 1
        st.dataframe(
            [
                {
                    "Agent": label,
                    "Questions tested": s["questions"],
                    "Answers": s["answers"],
                    "Average expected value per question": f"${s['ev'] / s['questions']:.2f}",
                    "Chose the safest option": f"{s['safest'] / s['answers']:.0%}",
                    "Same answer every run": (
                        f"{s['same'] / s['repeated']:.0%}" if s["repeated"] else "No repeats"
                    ),
                }
                for label, s in summary.items()
            ],
            hide_index=True,
            width="stretch",
        )

    # ---- Commit to tier ----
    st.divider()
    st.subheader("Step 3. Choose a control tier")
    st.markdown(
        "We take the tier cost from your remaining budget. "
        "After you choose, you cannot run more tests."
    )

    cols = st.columns(len(TIER_COSTS))
    for i, (tier, tier_cost) in enumerate(TIER_COSTS.items()):
        with cols[i]:
            affordable = st.session_state.budget_remaining >= tier_cost
            kept = st.session_state.budget_remaining - tier_cost
            st.markdown(f"**{tier}**")
            st.markdown(esc(f"Tier cost: ${tier_cost:.2f}"))
            if affordable:
                st.markdown(esc(f"You would keep ${kept:.2f}."))
            else:
                st.markdown("You cannot afford this tier.")
            st.caption(TIER_DESCRIPTIONS[tier])
            if st.button(
                f"Commit: {tier}",
                key=f"commit_{tier}",
                width="stretch",
                disabled=not affordable,
            ):
                st.session_state.committed = True
                st.session_state.selected_tier = tier
                st.session_state.budget_remaining = round(
                    st.session_state.budget_remaining - tier_cost, 2
                )
                if tier == "Default":
                    st.session_state.assigned_preset = random.choice(
                        list(PRESETS.keys())
                    )
                st.rerun()

# =====================================================================
# STEP 3: Configure agent (after commitment)
# =====================================================================
if st.session_state.committed and st.session_state.session_results is None:
    selected_tier = st.session_state.selected_tier
    budget_kept = st.session_state.budget_remaining

    st.subheader("Step 3. Set up your agent")

    test_spent = sum(r["cost"] for r in st.session_state.test_history)
    tier_cost = TIER_COSTS[selected_tier]
    with st.expander("Budget breakdown"):
        st.markdown(esc(f"- Starting budget: ${BASE_BUDGET:.2f}"))
        if test_spent > 0:
            st.markdown(esc(
                f"- Tests ({len(st.session_state.test_history)}): -${test_spent:.2f}"
            ))
        st.markdown(esc(f"- Tier ({selected_tier}): -${tier_cost:.2f}"))
        st.markdown(esc(f"- Budget kept: ${budget_kept:.2f}"))

    preset_names = list(PRESETS.keys())

    if selected_tier == "Default":
        assigned = st.session_state.assigned_preset
        st.markdown(f"You were assigned **{assigned}**.")
        st.markdown(f"*\"{esc(PRESETS[assigned]['description'])}\"*")
        with st.expander("See full instructions"):
            st.code(PRESETS[assigned]["instruction"], language=None)
        agent_config = {"tier": "Default", "preset": assigned}

    elif selected_tier == "Choose":
        chosen_preset = st.selectbox("Pick your agent:", preset_names)
        st.markdown(f"*\"{esc(PRESETS[chosen_preset]['description'])}\"*")
        with st.expander("See full instructions"):
            st.code(PRESETS[chosen_preset]["instruction"], language=None)
        agent_config = {"tier": "Choose", "preset": chosen_preset}

    elif selected_tier == "Custom":
        custom_instruction = st.text_area(
            "Write your agent's instructions from scratch:",
            value=st.session_state.last_own_instruction,
            height=200,
            key="custom_instruction",
            placeholder=(
                "Describe how your agent should make decisions. For example, "
                "say what it should care about and how much risk it should take."
            ),
        )
        st.caption(GENERAL_APPROACH_RULE)
        if st.session_state.last_own_instruction:
            st.caption(
                "We filled in the instructions you last tested. You can change them."
            )
        agent_config = {"tier": "Custom", "instruction": custom_instruction}

    st.divider()

    # ---- Run button ----
    st.subheader("Step 4. Send your agent to make the 8 decisions")
    st.markdown(
        "Your agent answers each decision once, and each lottery is drawn once."
    )

    run_clicked = st.button(
        "Run all 8 decisions",
        type="primary",
        width="stretch",
    )

    if run_clicked:
        if selected_tier == "Custom":
            instruction = agent_config["instruction"].strip()
            if not instruction:
                st.error("Please write instructions for your agent.")
                st.stop()
            try:
                with st.spinner("Checking your agent's instructions..."):
                    allowed, reason = check_own_instructions(instruction)
            except Exception as exc:
                st.error(f"We could not check your agent's instructions. {exc}")
                st.stop()
            if not allowed:
                st.error(esc(
                    "Your agent's instructions refer to one of the 8 decisions. "
                    f"Change them and try again. {reason}"
                ))
                st.stop()
            agent = create_agent("My_Agent", custom_instruction=instruction)
            display_name = "Custom Agent"
        else:
            preset = (
                st.session_state.assigned_preset
                if selected_tier == "Default"
                else agent_config["preset"]
            )
            agent = create_agent("My_Agent", preset_name=preset)
            display_name = preset

        with st.spinner("Your agent is making the 8 decisions..."):
            try:
                result = run_session(agent)
            except Exception as exc:
                st.error(f"Error running decisions: {exc}")
                st.exception(exc)
                st.stop()

        game_earnings = sum(o["payoff"] for o in result["outcomes"])
        st.session_state.session_results = {
            "result": result,
            "display_name": display_name,
            "budget_kept": budget_kept,
            "tier": selected_tier,
            "agent_config": agent_config,
        }

        try:
            log_session(
                game_key="delegation_lab",
                agent_configs=[agent_config],
                settings={
                    "tier": selected_tier,
                    "budget_kept": budget_kept,
                    "test_count": len(st.session_state.test_history),
                    "test_spent": test_spent,
                    "banks_tested": [
                        r["bank_key"] for r in st.session_state.test_history
                    ],
                },
                results={
                    "choices": result["choices"],
                    "outcomes": result["outcomes"],
                    "game_earnings": game_earnings,
                    "total_payout": budget_kept + game_earnings,
                },
            )
        except Exception:
            pass

        st.rerun()

# =====================================================================
# RESULTS
# =====================================================================
if st.session_state.session_results is not None:
    sr = st.session_state.session_results
    outcomes = sr["result"]["outcomes"]
    budget_kept = sr["budget_kept"]
    game_earnings = sum(o["payoff"] for o in outcomes)
    test_spent = sum(r["cost"] for r in st.session_state.test_history)

    st.divider()
    st.header("Results")
    st.markdown(esc(
        f"Your agent was {sr['display_name']}. It answered each decision once, "
        "and each lottery was drawn once."
    ))

    st.dataframe(
        [
            {
                "Decision": f"{i}. {o['title']}",
                "Your agent chose": o["chosen"],
                "What came up": what_came_up(o),
                "Payoff": f"${o['payoff']:.2f}",
            }
            for i, o in enumerate(outcomes, 1)
        ],
        hide_index=True,
        width="stretch",
    )

    st.subheader("Scorecard")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Starting Budget", f"${BASE_BUDGET:.2f}")
    sc2.metric("Tests Spent", f"${test_spent:.2f}")
    sc3.metric("Tier Cost", f"${TIER_COSTS[sr['tier']]:.2f}")
    sc4.metric("Budget Kept", f"${budget_kept:.2f}")
    sc5.metric("Game Earnings", f"${game_earnings:.2f}")

    st.markdown(esc(f"### Total Payout: ${budget_kept + game_earnings:.2f}"))

    # ---- Start over ----
    st.divider()
    if st.button("Start Over (new session)", width="stretch"):
        for key in list(DEFAULTS) + ["selected_tier", "assigned_preset"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

elif not st.session_state.committed:
    st.markdown("---")
    st.caption(
        "Read the 8 decisions, test agents if you want, then choose a control tier."
    )
