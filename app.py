"""
Agent Delegation Lab
=====================
Pre-commit to an AI agent, then watch it face 8 economic decisions.
How much of your budget will you spend for control -- and for information?

Launch:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import random

from agents import PRESETS, create_agent
from decisions import (
    DECISIONS,
    CATEGORY_DESCRIPTIONS,
    simulate_session,
    sample_preview_problems,
)
from engine import run_session, run_preview
from plots import (
    plot_total_payoff_distribution,
    plot_problem_breakdown,
    plot_comparison,
)
from logger import log_session

# ---- page config ----
st.set_page_config(
    page_title="Agent Delegation Lab",
    page_icon="🎯",
    layout="wide",
)

# ---- constants ----
BASE_BUDGET = 10.00
PREVIEW_COST_PRESET = 0.50
PREVIEW_COST_CUSTOM = 1.00
TIER_COSTS = {
    "Default": 0.00,
    "Choose": 1.00,
    "Edit": 3.00,
    "Custom": 5.00,
}
TIER_DESCRIPTIONS = {
    "Default": "A preset agent is assigned to you at random. No extra cost.",
    "Choose": "Pick which preset agent represents you.",
    "Edit": "Pick a preset and modify its instructions.",
    "Custom": "Write your agent's instructions from scratch.",
}

# ---- session state ----
if "budget_remaining" not in st.session_state:
    st.session_state.budget_remaining = BASE_BUDGET
if "preview_history" not in st.session_state:
    st.session_state.preview_history = []
if "committed" not in st.session_state:
    st.session_state.committed = False
if "session_results" not in st.session_state:
    st.session_state.session_results = None
if "history" not in st.session_state:
    st.session_state.history = []
if "preview_run_counter" not in st.session_state:
    st.session_state.preview_run_counter = 0

# =====================================================================
# HEADER
# =====================================================================
st.title("Agent Delegation Lab")

with st.expander("About this experiment", expanded=True):
    st.markdown(
        """
**What is this?** You delegate economic decisions to an AI agent.
The agent faces 8 decision problems -- lotteries, insurance choices,
investment allocations, and loss-framed gambles -- and you keep
whatever it earns.

**The twist: you pay for control.** You start with a $10 budget.
You can spend some of it to choose, customize, or even write your
own agent's instructions. You can also spend budget to *preview*
how an agent behaves on practice problems before you commit.
More control costs more, leaving less guaranteed money in your pocket.

**What we're studying:** How much do people pay for control over
their AI delegate? Does previewing agents change which ones people
pick? Do custom-written agents actually outperform the presets?

**How it works:**
1. **Scout** (optional) -- Pay to preview agents on practice problems
   (different from the real 8). See how they handle risk, losses, and ambiguity.
2. **Commit** -- Choose a control tier and configure your agent.
   The tier cost comes out of your remaining budget.
3. **Run** -- Your agent faces the real 8 decisions. You see
   the payoff distribution from 1,000 Monte Carlo simulations.

Each session makes one LLM call (8 questions, ~$0.005) and runs
instantly. All decisions use real behavioral economics paradigms.
"""
    )

# Budget display
budget = st.session_state.budget_remaining
spent = BASE_BUDGET - budget
col_b1, col_b2, col_b3 = st.columns(3)
col_b1.metric("Starting Budget", f"${BASE_BUDGET:.2f}")
col_b2.metric("Spent So Far", f"${spent:.2f}")
col_b3.metric("Budget Remaining", f"${budget:.2f}")

# =====================================================================
# STEP 1: Show what the agent will face
# =====================================================================
with st.expander("What will my agent face? (click to see categories)", expanded=False):
    st.markdown("Your agent will face **8 decision problems** drawn from these categories:")
    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        st.markdown(f"- **{cat}:** {desc}")
    st.markdown(
        "*You can see the categories, but not the specific problems. "
        "Your agent will see each problem for the first time when it decides.*"
    )

st.divider()

# =====================================================================
# STEP 2: Browse agents & optional previews (before commitment)
# =====================================================================
if not st.session_state.committed:
    st.subheader("Step 1: Scout your agents (optional)")
    st.markdown(
        "Preview any agent on a set of **practice problems** -- different from "
        "the real 8. "
        f"Previewing a preset costs **${PREVIEW_COST_PRESET:.2f}**. "
        f"Previewing a custom agent costs **${PREVIEW_COST_CUSTOM:.2f}**."
    )

    # ---- Preset previews ----
    st.markdown("#### Preset Agents")
    preset_names = list(PRESETS.keys())

    for preset_name in preset_names:
        preset = PRESETS[preset_name]
        with st.container(border=True):
            col_info, col_action = st.columns([3, 1])
            with col_info:
                st.markdown(f"**{preset_name}**")
                st.markdown(f"*{preset['description']}*")
                with st.expander("See full instructions"):
                    st.code(preset["instruction"], language=None)
            with col_action:
                can_afford = st.session_state.budget_remaining >= PREVIEW_COST_PRESET
                btn_key = (
                    f"preview_preset_{preset_name}_"
                    f"{st.session_state.preview_run_counter}"
                )
                if st.button(
                    f"Preview (${PREVIEW_COST_PRESET:.2f})",
                    key=btn_key,
                    use_container_width=True,
                    disabled=not can_afford,
                ):
                    st.session_state.budget_remaining -= PREVIEW_COST_PRESET
                    st.session_state.preview_run_counter += 1
                    agent = create_agent(
                        "Preview_Agent", preset_name=preset_name
                    )
                    problems = sample_preview_problems(n=5)
                    with st.spinner(
                        f"Previewing {preset_name} on 5 practice problems..."
                    ):
                        try:
                            result = run_preview(agent, problems)
                            st.session_state.preview_history.append({
                                "agent_label": preset_name,
                                "cost": PREVIEW_COST_PRESET,
                                "result": result,
                                "problems": problems,
                                "is_custom": False,
                            })
                            try:
                                log_session(
                                    game_key="delegation_lab_preview",
                                    agent_configs=[{"preset": preset_name}],
                                    settings={
                                        "preview_cost": PREVIEW_COST_PRESET,
                                        "budget_after": (
                                            st.session_state.budget_remaining
                                        ),
                                        "n_problems": len(problems),
                                        "problem_names": [
                                            p.name for p in problems
                                        ],
                                    },
                                    results={"choices": result["choices"]},
                                )
                            except Exception:
                                pass
                        except Exception as exc:
                            st.error(f"Preview failed: {exc}")
                            st.session_state.budget_remaining += (
                                PREVIEW_COST_PRESET
                            )
                    st.rerun()
                if not can_afford:
                    st.caption("Not enough budget")

    # ---- Custom preview ----
    st.markdown("#### Custom Agent Preview")
    st.markdown(
        f"Write your own instructions and preview how the agent behaves. "
        f"Costs **${PREVIEW_COST_CUSTOM:.2f}**."
    )
    custom_preview_instruction = st.text_area(
        "Custom agent instructions (for preview):",
        height=150,
        key="custom_preview_text",
        placeholder=(
            "Describe how your agent should approach decisions. "
            "What should it prioritize? How should it handle risk?"
        ),
    )
    can_afford_custom = (
        st.session_state.budget_remaining >= PREVIEW_COST_CUSTOM
    )
    if st.button(
        f"Preview Custom Agent (${PREVIEW_COST_CUSTOM:.2f})",
        disabled=(
            not can_afford_custom or not custom_preview_instruction.strip()
        ),
        use_container_width=True,
    ):
        st.session_state.budget_remaining -= PREVIEW_COST_CUSTOM
        st.session_state.preview_run_counter += 1
        agent = create_agent(
            "Preview_Custom",
            custom_instruction=custom_preview_instruction.strip(),
        )
        problems = sample_preview_problems(n=5)
        with st.spinner("Previewing custom agent on 5 practice problems..."):
            try:
                result = run_preview(agent, problems)
                st.session_state.preview_history.append({
                    "agent_label": "Custom Agent",
                    "cost": PREVIEW_COST_CUSTOM,
                    "result": result,
                    "problems": problems,
                    "is_custom": True,
                })
                try:
                    log_session(
                        game_key="delegation_lab_preview",
                        agent_configs=[{
                            "preset": "custom",
                            "instruction": custom_preview_instruction[:200],
                        }],
                        settings={
                            "preview_cost": PREVIEW_COST_CUSTOM,
                            "budget_after": (
                                st.session_state.budget_remaining
                            ),
                            "n_problems": len(problems),
                            "problem_names": [p.name for p in problems],
                        },
                        results={"choices": result["choices"]},
                    )
                except Exception:
                    pass
            except Exception as exc:
                st.error(f"Preview failed: {exc}")
                st.session_state.budget_remaining += PREVIEW_COST_CUSTOM
        st.rerun()

    # ---- Show preview results ----
    if st.session_state.preview_history:
        st.divider()
        st.subheader("Preview Results")
        for idx, preview in enumerate(
            reversed(st.session_state.preview_history)
        ):
            preview_num = len(st.session_state.preview_history) - idx
            result = preview["result"]
            sim = result["simulation"]
            per_problem = sim["per_problem"]

            with st.container(border=True):
                st.markdown(
                    f"**Preview #{preview_num}: {preview['agent_label']}** "
                    f"(cost: ${preview['cost']:.2f})"
                )

                # Disposition summary
                n_safe = 0
                n_total = len(per_problem)
                for pp in per_problem:
                    chosen = pp["chosen"]
                    decision = next(
                        (
                            d
                            for d in preview["problems"]
                            if d.name == pp["name"]
                        ),
                        None,
                    )
                    if decision and chosen == decision.options[0].label:
                        n_safe += 1

                st.markdown(
                    f"Chose the safer/default option in "
                    f"**{n_safe}/{n_total}** problems"
                )

                # Compact table of choices
                for pp in per_problem:
                    col_t, col_c = st.columns([2, 3])
                    with col_t:
                        st.caption(f"{pp['category']}: {pp['title']}")
                    with col_c:
                        chosen_short = pp["chosen"]
                        if len(chosen_short) > 40:
                            chosen_short = chosen_short[:37] + "..."
                        st.caption(
                            f"-> {chosen_short}  (EV: ${pp['ev']:.2f})"
                        )

                import numpy as np

                mean_ev = np.mean([pp["ev"] for pp in per_problem])
                st.caption(
                    f"Average EV across preview problems: ${mean_ev:.2f}"
                )

    # ---- Commit to tier ----
    st.divider()
    st.subheader("Step 2: Commit to your agent")
    st.markdown(
        "Choose a control tier. The tier cost is deducted from your "
        "remaining budget. Your agent will then face the **real 8 decisions**."
    )

    cols = st.columns(4)
    for i, (tier, cost) in enumerate(TIER_COSTS.items()):
        with cols[i]:
            affordable = st.session_state.budget_remaining >= cost
            kept = st.session_state.budget_remaining - cost
            st.markdown(f"**{tier}**")
            st.markdown(f"Tier cost: **${cost:.2f}**")
            if affordable:
                st.markdown(f"You'd keep: ${kept:.2f}")
            else:
                st.markdown(f"~~You'd keep: ${kept:.2f}~~ Can't afford")
            st.caption(TIER_DESCRIPTIONS[tier])
            if st.button(
                f"Commit: {tier}",
                key=f"commit_{tier}",
                use_container_width=True,
                disabled=not affordable,
            ):
                st.session_state.committed = True
                st.session_state.selected_tier = tier
                st.session_state.budget_remaining -= cost
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

    st.subheader("Step 3: Configure your agent")

    # Show spending breakdown
    preview_spent = sum(
        p["cost"] for p in st.session_state.preview_history
    )
    tier_cost = TIER_COSTS[selected_tier]
    with st.expander("Budget breakdown"):
        st.markdown(f"- Starting budget: **${BASE_BUDGET:.2f}**")
        if preview_spent > 0:
            st.markdown(
                f"- Previews ({len(st.session_state.preview_history)}): "
                f"**-${preview_spent:.2f}**"
            )
        st.markdown(f"- Tier ({selected_tier}): **-${tier_cost:.2f}**")
        st.markdown(f"- **Budget kept: ${budget_kept:.2f}**")

    preset_names = list(PRESETS.keys())

    if selected_tier == "Default":
        assigned = st.session_state.assigned_preset
        st.markdown(f"You've been assigned: **{assigned}**")
        st.markdown(f"*\"{PRESETS[assigned]['description']}\"*")
        with st.expander("See full instructions"):
            st.code(PRESETS[assigned]["instruction"], language=None)
        agent_config = {"tier": "Default", "preset": assigned}

    elif selected_tier == "Choose":
        chosen = st.selectbox("Pick your agent:", preset_names)
        st.markdown(f"*\"{PRESETS[chosen]['description']}\"*")
        with st.expander("See full instructions"):
            st.code(PRESETS[chosen]["instruction"], language=None)
        agent_config = {"tier": "Choose", "preset": chosen}

    elif selected_tier == "Edit":
        base = st.selectbox("Start from:", preset_names)
        st.markdown(f"*Original: \"{PRESETS[base]['description']}\"*")
        edited_instruction = st.text_area(
            "Edit the instructions:",
            value=PRESETS[base]["instruction"],
            height=200,
            key="edit_instruction",
        )
        agent_config = {
            "tier": "Edit",
            "base_preset": base,
            "instruction": edited_instruction,
        }

    elif selected_tier == "Custom":
        custom_instruction = st.text_area(
            "Write your agent's instructions from scratch:",
            height=200,
            key="custom_instruction",
            placeholder=(
                "Describe how your agent should approach decisions. "
                "What should it prioritize? How should it handle risk?"
            ),
        )
        agent_config = {"tier": "Custom", "instruction": custom_instruction}

    st.divider()

    # ---- Run button ----
    st.subheader("Step 4: Send your agent into the arena")

    run_clicked = st.button(
        "Run all 8 decisions",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        try:
            if selected_tier == "Default":
                agent = create_agent(
                    "My_Agent",
                    preset_name=st.session_state.assigned_preset,
                )
                display_name = st.session_state.assigned_preset
            elif selected_tier == "Choose":
                agent = create_agent(
                    "My_Agent", preset_name=agent_config["preset"]
                )
                display_name = agent_config["preset"]
            elif selected_tier == "Edit":
                instruction = agent_config.get("instruction", "").strip()
                if not instruction:
                    st.error("Please write some instructions for your agent.")
                    st.stop()
                agent = create_agent(
                    "My_Agent", custom_instruction=instruction
                )
                display_name = f"{agent_config['base_preset']} (edited)"
            elif selected_tier == "Custom":
                instruction = agent_config.get("instruction", "").strip()
                if not instruction:
                    st.error("Please write instructions for your agent.")
                    st.stop()
                agent = create_agent(
                    "My_Agent", custom_instruction=instruction
                )
                display_name = "Custom Agent"
        except Exception as exc:
            st.error(f"Error creating agent: {exc}")
            st.stop()

        with st.spinner(
            "Your agent is facing 8 decisions... (30-60 seconds)"
        ):
            try:
                result = run_session(agent, n_sims=1000)
            except Exception as exc:
                st.error(f"Error running decisions: {exc}")
                st.exception(exc)
                st.stop()

        st.session_state.session_results = {
            "result": result,
            "display_name": display_name,
            "budget_kept": budget_kept,
            "tier": selected_tier,
            "agent_config": agent_config,
        }

        st.session_state.history.append(
            (display_name, result["simulation"], budget_kept)
        )

        try:
            log_session(
                game_key="delegation_lab",
                agent_configs=[agent_config],
                settings={
                    "tier": selected_tier,
                    "budget_kept": budget_kept,
                    "preview_count": len(st.session_state.preview_history),
                    "preview_spent": sum(
                        p["cost"]
                        for p in st.session_state.preview_history
                    ),
                },
                results={"choices": result["choices"]},
            )
        except Exception:
            pass

        st.rerun()

# =====================================================================
# RESULTS
# =====================================================================
if st.session_state.session_results is not None:
    sr = st.session_state.session_results
    result = sr["result"]
    display_name = sr["display_name"]
    budget_kept = sr["budget_kept"]
    simulation = result["simulation"]

    st.divider()
    st.header("Results")

    # ---- Problem-by-problem choices ----
    st.subheader("What your agent chose")

    for i, prob in enumerate(simulation["per_problem"]):
        decision = DECISIONS[i]
        col_q, col_a = st.columns([3, 2])
        with col_q:
            st.markdown(
                f"**{i+1}. {prob['title']}** ({prob['category']})"
            )
            st.caption(
                decision.description[:150] + "..."
                if len(decision.description) > 150
                else decision.description
            )
        with col_a:
            st.markdown(f"Chose: **{prob['chosen']}**")
            st.markdown(f"Expected value: ${prob['ev']:.2f}")

    # ---- Payoff distribution ----
    st.divider()
    st.subheader("Payoff distribution (1,000 simulations)")
    st.caption(
        "Your agent's choices determine which lotteries it entered. "
        "We simulated the random outcomes 1,000 times to show the "
        "range of possible payoffs."
    )

    fig1 = plot_total_payoff_distribution(
        simulation, display_name, budget_kept
    )
    st.pyplot(fig1)

    # ---- Problem breakdown ----
    st.subheader("Expected value by problem")
    fig2 = plot_problem_breakdown(simulation, display_name)
    st.pyplot(fig2)

    # ---- Comparison (if multiple runs) ----
    if len(st.session_state.history) > 1:
        st.divider()
        st.subheader("Compare agents")
        names = [h[0] for h in st.session_state.history]
        sims = [h[1] for h in st.session_state.history]
        bks = [h[2] for h in st.session_state.history]
        fig3 = plot_comparison(sims, names, bks)
        st.pyplot(fig3)

    # ---- Scorecard ----
    st.divider()
    import numpy as np

    total_payoffs = simulation["total_payoffs"]
    mean_earnings = np.mean(total_payoffs)
    preview_spent = sum(
        p["cost"] for p in st.session_state.preview_history
    )
    tier_cost = TIER_COSTS[sr["tier"]]

    st.subheader("Scorecard")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Starting Budget", f"${BASE_BUDGET:.2f}")
    sc2.metric("Previews Spent", f"${preview_spent:.2f}")
    sc3.metric("Tier Cost", f"${tier_cost:.2f}")
    sc4.metric("Budget Kept", f"${budget_kept:.2f}")
    sc5.metric("Mean Game Earnings", f"${mean_earnings:.2f}")

    st.markdown(
        f"### Expected Total Payout: **${budget_kept + mean_earnings:.2f}**"
    )

    # ---- Start over ----
    st.divider()
    if st.button("Start Over (new session)", use_container_width=True):
        for key in [
            "budget_remaining",
            "preview_history",
            "committed",
            "session_results",
            "selected_tier",
            "assigned_preset",
            "preview_run_counter",
            "history",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

elif not st.session_state.committed:
    st.markdown("---")
    st.markdown(
        "*Browse the agents above, optionally preview them, then commit "
        "to a tier and run the real 8 decisions.*"
    )
