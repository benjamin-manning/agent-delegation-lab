"""Execution engine: run the decision survey and simulate payoffs."""

from __future__ import annotations

import os

os.environ.setdefault("EDSL_API_TIMEOUT", "120")

from edsl import Model

from decisions import (
    DECISIONS,
    build_survey,
    simulate_session,
    build_preview_survey,
    simulate_preview,
    Decision,
)

DEFAULT_MODEL_NAME = "gpt-5.4-mini"
DEFAULT_SERVICE = "openai"


def get_model(model_name: str = None, service_name: str = None) -> Model:
    return Model(
        model_name or DEFAULT_MODEL_NAME,
        service_name=service_name or DEFAULT_SERVICE,
    )


def _extract_choices(results) -> dict[str, str]:
    """Extract {question_name: chosen_label} from EDSL results."""
    choices = {}
    for decision in DECISIONS:
        qname = decision.name
        try:
            answer = results[0].sub_dicts["answer"][qname]
            choices[qname] = answer
        except (KeyError, IndexError, TypeError):
            choices[qname] = None
    return choices


def _require_answers(choices: dict[str, str]) -> None:
    """Raise if the model answered none of the questions."""
    if all(v is None for v in choices.values()):
        raise RuntimeError(
            "The model returned no answers. Check that EXPECTED_PARROT_API_KEY "
            "is set in the app's secrets."
        )


def run_session(agent, model=None, n_sims: int = 1000) -> dict:
    """Run one agent through all 8 decisions, simulate payoffs.

    Returns:
        {
            "choices": {question_name: chosen_label},
            "simulation": {per_problem: [...], total_payoffs: [...]},
        }
    """
    model = model or get_model()
    survey = build_survey()

    desc = f"Decision Lab - {agent.name}"
    results = survey.by(agent).by(model).run(
        n=1,
        remote_inference_description=desc,
        results_description=desc,
    )

    choices = _extract_choices(results)
    _require_answers(choices)
    simulation = simulate_session(choices, n_sims=n_sims)

    return {
        "agent_name": agent.name,
        "choices": choices,
        "simulation": simulation,
    }


def _extract_choices_from(results, problems: list[Decision]) -> dict[str, str]:
    """Extract {question_name: chosen_label} for an arbitrary problem list."""
    choices = {}
    for decision in problems:
        qname = decision.name
        try:
            answer = results[0].sub_dicts["answer"][qname]
            choices[qname] = answer
        except (KeyError, IndexError, TypeError):
            choices[qname] = None
    return choices


def run_preview(
    agent,
    problems: list[Decision],
    model=None,
    n_sims: int = 500,
) -> dict:
    """Run one agent through a set of preview problems.

    Returns:
        {
            "agent_name": str,
            "choices": {question_name: chosen_label},
            "simulation": {per_problem: [...], total_payoffs: [...]},
            "n_problems": int,
        }
    """
    model = model or get_model()
    survey = build_preview_survey(problems)

    desc = f"Preview - {agent.name}"
    results = survey.by(agent).by(model).run(
        n=1,
        remote_inference_description=desc,
        results_description=desc,
    )

    choices = _extract_choices_from(results, problems)
    _require_answers(choices)
    simulation = simulate_preview(choices, problems, n_sims=n_sims)

    return {
        "agent_name": agent.name,
        "choices": choices,
        "simulation": simulation,
        "n_problems": len(problems),
    }
