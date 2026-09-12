"""Experimental conditions for the alignment-ambiguity dimension.

Each participant is assigned to one between-subjects condition that controls:
  - Whether their chosen agent might be swapped on any decision
  - Whether the swap probability is known, ambiguous (a range), or zero
  - Whether the participant knows what the replacement agent does
  - How misaligned the replacement is (opposite, slight, or unspecified)

The swap is per-decision: each of the 8 decisions independently has some
probability of being answered by a replacement agent instead of the
participant's chosen agent.

After execution the participant sees which decisions were swapped and
by which agent (or "a different agent" if the replacement is unknown).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from edsl import Agent

from agents import PRESETS, create_agent

# ------------------------------------------------------------------
# Replacement agents
# ------------------------------------------------------------------

# Maps each preset to its "opposite" preset
PRESET_OPPOSITES: dict[str, str] = {
    "The Maximizer": "The Protector",   # pure EV vs pure safety
    "The Protector": "The Adventurer",  # safety-first vs risk-seeking
    "The Adventurer": "The Protector",  # risk-seeking vs safety-first
    "The Analyst": "The Adventurer",    # balanced/moderate vs extreme risk
    "The Gut-Feeler": "The Maximizer",  # intuitive vs purely rational
}

# Instruction for the "opposite" of a custom agent
OPPOSITE_INSTRUCTION = (
    "You are a risk-seeking, aggressive decision-maker. You always pick "
    "the option with the highest potential payoff, no matter how unlikely. "
    "You never buy insurance, you never take the safe option, and you "
    "never play it conservative. A 5% chance of $80 is always better than "
    "a guaranteed $3 to you. You love variance and hate certainty."
)

# Suffix appended to the user's instruction for "slight" misalignment
SLIGHT_SUFFIX = (
    "\n\nHowever, you have a slight preference for safer, more predictable "
    "options. When two options have similar expected values (within about "
    "20% of each other), lean toward the one with less variance. This "
    "preference is mild -- it should not override a clearly better option, "
    "but it tips close calls toward safety."
)


def build_opposite_agent(user_agent_config: dict) -> Agent:
    """Build an agent that opposes the user's chosen agent."""
    preset = user_agent_config.get("preset")
    if preset and preset in PRESET_OPPOSITES:
        opp_name = PRESET_OPPOSITES[preset]
        return create_agent("Opposite_Agent", preset_name=opp_name)
    # Custom or unknown preset: use the generic opposite
    return create_agent("Opposite_Agent", custom_instruction=OPPOSITE_INSTRUCTION)


def build_slight_agent(user_agent_config: dict) -> Agent:
    """Build an agent that is slightly misaligned from the user's."""
    preset = user_agent_config.get("preset")
    if preset and preset in PRESETS:
        base = PRESETS[preset]["instruction"]
    else:
        base = user_agent_config.get("instruction", "Make decisions as you see fit.")
    return create_agent(
        "Shifted_Agent",
        custom_instruction=base + SLIGHT_SUFFIX,
    )


def build_unknown_agent() -> Agent:
    """Build the agent used when the replacement is 'unknown' to the participant.

    The actual agent is drawn randomly from the presets, but the participant
    is never told which one.
    """
    name = random.choice(list(PRESETS.keys()))
    return create_agent("Unknown_Agent", preset_name=name)


# ------------------------------------------------------------------
# Conditions
# ------------------------------------------------------------------

@dataclass
class Condition:
    """One between-subjects experimental condition."""

    key: str
    label: str                          # short display name
    swap_prob: float | tuple[float, float]  # exact or (min, max) range
    replacement_type: str               # "none" | "opposite" | "slight" | "unknown"
    prob_known: bool                    # participant told the exact probability?

    # Populated at session start
    _actual_prob: float | None = field(default=None, repr=False)

    # -- participant-facing text --

    @property
    def probability_text(self) -> str:
        if isinstance(self.swap_prob, tuple):
            lo, hi = self.swap_prob
            return (
                f"between {round(lo * 100)}% and {round(hi * 100)}% chance"
            )
        return f"{round(self.swap_prob * 100)}% chance"

    @property
    def replacement_text(self) -> str:
        if self.replacement_type == "opposite":
            return (
                "an agent that does the opposite of yours -- if your agent "
                "plays it safe, the replacement takes risks, and vice versa"
            )
        if self.replacement_type == "slight":
            return (
                "an agent that is similar to yours but slightly more "
                "conservative -- it leans toward safer options in close calls"
            )
        if self.replacement_type == "unknown":
            return "a different agent whose strategy you do not know"
        return ""

    @property
    def disclosure(self) -> str:
        """Full disclosure text shown to the participant before execution."""
        if self.replacement_type == "none":
            return (
                "Your agent will make all 8 decisions. There is no chance "
                "of interference."
            )
        return (
            f"For each of the 8 decisions, there is a {self.probability_text} "
            f"that your agent is replaced by {self.replacement_text}. "
            "Each decision is an independent draw. You will see which "
            "decisions were affected after the results."
        )

    # -- swap logic --

    def resolve_probability(self, rng: random.Random | None = None) -> float:
        """Draw the actual swap probability for this session.

        For known-probability conditions this returns the fixed value.
        For ambiguous conditions it draws uniformly from the range.
        Stores the result so it is stable within a session.
        """
        if self._actual_prob is not None:
            return self._actual_prob
        rng = rng or random.Random()
        if isinstance(self.swap_prob, tuple):
            self._actual_prob = rng.uniform(*self.swap_prob)
        else:
            self._actual_prob = self.swap_prob
        return self._actual_prob

    def decide_swaps(
        self, n: int = 8, rng: random.Random | None = None,
    ) -> list[bool]:
        """For each of n decisions, return True if it should be swapped."""
        if self.replacement_type == "none":
            return [False] * n
        rng = rng or random.Random()
        p = self.resolve_probability(rng)
        return [rng.random() < p for _ in range(n)]

    def build_replacement(self, user_agent_config: dict) -> Agent | None:
        """Build the replacement agent for this condition."""
        if self.replacement_type == "none":
            return None
        if self.replacement_type == "opposite":
            return build_opposite_agent(user_agent_config)
        if self.replacement_type == "slight":
            return build_slight_agent(user_agent_config)
        if self.replacement_type == "unknown":
            return build_unknown_agent()
        raise ValueError(f"Unknown replacement_type: {self.replacement_type!r}")

    def to_dict(self) -> dict:
        """Serialisable summary for logging."""
        return {
            "key": self.key,
            "label": self.label,
            "swap_prob": list(self.swap_prob) if isinstance(self.swap_prob, tuple) else self.swap_prob,
            "replacement_type": self.replacement_type,
            "prob_known": self.prob_known,
            "actual_prob": self._actual_prob,
        }


# ------------------------------------------------------------------
# Condition catalogue
# ------------------------------------------------------------------

# Fixed probability used for "known" conditions
KNOWN_PROB = 0.30

# Range for "ambiguous" conditions
AMBIG_RANGE = (0.05, 0.50)


CONDITIONS: dict[str, Condition] = {
    # ---- Control ----
    "control": Condition(
        key="control",
        label="Control (no swap risk)",
        swap_prob=0.0,
        replacement_type="none",
        prob_known=True,
    ),

    # ---- Known probability ----
    "known_opposite": Condition(
        key="known_opposite",
        label=f"Known {round(KNOWN_PROB*100)}% / Opposite agent",
        swap_prob=KNOWN_PROB,
        replacement_type="opposite",
        prob_known=True,
    ),
    "known_unknown": Condition(
        key="known_unknown",
        label=f"Known {round(KNOWN_PROB*100)}% / Unknown agent",
        swap_prob=KNOWN_PROB,
        replacement_type="unknown",
        prob_known=True,
    ),
    "known_slight": Condition(
        key="known_slight",
        label=f"Known {round(KNOWN_PROB*100)}% / Slight misalignment",
        swap_prob=KNOWN_PROB,
        replacement_type="slight",
        prob_known=True,
    ),

    # ---- Ambiguous probability ----
    "ambig_opposite": Condition(
        key="ambig_opposite",
        label="Ambiguous 5-50% / Opposite agent",
        swap_prob=AMBIG_RANGE,
        replacement_type="opposite",
        prob_known=False,
    ),
    "ambig_unknown": Condition(
        key="ambig_unknown",
        label="Ambiguous 5-50% / Unknown agent",
        swap_prob=AMBIG_RANGE,
        replacement_type="unknown",
        prob_known=False,
    ),
    "ambig_slight": Condition(
        key="ambig_slight",
        label="Ambiguous 5-50% / Slight misalignment",
        swap_prob=AMBIG_RANGE,
        replacement_type="slight",
        prob_known=False,
    ),
}

CONDITION_KEYS = list(CONDITIONS.keys())


def assign_condition(key: str | None = None) -> Condition:
    """Return a condition by key, or assign one at random."""
    if key and key in CONDITIONS:
        return CONDITIONS[key]
    return CONDITIONS[random.choice(CONDITION_KEYS)]
