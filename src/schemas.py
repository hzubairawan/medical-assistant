"""
schemas.py
==========
Pydantic models that VALIDATE the JSON returned by the LLM.

Why Pydantic?
    LLM output is never 100% predictable. A Pydantic model gives us one clear
    place that guarantees: every key exists, every list really is a list,
    and the urgency level is always one of LOW / MEDIUM / HIGH / EMERGENCY.
    Bad or missing fields are repaired with safe defaults instead of crashing.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator

from src import config

_URGENCY_ALIASES = {
    "MODERATE": "MEDIUM",
    "URGENT": "HIGH",
    "SEVERE": "HIGH",
    "CRITICAL": "EMERGENCY",
}


def _clean_str_list(value) -> List[str]:
    """Accept a list OR a single string and return a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


class PossibleCondition(BaseModel):
    """One educational 'possible condition' (never a diagnosis)."""

    name: str
    reason: str = ""

    @field_validator("name", "reason", mode="before")
    @classmethod
    def _to_text(cls, value):
        return "" if value is None else str(value).strip()


class Assessment(BaseModel):
    """The exact structure the assignment requires (section 10)."""

    summary: str = "No summary was returned."
    possible_conditions: List[PossibleCondition] = Field(default_factory=list)
    urgency_level: str = "MEDIUM"
    recommended_next_steps: List[str] = Field(default_factory=list)
    questions_for_doctor: List[str] = Field(default_factory=list)
    warning_signs: List[str] = Field(default_factory=list)

    @field_validator("summary", mode="before")
    @classmethod
    def _summary_text(cls, value):
        text = "" if value is None else str(value).strip()
        return text or "No summary was returned."

    @field_validator("possible_conditions", mode="before")
    @classmethod
    def _conditions(cls, value):
        if not isinstance(value, (list, tuple)):
            return []
        cleaned = []
        for item in value:
            if isinstance(item, dict):
                name, reason = item.get("name"), item.get("reason")
            else:
                name, reason = item, ""
            if name is not None and str(name).strip():
                cleaned.append({"name": name, "reason": reason})
        return cleaned

    @field_validator(
        "recommended_next_steps", "questions_for_doctor", "warning_signs", mode="before"
    )
    @classmethod
    def _lists(cls, value):
        return _clean_str_list(value)

    @field_validator("urgency_level", mode="before")
    @classmethod
    def _urgency(cls, value):
        level = ("" if value is None else str(value)).strip().upper()
        level = _URGENCY_ALIASES.get(level, level)
        # Unknown value -> MEDIUM, a cautious middle ground.
        return level if level in config.URGENCY_LEVELS else "MEDIUM"


def validate_assessment(data: dict) -> dict:
    """Validate/repair a parsed JSON dict and return a plain dict."""
    return Assessment.model_validate(data).model_dump()
