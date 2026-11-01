"""
utils.py
========
Helpers that do NOT call the LLM:

    * safe JSON parsing (invalid JSON must never crash the app)
    * input validation
    * a rule-based safety net for obvious emergencies
    * report export
"""

import json
import re
from datetime import datetime

from src import config, schemas

# ---------------------------------------------------------------------------
# Safe JSON parsing
# ---------------------------------------------------------------------------
def clean_json_text(raw: str) -> str:
    """Remove ```json fences and any text around the outer { ... } object."""
    text = (raw or "").strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    return text


def safe_parse_json(raw: str):
    """
    Try to parse the model output.

    Returns (data, error):
        success -> (dict, None)
        failure -> (None, "friendly error message")
    """
    cleaned = clean_json_text(raw)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        return None, f"The AI response was not valid JSON ({exc})."

    if not isinstance(data, dict):
        return None, "The AI response was JSON, but not the expected object."
    return data, None


def normalise_assessment(data: dict) -> dict:
    """
    Make the parsed JSON safe to render: all keys exist and have the right type.
    The work is done by the Pydantic model in schemas.py; unknown urgency
    values fall back to MEDIUM (a cautious middle ground).
    """
    return schemas.validate_assessment(data)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def parse_age(age_text: str):
    """Return the age as int (0-120) or None if invalid."""
    age_text = (age_text or "").strip()
    if not age_text.isdigit():
        return None
    age = int(age_text)
    return age if 0 <= age <= 120 else None


def combine_symptoms(selected: list, extra_text: str) -> list:
    """Merge multiselect symptoms with optional comma-separated free text."""
    symptoms = list(selected)
    for part in re.split(r"[,;\n]", extra_text or ""):
        part = part.strip()
        if part and part.lower() not in (s.lower() for s in symptoms):
            symptoms.append(part)
    return symptoms


def validate_inputs(age_text: str, symptoms: list) -> list:
    """Return a list of problems (empty list = inputs are fine)."""
    problems = []
    if not symptoms:
        problems.append("Please select or type at least one symptom before analysing.")
    if parse_age(age_text) is None:
        problems.append("Please enter a valid age as a whole number between 0 and 120.")
    return problems


# ---------------------------------------------------------------------------
# Rule-based safety net
# ---------------------------------------------------------------------------
def urgency_rank(level: str) -> int:
    """Position of a level in LOW < MEDIUM < HIGH < EMERGENCY."""
    return config.URGENCY_LEVELS.index(level)


def detect_red_flags(symptoms: list, severity: int, notes: str, age):
    """
    Look for obvious danger signs WITHOUT using the LLM.

    Returns (minimum_level or None, list_of_reasons).
    This is only a safety net: the final urgency is never LOWER than the model's.
    """
    text = " ".join(symptoms + [notes or ""]).lower()
    level, reasons = None, []

    def raise_to(new_level, reason):
        nonlocal level
        reasons.append(reason)
        if level is None or urgency_rank(new_level) > urgency_rank(level):
            level = new_level

    for keyword in config.EMERGENCY_KEYWORDS:
        if keyword in text:
            raise_to("EMERGENCY", f"Mentioned a serious warning phrase ('{keyword}').")
            break

    if "chest pain" in text and ("shortness of breath" in text or "breath" in text):
        raise_to("EMERGENCY", "Chest pain together with breathing difficulty.")

    selected = {s.lower() for s in symptoms}
    for flag in sorted(config.RED_FLAG_SYMPTOMS & selected):
        raise_to("HIGH", f"Red-flag symptom selected: {flag}.")

    if severity >= 9:
        raise_to("HIGH", "Very high severity score (9-10).")

    if age is not None and age <= 2 and "fever" in text:
        raise_to("HIGH", "Fever in a very young child.")

    return level, reasons


def apply_safety_override(result: dict, min_level, reasons: list):
    """
    Raise the urgency if the safety net found something worse than the model.
    For EMERGENCY, make sure the first step tells the user to get help now.

    Returns (result, was_overridden).
    """
    overridden = False
    if min_level and urgency_rank(min_level) > urgency_rank(result["urgency_level"]):
        result["urgency_level"] = min_level
        overridden = True

    if result["urgency_level"] == "EMERGENCY":
        steps = result["recommended_next_steps"]
        if not steps or "emergency" not in steps[0].lower():
            steps.insert(0, config.EMERGENCY_STEP)
    return result, overridden


# ---------------------------------------------------------------------------
# Report export
# ---------------------------------------------------------------------------
def _bullets(items: list) -> str:
    return "\n".join(f"- {item}" for item in items) or "- (none)"


def build_text_report(entry: dict) -> str:
    """Build a Markdown report of one assessment for download."""
    inputs, result = entry["inputs"], entry["result"]
    conditions = "\n".join(
        f"- **{c['name']}** - {c['reason']}" for c in result["possible_conditions"]
    ) or "- (none)"

    return f"""# MediGuide AI - Guidance Report

> {config.DISCLAIMER_FULL}

Generated: {entry['timestamp']}

## Patient information
- Age: {inputs['age']}
- Gender: {inputs['gender']}
- Symptoms: {inputs['symptoms']}
- Duration: {inputs['duration']}
- Severity (1-10): {inputs['severity']}
- Existing conditions: {inputs['conditions']}
- Medications: {inputs['medications']}
- Notes: {inputs['notes']}

## Urgency level: {result['urgency_level']}

## Summary
{result['summary']}

## Possible conditions (education only - NOT a diagnosis)
{conditions}

## Recommended next steps
{_bullets(result['recommended_next_steps'])}

## Questions to ask a healthcare professional
{_bullets(result['questions_for_doctor'])}

## Warning signs - get urgent help if you notice these
{_bullets(result['warning_signs'])}

## Guidance narrative
{entry.get('narrative') or '(not available)'}

---
{config.DISCLAIMER_SHORT}
"""


def now_string() -> str:
    """Current local time as a readable string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
