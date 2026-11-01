"""
Pure-Python tests (Pydantic schema, PDF report, demo scenarios, safety net).
No OpenAI key and no internet needed.

Run from the project folder:   python -m pytest -q
"""

import pytest

from src import config, report, schemas, utils


def make_entry(**overrides):
    result = utils.normalise_assessment({
        "summary": "You have a fever and a cough.",
        "possible_conditions": [{"name": "Flu", "reason": "Fever and cough together"}],
        "urgency_level": "MEDIUM",
        "recommended_next_steps": ["Rest", "Drink fluids"],
        "questions_for_doctor": ["Is it viral?"],
        "warning_signs": ["Trouble breathing"],
    })
    entry = {
        "inputs": {
            "age": "40", "gender": "Male", "symptoms": "Fever, Cough", "duration": "4-7 days",
            "severity": 6, "conditions": "None reported", "medications": "None reported",
            "notes": "None reported", "language": "English",
        },
        "result": result,
        "narrative": "### Overview\n- Could be **viral**\nPlain line & text",
        "timestamp": "2026-01-01 10:00:00",
        "model": "gpt-4o-mini",
    }
    entry.update(overrides)
    return entry


# ------------------------------ Pydantic ----------------------------------
def test_schema_repairs_messy_model_output():
    result = schemas.validate_assessment({
        "urgency_level": "urgent",
        "possible_conditions": ["Flu", {"name": "Cold", "reason": None}, {"name": ""}],
        "warning_signs": "fever",
        "recommended_next_steps": None,
        "summary": None,
    })
    assert result["urgency_level"] == "HIGH"
    assert [c["name"] for c in result["possible_conditions"]] == ["Flu", "Cold"]
    assert result["warning_signs"] == ["fever"]
    assert result["recommended_next_steps"] == []
    assert result["summary"] == "No summary was returned."


def test_schema_unknown_urgency_defaults_to_medium():
    assert schemas.validate_assessment({})["urgency_level"] == "MEDIUM"
    assert schemas.validate_assessment({"urgency_level": "banana"})["urgency_level"] == "MEDIUM"


def test_schema_output_has_exact_assignment_keys():
    keys = set(schemas.validate_assessment({}).keys())
    assert keys == {
        "summary", "possible_conditions", "urgency_level",
        "recommended_next_steps", "questions_for_doctor", "warning_signs",
    }


# ------------------------------ Scenarios ---------------------------------
def test_demo_scenarios_only_use_valid_options():
    for name, scenario in config.DEMO_SCENARIOS.items():
        assert set(scenario["symptoms"]) <= set(config.SYMPTOM_OPTIONS), name
        assert scenario["gender"] in config.GENDER_OPTIONS, name
        assert scenario["duration"] in config.DURATION_OPTIONS, name
        assert scenario["language"] in config.LANGUAGE_OPTIONS, name
        assert 1 <= scenario["severity"] <= 10, name
        assert utils.validate_inputs(scenario["age"], scenario["symptoms"]) == [], name


def test_safety_net_on_demo_scenarios():
    scenarios = config.DEMO_SCENARIOS

    def level(name):
        s = scenarios[name]
        return utils.detect_red_flags(s["symptoms"], s["severity"], s["notes"], int(s["age"]))[0]

    assert level("1 · Mild cold (LOW)") is None
    assert level("3 · Chest pain (EMERGENCY)") == "EMERGENCY"
    assert level("4 · Toddler with fever (safety net)") == "HIGH"


# ------------------------------ PDF ---------------------------------------
def test_pdf_report_is_created():
    data = report.build_pdf_report(make_entry())
    assert data[:4] == b"%PDF" and len(data) > 2000


def test_pdf_handles_emoji_and_special_characters():
    entry = make_entry()
    entry["result"]["summary"] = "Fever — and cough \U0001F637 <b>tags</b> & symbols"
    assert report.build_pdf_report(entry)[:4] == b"%PDF"


def test_pdf_refuses_non_latin_languages_cleanly():
    entry = make_entry(narrative="یہ ایک اردو متن ہے" * 5)
    entry["result"]["summary"] = "بخار اور کھانسی کی علامات ہیں"
    with pytest.raises(report.PdfUnsupported):
        report.build_pdf_report(entry)


def test_markdown_report_contains_disclaimer_and_urgency():
    text = utils.build_text_report(make_entry())
    assert "NOT" in text and "MEDIUM" in text and "Flu" in text
