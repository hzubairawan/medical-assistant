"""
Offline tests (no OpenAI key or internet needed).

Run from the project folder:   python -m pytest -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.globals import set_llm_cache
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src import cache_manager, chains, config, prompts, utils

SAMPLE_JSON = (
    '{"summary": "Mild cold.", "possible_conditions": [{"name": "Common cold", "reason": "Runny nose"}],'
    ' "urgency_level": "low", "recommended_next_steps": ["Rest"], "questions_for_doctor": ["Is it viral?"],'
    ' "warning_signs": ["High fever"]}'
)


def sample_inputs(**overrides):
    values = dict(
        age=25, gender="Female", symptoms=["Runny nose", "Sore throat"], duration="1-3 days",
        severity=2, conditions="", medications="", notes="", language="English",
    )
    values.update(overrides)
    return prompts.build_prompt_inputs(**values)


# --------------------------- JSON parsing ---------------------------------
def test_parse_plain_json():
    data, error = utils.safe_parse_json(SAMPLE_JSON)
    assert error is None and data["summary"] == "Mild cold."


def test_parse_json_with_fences_and_extra_text():
    raw = "Here you go:\n```json\n" + SAMPLE_JSON + "\n```\nHope it helps!"
    data, error = utils.safe_parse_json(raw)
    assert error is None and data is not None


def test_invalid_json_never_crashes():
    for raw in ["not json at all", "", None, "[1, 2, 3]", "{broken"]:
        data, error = utils.safe_parse_json(raw)
        assert data is None and error


def test_normalise_fills_missing_keys_and_fixes_urgency():
    result = utils.normalise_assessment({"urgency_level": "weird"})
    assert result["urgency_level"] == "MEDIUM"
    assert result["possible_conditions"] == [] and result["warning_signs"] == []
    assert utils.normalise_assessment({"urgency_level": "low"})["urgency_level"] == "LOW"


# --------------------------- Validation -----------------------------------
def test_empty_symptoms_and_bad_age_are_rejected():
    assert utils.validate_inputs("25", [])            # no symptoms
    assert utils.validate_inputs("abc", ["Cough"])    # bad age
    assert utils.validate_inputs("200", ["Cough"])    # impossible age
    assert utils.validate_inputs("25", ["Cough"]) == []


def test_combine_symptoms_removes_duplicates():
    assert utils.combine_symptoms(["Cough"], "cough, dry mouth") == ["Cough", "dry mouth"]


# --------------------------- Safety net -----------------------------------
def test_chest_pain_with_breathlessness_is_emergency():
    level, _ = utils.detect_red_flags(["Chest pain", "Shortness of breath"], 9, "", 50)
    assert level == "EMERGENCY"


def test_mild_cold_has_no_red_flags():
    level, _ = utils.detect_red_flags(["Runny nose", "Sore throat"], 2, "", 25)
    assert level is None


def test_override_raises_but_never_lowers():
    result = utils.normalise_assessment({"urgency_level": "LOW"})
    result, changed = utils.apply_safety_override(result, "EMERGENCY", ["x"])
    assert changed and result["urgency_level"] == "EMERGENCY"
    assert "emergency" in result["recommended_next_steps"][0].lower()

    result = utils.normalise_assessment({"urgency_level": "HIGH"})
    result, changed = utils.apply_safety_override(result, "LOW", [])
    assert not changed and result["urgency_level"] == "HIGH"


# --------------------------- Prompts --------------------------------------
def test_chat_template_renders_system_and_human_messages():
    messages = prompts.ASSESSMENT_CHAT_TEMPLATE.format_messages(**sample_inputs())
    assert [m.type for m in messages] == ["system", "human"]
    assert "NOT a doctor" in messages[0].content and "urgency_level" in messages[0].content
    assert "Runny nose" in messages[1].content


def test_prompt_template_has_all_variables():
    text = prompts.PATIENT_PROFILE_TEMPLATE.format(
        **{k: sample_inputs()[k] for k in prompts.PATIENT_PROFILE_TEMPLATE.input_variables}
    )
    assert "Age: 25" in text


# --------------------------- Chains + cache -------------------------------
def test_llmchain_runs_with_fake_model():
    cache_manager.configure_cache(config.CACHE_NONE)
    chain = chains.build_assessment_chain(FakeListChatModel(responses=[SAMPLE_JSON]))
    raw = chains.run_assessment(chain, sample_inputs())
    assert utils.safe_parse_json(raw)[0]["summary"] == "Mild cold."


def test_streaming_yields_pieces():
    cache_manager.configure_cache(config.CACHE_NONE)
    llm = FakeListChatModel(responses=["Hello patient"])
    pieces = list(chains.stream_narrative(llm, {**sample_inputs(), "urgency_level": "LOW"}))
    assert "".join(pieces) == "Hello patient" and len(pieces) > 1


def test_inmemory_cache_serves_second_identical_call():
    cache_manager.configure_cache(config.CACHE_MEMORY)
    cache_manager.clear_cache(config.CACHE_MEMORY)
    llm = FakeListChatModel(responses=["first answer", "second answer"])
    inputs = {**sample_inputs(), "urgency_level": "LOW"}

    first = "".join(chains.stream_narrative(llm, inputs))
    second = "".join(chains.stream_narrative(llm, inputs))  # replayed from cache
    assert first == second == "first answer"

    chain = chains.build_assessment_chain(FakeListChatModel(responses=[SAMPLE_JSON, "{}"]))
    one = chains.run_assessment(chain, sample_inputs())
    two = chains.run_assessment(chain, sample_inputs())  # LangChain cache hit
    assert one == two == SAMPLE_JSON
    set_llm_cache(None)


def test_sqlite_cache_persists_in_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "SQLITE_CACHE_PATH", tmp_path / "test_cache.db")
    monkeypatch.setattr(cache_manager, "_sqlite_cache", None)

    cache_manager.configure_cache(config.CACHE_SQLITE)
    llm = FakeListChatModel(responses=["stored answer", "other answer"])
    inputs = {**sample_inputs(), "urgency_level": "LOW"}
    assert "".join(chains.stream_narrative(llm, inputs)) == "stored answer"
    assert (tmp_path / "test_cache.db").exists()
    assert "".join(chains.stream_narrative(llm, inputs)) == "stored answer"
    set_llm_cache(None)


# --------------------------- Streamlit UI ---------------------------------
def test_app_warns_on_empty_symptoms_without_calling_api():
    from streamlit.testing.v1 import AppTest

    app_file = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30).run()
    assert not at.exception

    at.text_input(key="age").set_value("30")
    at.button(key="analyze_btn").click().run()
    assert not at.exception
    assert any("at least one symptom" in w.value for w in at.warning)


# --------------------------- Follow-up chat -------------------------------
def test_followup_messages_keep_conversation_order():
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    messages = chains.build_followup_messages(
        sample_inputs(), "LOW", "First answer", history, "new question"
    )
    assert [m.type for m in messages] == ["system", "human", "ai", "human", "ai", "human"]
    assert "LOW" in messages[0].content
    assert messages[2].content == "First answer"
    assert messages[-1].content == "new question"


def test_followup_stream_yields_text():
    llm = FakeListChatModel(responses=["Short answer"])
    messages = chains.build_followup_messages(sample_inputs(), "LOW", "x", [], "why?")
    assert "".join(chains.stream_followup(llm, messages)) == "Short answer"
