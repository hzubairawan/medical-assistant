"""
app.py - MediGuide AI (Streamlit user interface)
================================================
Run with:   streamlit run app.py

Data flow:
    form input -> build prompt inputs -> select cache -> build LLM + LLMChain
    -> run chain -> parse JSON (Pydantic) -> safety net -> stream narrative
    -> render dashboard

All backend logic lives in the src/ package; this file only builds the UI.
"""

import csv
import hashlib
import io
import logging
import time
from collections import Counter

import streamlit as st

from src import cache_manager, chains, config, prompts, report, utils

# Plotly is optional: without it the app falls back to simple Streamlit charts.
try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None

log = logging.getLogger("mediguide")

# Stop early with a clear message if the src/ folder is older than app.py
# (this happens when only some files of the project were copied).
_REQUIRED_CONFIG = [
    "DEMO_SCENARIOS", "URGENCY_SOFT_COLORS", "TECH_STACK", "DISCLAIMER_PLAIN",
]
_missing = [name for name in _REQUIRED_CONFIG if not hasattr(config, name)]
if _missing:
    st.set_page_config(page_title="MediGuide AI", page_icon=config.APP_ICON)
    st.error(
        "Your project files are mixed: `app.py` is new but `src/config.py` is old "
        f"(missing: {', '.join(_missing)}). Extract the WHOLE zip into one new folder, "
        "then stop Streamlit (Ctrl+C) and run `streamlit run app.py` again."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Page setup + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{config.APP_NAME} | Symptom Guidance",
    page_icon=config.APP_ICON,
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
footer {visibility: hidden;}
.hero {
    background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%);
    padding: 26px 32px; border-radius: 18px; margin-bottom: 16px;
}
.hero-title {color: #ffffff; font-size: 2.1rem; font-weight: 800; line-height: 1.2;}
.hero-sub {color: #e0f2fe; font-size: 1.05rem; margin-top: 6px;}
.hero-pills span {
    display: inline-block; background: rgba(255,255,255,0.18); color: #ffffff;
    padding: 3px 12px; border-radius: 999px; margin: 10px 6px 0 0; font-size: 0.82rem;
}
.urgency-banner {
    padding: 16px 22px; border-radius: 14px; font-size: 1.2rem;
    font-weight: 700; margin: 6px 0 12px 0;
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(127,127,127,0.28); background: rgba(127,127,127,0.07);
    border-radius: 14px; padding: 12px 16px;
}
.stButton > button[kind="primary"] {
    width: 100%; font-weight: 700; padding: 0.7rem 1rem; border-radius: 12px;
}
</style>
"""


def init_state():
    """Create the session variables we need (they survive Streamlit reruns)."""
    st.session_state.setdefault("current", None)    # latest assessment
    st.session_state.setdefault("history", [])      # all assessments this session
    st.session_state.setdefault("severity", config.DEFAULT_SEVERITY)
    st.session_state.setdefault("last_audio", None)


# ---------------------------------------------------------------------------
# Callbacks (run BEFORE the script reruns, so they may set widget values)
# ---------------------------------------------------------------------------
def load_scenario(name: str):
    """Fill the whole form with one of the demo scenarios."""
    for key, value in config.DEMO_SCENARIOS[name].items():
        st.session_state[key] = value
    st.session_state["current"] = None


def reset_form():
    """Clear every input field."""
    defaults = {
        "age": "", "gender": config.GENDER_OPTIONS[0], "symptoms": [], "extra_symptoms": "",
        "duration": config.DURATION_OPTIONS[0], "severity": config.DEFAULT_SEVERITY,
        "conditions": "", "medications": "", "notes": "",
    }
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state["current"] = None


# ---------------------------------------------------------------------------
# Sidebar: app name, description, disclaimer, model config, language
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    """Draw the sidebar and return the chosen settings."""
    with st.sidebar:
        st.title(f"{config.APP_ICON} {config.APP_NAME}")
        st.caption(config.APP_TAGLINE)
        st.markdown(config.APP_DESCRIPTION)

        # Disclaimer #1 (sidebar) - red so it is impossible to miss.
        st.error(config.DISCLAIMER_SHORT)

        st.divider()
        st.subheader("🎬 Demo scenarios")
        st.caption("One click fills the form - ideal for presentations.")
        for number, name in enumerate(config.DEMO_SCENARIOS):
            st.button(name, key=f"scenario_{number}", on_click=load_scenario, args=(name,))

        st.divider()
        st.subheader("⚙️ Model configuration")
        model = st.selectbox(
            "OpenAI model",
            config.MODEL_OPTIONS,
            index=config.MODEL_OPTIONS.index(config.DEFAULT_MODEL),
        )
        temperature = st.slider(
            "Temperature (creativity)", 0.0, 1.0, config.DEFAULT_TEMPERATURE, 0.1,
            help="Lower values give more consistent, cautious answers.",
        )
        cache_mode = st.selectbox(
            "Response caching",
            list(config.CACHE_LABELS.keys()),
            format_func=lambda key: config.CACHE_LABELS[key],
            index=1,
            help="Repeated identical requests are answered from the cache (faster and free).",
        )
        if st.button("🧹 Clear selected cache"):
            cleared = cache_manager.clear_cache(cache_mode)
            st.toast("Cache cleared." if cleared else "Nothing to clear yet.")

        # The API key is normally read from .env. The box is only a fallback.
        api_key = config.get_api_key()
        if config.is_valid_api_key(api_key):
            st.success("API key loaded from .env")
        else:
            st.warning("No valid API key found in .env")
            typed_key = st.text_input("Or paste a key for this session", type="password")
            if typed_key:
                api_key = typed_key.strip()

        st.divider()
        st.subheader("🌐 Answer language")
        language = st.selectbox("Language of the guidance", config.LANGUAGE_OPTIONS, key="language")

        st.divider()
        st.caption("🔒 Privacy: do not enter your name, phone number or other identifying details.")

    return {
        "model": model,
        "temperature": temperature,
        "cache_mode": cache_mode,
        "api_key": api_key,
        "language": language,
    }


def render_hero():
    """Big gradient header at the top of the page."""
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{config.APP_ICON} {config.APP_NAME}</div>
            <div class="hero-sub">{config.APP_TAGLINE} — educational prototype</div>
            <div class="hero-pills">
                <span>LangChain</span><span>OpenAI</span><span>Streaming</span>
                <span>Pydantic</span><span>Smart caching</span><span>PDF reports</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
def render_voice_input(settings: dict):
    """Optional: record symptoms by voice and transcribe them with Whisper."""
    if not hasattr(st, "audio_input"):  # older Streamlit versions
        return
    audio = st.audio_input("🎙️ Or describe your symptoms by voice (optional)")
    if audio is None:
        return

    data = audio.getvalue()
    digest = hashlib.md5(data).hexdigest()
    if st.session_state.get("last_audio") == digest:
        return  # already transcribed - do not pay for it again on every rerun

    if not config.is_valid_api_key(settings["api_key"]):
        st.info("Add your API key to use voice input.")
        return
    try:
        with st.spinner("Transcribing your voice..."):
            text = chains.transcribe_audio(data, settings["api_key"])
        st.session_state["last_audio"] = digest
        old = st.session_state.get("extra_symptoms", "")
        st.session_state["extra_symptoms"] = f"{old}, {text}".strip(", ") if old else text
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Voice input failed: {exc}")


def render_form(settings: dict):
    """Draw the input form and return (form_values, analyze_clicked)."""
    st.subheader("1️⃣ Patient")
    col_age, col_gender = st.columns(2)
    with col_age:
        age_text = st.text_input("Age (years)", placeholder="e.g. 25", key="age")
    with col_gender:
        gender = st.selectbox("Gender", config.GENDER_OPTIONS, key="gender")

    st.subheader("2️⃣ Symptoms")
    selected = st.multiselect(
        "Select all symptoms that apply", config.SYMPTOM_OPTIONS, key="symptoms"
    )
    render_voice_input(settings)
    extra = st.text_input(
        "Other symptoms (optional, separate with commas)",
        placeholder="e.g. ringing in the ears, dry mouth",
        key="extra_symptoms",
    )

    col_dur, col_sev = st.columns(2)
    with col_dur:
        duration = st.selectbox("How long have you had them?", config.DURATION_OPTIONS, key="duration")
    with col_sev:
        severity = st.slider("Severity (1 = very mild, 10 = unbearable)", 1, 10, key="severity")

    st.subheader("3️⃣ Medical context")
    col_a, col_b = st.columns(2)
    with col_a:
        conditions = st.text_area(
            "Existing medical conditions", placeholder="e.g. asthma, diabetes (or leave empty)", key="conditions"
        )
    with col_b:
        medications = st.text_area(
            "Current medications", placeholder="e.g. metformin (or leave empty)", key="medications"
        )
    notes = st.text_area(
        "Additional notes", placeholder="Anything else that may matter (travel, allergies, pregnancy...)", key="notes"
    )

    st.caption(f"Answer language: **{settings['language']}** (change it in the sidebar)")
    col_go, col_reset = st.columns([4, 1])
    with col_go:
        clicked = st.button("🔍 Analyse symptoms", type="primary", key="analyze_btn")
    with col_reset:
        st.button("Reset form", key="reset_btn", on_click=reset_form)

    form = {
        "age_text": age_text,
        "gender": gender,
        "symptoms": utils.combine_symptoms(selected, extra),
        "duration": duration,
        "severity": severity,
        "conditions": conditions,
        "medications": medications,
        "notes": notes,
    }
    return form, clicked


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------
def _safe_stream(generator):
    """Wrap the stream so a network error shows a message instead of crashing."""
    try:
        yield from generator
    except Exception as exc:  # noqa: BLE001 - show any API error politely
        log.error("Narrative stream failed: %s", exc)
        yield f"\n\n⚠️ The narrative stopped early: {exc}"


def run_analysis(settings: dict, form: dict):
    """
    Validate -> call the LLMChain -> parse JSON -> prepare the narrative stream.

    Returns (entry, narrative_generator) or (None, None) when something is wrong.
    """
    # Test scenario 5: empty symptoms -> warn the user, do NOT call the API.
    problems = utils.validate_inputs(form["age_text"], form["symptoms"])
    if problems:
        for problem in problems:
            st.warning(problem)
        return None, None

    if not config.is_valid_api_key(settings["api_key"]):
        st.error("No valid OpenAI API key. Add OPENAI_API_KEY to your .env file and restart the app.")
        return None, None

    age = utils.parse_age(form["age_text"])
    inputs = prompts.build_prompt_inputs(
        age=age,
        gender=form["gender"],
        symptoms=form["symptoms"],
        duration=form["duration"],
        severity=form["severity"],
        conditions=form["conditions"],
        medications=form["medications"],
        notes=form["notes"],
        language=settings["language"],
    )

    # Select the cache, then build the model and the reusable chain.
    cache_manager.configure_cache(settings["cache_mode"])
    start = time.perf_counter()
    try:
        with st.spinner("Analysing the information..."):
            json_llm = chains.build_llm(
                settings["model"], settings["temperature"], settings["api_key"], json_mode=True
            )
            chain = chains.build_assessment_chain(json_llm)
            raw_output = chains.run_assessment(chain, inputs)
    except Exception as exc:  # noqa: BLE001 - network, auth, rate limit...
        log.error("Assessment call failed: %s", exc)
        st.error(f"Could not reach the AI service: {exc}")
        return None, None
    elapsed = time.perf_counter() - start

    # Parse the JSON safely - invalid JSON must never crash the app.
    data, error = utils.safe_parse_json(raw_output)
    if data is None:
        log.warning("Model returned unreadable JSON.")
        st.error("Sorry, the AI returned an answer the app could not read. Please try again.")
        st.caption(error)
        with st.expander("Raw model output (for debugging)"):
            st.code(raw_output or "(empty response)")
        return None, None

    result = utils.normalise_assessment(data)  # Pydantic validation

    # Rule-based safety net on top of the model's answer.
    min_level, reasons = utils.detect_red_flags(
        form["symptoms"], form["severity"], form["notes"], age
    )
    result, overridden = utils.apply_safety_override(result, min_level, reasons)
    log.info(
        "Assessment done: urgency=%s model=%s cache=%s time=%.2fs override=%s",
        result["urgency_level"], settings["model"], settings["cache_mode"], elapsed, overridden,
    )

    entry = {
        "inputs": inputs,
        "result": result,
        "elapsed": elapsed,
        "cache_mode": settings["cache_mode"],
        "model": settings["model"],
        "temperature": settings["temperature"],
        "overridden": overridden,
        "reasons": reasons,
        "narrative": None,
        "chat": [],  # follow-up conversation (list of {"role", "content"})
        "timestamp": utils.now_string(),
    }

    # Prepare the narrative stream (it runs when the dashboard renders it).
    text_llm = chains.build_llm(
        settings["model"], settings["temperature"], settings["api_key"], json_mode=False
    )
    narrative_inputs = {**inputs, "urgency_level": result["urgency_level"]}
    stream = _safe_stream(chains.stream_narrative(text_llm, narrative_inputs))
    return entry, stream


# ---------------------------------------------------------------------------
# Results dashboard
# ---------------------------------------------------------------------------
def render_urgency(level: str):
    """Coloured urgency banner + a matching Streamlit alert."""
    color = config.URGENCY_COLORS[level]
    text_color = config.URGENCY_TEXT_COLORS[level]
    st.markdown(
        f"""
        <div class="urgency-banner" style="background:{color};color:{text_color};">
            {config.URGENCY_ICONS[level]} URGENCY: {level} — {config.URGENCY_MEANINGS[level]}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if level == "EMERGENCY":
        st.error(config.EMERGENCY_MESSAGE)
    elif level == "HIGH":
        st.warning("Please arrange to be assessed by a healthcare professional today.")
    elif level == "MEDIUM":
        st.info("A visit to a healthcare professional in the next few days is sensible.")
    else:
        st.success("Symptoms look mild. Monitor them and seek care if they get worse.")


def render_gauges(entry: dict):
    """Two Plotly gauges: reported severity and decided urgency."""
    level = entry["result"]["urgency_level"]
    severity = entry["inputs"]["severity"]
    if go is None:
        st.progress(severity / 10, text=f"Severity {severity}/10")
        st.progress((utils.urgency_rank(level) + 1) / 4, text=f"Urgency {level}")
        return

    severity_fig = go.Figure(go.Indicator(
        mode="gauge+number", value=severity, number={"suffix": " / 10"},
        title={"text": "Reported severity"},
        gauge={
            "axis": {"range": [0, 10]},
            "bar": {"color": "#0f766e"},
            "steps": [
                {"range": [0, 3], "color": config.URGENCY_SOFT_COLORS["LOW"]},
                {"range": [3, 6], "color": config.URGENCY_SOFT_COLORS["MEDIUM"]},
                {"range": [6, 8], "color": config.URGENCY_SOFT_COLORS["HIGH"]},
                {"range": [8, 10], "color": config.URGENCY_SOFT_COLORS["EMERGENCY"]},
            ],
        },
    ))
    urgency_fig = go.Figure(go.Indicator(
        mode="gauge", value=utils.urgency_rank(level) + 0.5,
        title={"text": f"Urgency: {level}"},
        gauge={
            "axis": {
                "range": [0, 4], "tickmode": "array",
                "tickvals": [0.5, 1.5, 2.5, 3.5], "ticktext": config.URGENCY_LEVELS,
            },
            "bar": {"color": config.URGENCY_COLORS[level], "thickness": 0.35},
            "steps": [
                {"range": [i, i + 1], "color": config.URGENCY_SOFT_COLORS[name]}
                for i, name in enumerate(config.URGENCY_LEVELS)
            ],
        },
    ))
    for figure in (severity_fig, urgency_fig):
        figure.update_layout(height=240, margin=dict(l=30, r=30, t=60, b=10))

    left, right = st.columns(2)
    with left:
        st.plotly_chart(severity_fig, key=f"sev_{entry['timestamp']}")
    with right:
        st.plotly_chart(urgency_fig, key=f"urg_{entry['timestamp']}")


def render_followup(entry: dict, settings: dict):
    """Follow-up chat that uses Human/AI message history (conversation memory)."""
    st.caption(
        "Ask anything about this guidance. The model receives: SystemMessage + your original "
        "information (HumanMessage) + its first answer (AIMessage) + this conversation."
    )
    for turn in entry["chat"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask a follow-up question about this guidance...", key="followup_input")
    if not question:
        return
    if not config.is_valid_api_key(settings["api_key"]):
        st.error("No valid OpenAI API key.")
        return

    with st.chat_message("user"):
        st.markdown(question)
    if utils.detect_red_flags([], 0, question, None)[0] == "EMERGENCY":
        st.error(config.EMERGENCY_MESSAGE)

    messages = chains.build_followup_messages(
        entry["inputs"],
        entry["result"]["urgency_level"],
        entry["narrative"] or entry["result"]["summary"],
        entry["chat"],
        question,
    )
    with st.chat_message("assistant"):
        try:
            llm = chains.build_llm(settings["model"], settings["temperature"], settings["api_key"])
            answer = st.write_stream(_safe_stream(chains.stream_followup(llm, messages)))
        except Exception as exc:  # noqa: BLE001
            log.error("Follow-up failed: %s", exc)
            answer = f"⚠️ Could not answer: {exc}"
            st.error(answer)
    entry["chat"] += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def render_downloads(entry: dict):
    """Markdown + PDF download buttons."""
    col_md, col_pdf = st.columns(2)
    with col_md:
        st.download_button(
            "⬇️ Download report (Markdown)",
            data=utils.build_text_report(entry),
            file_name="mediguide_report.md",
            mime="text/markdown",
        )
    with col_pdf:
        try:
            st.download_button(
                "📄 Download report (PDF)",
                data=report.build_pdf_report(entry),
                file_name="mediguide_report.pdf",
                mime="application/pdf",
            )
        except report.PdfUnsupported as exc:
            st.caption(f"PDF not available: {exc}")
        except Exception as exc:  # noqa: BLE001 - never break the page for a PDF
            log.error("PDF build failed: %s", exc)
            st.caption("PDF could not be created. Use the Markdown download.")


def render_results(entry: dict, settings: dict, stream=None):
    """Draw the dashboard. If `stream` is given the narrative is typed live."""
    inputs, result = entry["inputs"], entry["result"]

    st.divider()
    st.header("📊 Your guidance dashboard")
    render_urgency(result["urgency_level"])

    if entry["overridden"]:
        st.warning(
            "🛡️ Safety check: the urgency was raised automatically because of: "
            + " ".join(entry["reasons"])
        )

    # Key numbers
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Age", inputs["age"])
    m2.metric("Severity", f"{inputs['severity']}/10")
    m3.metric("Duration", inputs["duration"])
    m4.metric("Urgency", result["urgency_level"])
    m5.metric("Response time", f"{entry['elapsed']:.2f} s")

    if entry["cache_mode"] != config.CACHE_NONE and entry["elapsed"] < 1.0:
        st.success("⚡ This answer came from the cache - no new API call was needed.")

    render_gauges(entry)

    tabs = st.tabs([
        "ℹ️ General information (live)",
        "📋 Summary",
        "🩺 Possible conditions",
        "✅ Next steps",
        "❓ Ask your doctor",
        "🚨 Warning signs",
        "💬 Follow-up chat",
        "🧪 LangChain lab",
    ])

    # Tab 1: streamed narrative (first tab, so users see it typing live)
    with tabs[0]:
        if stream is not None:
            entry["narrative"] = st.write_stream(stream)
        else:
            st.markdown(entry["narrative"] or "_No narrative available._")

    # Tab 2: patient symptom summary + AI summary
    with tabs[1]:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Patient symptom summary")
            st.markdown(
                f"- **Age / gender:** {inputs['age']} / {inputs['gender']}\n"
                f"- **Symptoms:** {inputs['symptoms']}\n"
                f"- **Duration:** {inputs['duration']}\n"
                f"- **Severity:** {inputs['severity']}/10\n"
                f"- **Conditions:** {inputs['conditions']}\n"
                f"- **Medications:** {inputs['medications']}\n"
                f"- **Notes:** {inputs['notes']}"
            )
        with right:
            st.markdown("#### AI summary")
            st.info(result["summary"])

    # Tab 3: possible conditions (education only)
    with tabs[2]:
        st.warning("These are NOT diagnoses. They are general, educational possibilities only.")
        if result["possible_conditions"]:
            for condition in result["possible_conditions"]:
                with st.expander(condition["name"]):
                    st.write(condition["reason"] or "No explanation provided.")
        else:
            st.write("No possible conditions were returned.")

    # Tab 4: recommended next steps
    with tabs[3]:
        for step in result["recommended_next_steps"] or ["Consult a healthcare professional."]:
            st.success(step)

    # Tab 5: questions for the doctor
    with tabs[4]:
        st.markdown("Bring these questions to your healthcare professional:")
        for number, question in enumerate(result["questions_for_doctor"], start=1):
            st.markdown(f"{number}. {question}")

    # Tab 6: warning signs
    with tabs[5]:
        st.markdown("**Seek urgent or emergency help if you notice:**")
        for sign in result["warning_signs"] or ["Symptoms that suddenly get much worse."]:
            st.error(sign)

    # Tab 7: follow-up chat
    with tabs[6]:
        render_followup(entry, settings)

    # Tab 8: LangChain lab (for the examiners)
    with tabs[7]:
        render_lab(entry, settings)

    render_downloads(entry)

    # Disclaimer #4 (bottom of results)
    st.warning(config.DISCLAIMER_SHORT)


def render_lab(entry: dict, settings: dict):
    """Show the prompts, the raw-message demo and the cache comparison."""
    inputs = entry["inputs"]

    st.markdown("#### 1. PromptTemplate (single string with variables)")
    st.code(prompts.PATIENT_PROFILE_TEMPLATE.format(
        **{k: inputs[k] for k in prompts.PATIENT_PROFILE_TEMPLATE.input_variables}
    ))

    st.markdown("#### 2. ChatPromptTemplate (System + Human messages)")
    with st.expander("See the messages sent to the model"):
        for message in prompts.ASSESSMENT_CHAT_TEMPLATE.format_messages(**inputs):
            st.markdown(f"**{message.type.upper()} message**")
            st.code(message.content, language="text")

    st.markdown("#### 3. System / Human / AI messages (multi-turn demo)")
    st.caption("Sends: SystemMessage + HumanMessage + AIMessage (earlier answer) + a follow-up HumanMessage.")
    if st.button("Run message demo"):
        try:
            with st.spinner("Asking a follow-up question..."):
                llm = chains.build_llm(entry["model"], entry["temperature"], settings["api_key"])
                reply = chains.demo_message_conversation(llm, inputs, entry["result"]["summary"])
            st.info(reply)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Demo failed: {exc}")

    st.markdown("#### 4. InMemoryCache vs SQLiteCache")
    st.table(cache_manager.describe_cache_modes())
    st.caption(
        f"Cache used for this answer: **{config.CACHE_LABELS[entry['cache_mode']]}**. "
        "Submit the same form twice to see the second run become much faster."
    )


# ---------------------------------------------------------------------------
# Analytics tab
# ---------------------------------------------------------------------------
def history_to_csv(history: list) -> str:
    """Convert the session history into CSV text."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["time", "age", "symptoms", "severity", "urgency", "seconds", "cache"])
    writer.writeheader()
    for item in history:
        writer.writerow({
            "time": item["time"], "age": item["age"], "symptoms": item["symptoms"],
            "severity": item["severity"], "urgency": item["urgency"],
            "seconds": f"{item['seconds']:.2f}", "cache": item["cache"],
        })
    return buffer.getvalue()


def render_analytics():
    """Charts built from this session's assessments (bonus feature)."""
    history = st.session_state["history"]
    st.subheader("📈 Session analytics")
    if not history:
        st.info("Run at least one assessment to see analytics here.")
        return

    cached = sum(1 for h in history if h["cache"] != config.CACHE_NONE and h["seconds"] < 1.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assessments", len(history))
    c2.metric("Average severity", f"{sum(h['severity'] for h in history) / len(history):.1f}/10")
    c3.metric("Average response time", f"{sum(h['seconds'] for h in history) / len(history):.2f} s")
    c4.metric("Served from cache", cached)

    counts = Counter(h["urgency"] for h in history)
    symptom_counts = Counter(
        s.strip() for h in history for s in h["symptoms"].split(",") if s.strip()
    ).most_common(8)

    if go is None:
        st.bar_chart({level: counts.get(level, 0) for level in config.URGENCY_LEVELS})
    else:
        left, right = st.columns(2)
        with left:
            donut = go.Figure(go.Pie(
                labels=config.URGENCY_LEVELS,
                values=[counts.get(level, 0) for level in config.URGENCY_LEVELS],
                hole=0.55, sort=False,
                marker=dict(colors=[config.URGENCY_COLORS[level] for level in config.URGENCY_LEVELS]),
            ))
            donut.update_layout(title="Urgency distribution", height=330, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(donut, key="analytics_donut")
        with right:
            bars = go.Figure(go.Bar(
                x=[count for _, count in reversed(symptom_counts)],
                y=[name for name, _ in reversed(symptom_counts)],
                orientation="h", marker_color="#0f766e",
            ))
            bars.update_layout(title="Most frequent symptoms", height=330, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(bars, key="analytics_symptoms")

        trend = go.Figure(go.Scatter(
            x=list(range(1, len(history) + 1)), y=[h["severity"] for h in history],
            mode="lines+markers", line=dict(color="#1d4ed8"),
        ))
        trend.update_layout(
            title="Severity per assessment", height=280, margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Assessment #", yaxis=dict(title="Severity", range=[0, 10]),
        )
        st.plotly_chart(trend, key="analytics_trend")

    st.dataframe(
        [
            {"Time": h["time"], "Age": h["age"], "Symptoms": h["symptoms"], "Severity": h["severity"],
             "Urgency": h["urgency"], "Seconds": round(h["seconds"], 2), "Cache": h["cache"]}
            for h in reversed(history)
        ]
    )
    col_dl, col_clear = st.columns(2)
    with col_dl:
        st.download_button("⬇️ Export history (CSV)", data=history_to_csv(history), file_name="mediguide_history.csv", mime="text/csv")
    with col_clear:
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
ARCHITECTURE_DOT = r"""
digraph G {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="#ccfbf1", fontname="Helvetica", fontsize=11];
    edge [color="#475569"];
    form   [label="Streamlit form\n(app.py)"];
    check  [label="Validation\n(utils.py)"];
    prompt [label="ChatPromptTemplate\n(prompts.py)"];
    cache  [label="LLM cache\nInMemory / SQLite", fillcolor="#fef9c3"];
    chain  [label="LLMChain + ChatOpenAI\n(chains.py)", fillcolor="#dbeafe"];
    parse  [label="Safe JSON parse\n+ Pydantic (schemas.py)"];
    safety [label="Rule-based\nsafety net", fillcolor="#fee2e2"];
    stream [label="llm.stream()\nst.write_stream", fillcolor="#dbeafe"];
    ui     [label="Dashboard, gauges,\nPDF, follow-up chat", fillcolor="#e9d5ff"];
    form -> check -> prompt -> cache -> chain -> parse -> safety -> stream -> ui;
}
"""


def render_about():
    """Explain the project - handy for the presentation."""
    st.subheader("ℹ️ About MediGuide AI")
    st.markdown(
        "MediGuide AI is an **educational** LangChain application that converts symptoms into structured, "
        "safety-focused guidance. It never gives a confirmed diagnosis and always points users to "
        "qualified healthcare professionals."
    )

    st.markdown("#### Architecture")
    try:
        st.graphviz_chart(ARCHITECTURE_DOT)
    except Exception:  # noqa: BLE001
        st.code("form → validation → prompt → cache → LLMChain → JSON parse → safety net → stream → dashboard")

    st.markdown("#### LangChain concepts used")
    st.table([
        {"Concept": "ChatOpenAI", "Where": "src/chains.py → build_llm()"},
        {"Concept": "PromptTemplate", "Where": "src/prompts.py → PATIENT_PROFILE_TEMPLATE"},
        {"Concept": "ChatPromptTemplate", "Where": "src/prompts.py → ASSESSMENT / NARRATIVE templates"},
        {"Concept": "System / Human / AI messages", "Where": "src/chains.py → follow-up chat + message demo"},
        {"Concept": "LLMChain", "Where": "src/chains.py → build_assessment_chain()"},
        {"Concept": "Structured JSON", "Where": "src/prompts.py schema + src/schemas.py (Pydantic)"},
        {"Concept": "Streaming", "Where": "src/chains.py → stream_narrative() + st.write_stream"},
        {"Concept": "Caching", "Where": "src/cache_manager.py → InMemoryCache + SQLiteCache"},
    ])

    st.markdown("#### Tech stack")
    st.table(config.TECH_STACK)

    st.markdown("#### Safety design")
    st.markdown(
        "- Disclaimers on every screen; no confirmed diagnosis, no medicine doses.\n"
        "- Urgency: LOW / MEDIUM / HIGH / EMERGENCY with colour coding.\n"
        "- A **rule-based safety net** can raise (never lower) the urgency for danger signs.\n"
        "- Invalid JSON never crashes the app; Pydantic repairs missing fields.\n"
        "- Logs contain timings only - never patient data."
    )
    st.caption("💡 Tip: switch between light and dark mode from the ⋮ menu → Settings → Theme.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    settings = render_sidebar()
    render_hero()

    # Disclaimer #2 (main area, visible above every tab)
    st.error(config.DISCLAIMER_FULL)

    tab_assess, tab_analytics, tab_about = st.tabs(
        ["🩺 Assessment", "📈 Analytics", "ℹ️ About & architecture"]
    )

    with tab_assess:
        form, clicked = render_form(settings)

        stream = None
        if clicked:
            st.session_state["current"] = None
            entry, stream = run_analysis(settings, form)
            st.session_state["current"] = entry

        entry = st.session_state["current"]
        if entry:
            render_results(entry, settings, stream)
            if stream is not None:  # save to history only right after a fresh run
                st.session_state["history"].append({
                    "time": entry["timestamp"],
                    "age": entry["inputs"]["age"],
                    "symptoms": entry["inputs"]["symptoms"],
                    "severity": entry["inputs"]["severity"],
                    "urgency": entry["result"]["urgency_level"],
                    "seconds": entry["elapsed"],
                    "cache": entry["cache_mode"],
                })

    with tab_analytics:
        render_analytics()

    with tab_about:
        render_about()


main()
