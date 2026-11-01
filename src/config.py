"""
config.py
=========
Central place for settings, form options and safety text.

Beginner note:
    Keeping constants here means the rest of the code never contains
    "magic" strings, and the API key is read from the environment
    (.env file) instead of being hard-coded.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths + environment
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Reads the .env file (if it exists) and puts its values into os.environ.
load_dotenv(BASE_DIR / ".env")

# SQLiteCache stores its database here (folder is created automatically).
CACHE_DIR = BASE_DIR / ".cache"
SQLITE_CACHE_PATH = CACHE_DIR / "mediguide_cache.db"


# Logging: we log timings and errors, NEVER patient data (privacy).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("mediguide")


def get_api_key() -> str:
    """Return the OpenAI API key from the environment ('' if missing)."""
    return os.getenv("OPENAI_API_KEY", "").strip()


def is_valid_api_key(key: str) -> bool:
    """Very small sanity check so the placeholder from .env.example is rejected."""
    return bool(key) and key.startswith("sk") and "your-key" not in key.lower()


# ---------------------------------------------------------------------------
# App identity
# ---------------------------------------------------------------------------
APP_NAME = "MediGuide AI"
APP_ICON = "🩺"
APP_TAGLINE = "AI-powered symptom assessment & patient guidance"
APP_DESCRIPTION = (
    "MediGuide AI turns the symptoms you enter into structured, easy-to-read "
    "**educational** guidance: an urgency level, possible next steps, questions "
    "for your doctor and warning signs to watch for."
)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
if DEFAULT_MODEL not in MODEL_OPTIONS:
    MODEL_OPTIONS.insert(0, DEFAULT_MODEL)

DEFAULT_TEMPERATURE = 0.2  # low = more consistent, safer answers

# ---------------------------------------------------------------------------
# Cache modes (used by cache_manager.py and the sidebar)
# ---------------------------------------------------------------------------
CACHE_NONE = "none"
CACHE_MEMORY = "memory"
CACHE_SQLITE = "sqlite"
CACHE_LABELS = {
    CACHE_NONE: "Off (always call the API)",
    CACHE_MEMORY: "InMemoryCache (RAM, this session)",
    CACHE_SQLITE: "SQLiteCache (file on disk)",
}

# ---------------------------------------------------------------------------
# Form options
# ---------------------------------------------------------------------------
GENDER_OPTIONS = ["Female", "Male", "Other", "Prefer not to say"]

SYMPTOM_OPTIONS = [
    "Fever",
    "Cough",
    "Sore throat",
    "Runny nose",
    "Nasal congestion",
    "Sneezing",
    "Headache",
    "Dizziness",
    "Fatigue",
    "Body aches",
    "Chills",
    "Nausea",
    "Vomiting",
    "Diarrhea",
    "Constipation",
    "Abdominal pain",
    "Heartburn",
    "Loss of appetite",
    "Chest pain",
    "Shortness of breath",
    "Palpitations",
    "Back pain",
    "Joint pain",
    "Skin rash",
    "Itching",
    "Painful urination",
    "Ear pain",
    "Loss of taste or smell",
    "Swelling in legs",
    "Blurred vision",
    "Confusion",
    "Fainting / loss of consciousness",
]

DURATION_OPTIONS = [
    "Less than 24 hours",
    "1-3 days",
    "4-7 days",
    "1-2 weeks",
    "2-4 weeks",
    "More than 1 month",
]

LANGUAGE_OPTIONS = ["English", "Urdu", "Hindi", "Arabic", "Spanish", "French"]

DEFAULT_SEVERITY = 3

# ---------------------------------------------------------------------------
# Urgency levels
# ---------------------------------------------------------------------------
URGENCY_LEVELS = ["LOW", "MEDIUM", "HIGH", "EMERGENCY"]  # lowest -> highest

URGENCY_COLORS = {
    "LOW": "#2e7d32",        # green
    "MEDIUM": "#f9a825",     # amber
    "HIGH": "#ef6c00",       # orange
    "EMERGENCY": "#c62828",  # red
}
URGENCY_TEXT_COLORS = {
    "LOW": "#ffffff",
    "MEDIUM": "#1b1b1b",
    "HIGH": "#ffffff",
    "EMERGENCY": "#ffffff",
}
URGENCY_SOFT_COLORS = {  # translucent versions used as gauge backgrounds
    "LOW": "rgba(46,125,50,0.35)",
    "MEDIUM": "rgba(249,168,37,0.35)",
    "HIGH": "rgba(239,108,0,0.35)",
    "EMERGENCY": "rgba(198,40,40,0.35)",
}
URGENCY_ICONS = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "EMERGENCY": "🔴"}
URGENCY_MEANINGS = {
    "LOW": "Monitor at home and watch for changes",
    "MEDIUM": "Consider seeing a healthcare professional soon",
    "HIGH": "Get assessed by a healthcare professional today",
    "EMERGENCY": "Seek emergency medical help immediately",
}

# ---------------------------------------------------------------------------
# Rule-based safety net (runs in addition to the LLM, see utils.py)
# ---------------------------------------------------------------------------
# Selected symptoms that should never be rated below HIGH.
RED_FLAG_SYMPTOMS = {
    "chest pain",
    "shortness of breath",
    "fainting / loss of consciousness",
    "confusion",
}

# Words in free text that always mean EMERGENCY.
EMERGENCY_KEYWORDS = [
    "suicid",
    "kill myself",
    "want to die",
    "self-harm",
    "overdose",
    "not breathing",
    "can't breathe",
    "cannot breathe",
    "unconscious",
    "severe bleeding",
    "coughing blood",
    "face drooping",
    "slurred speech",
    "seizure",
]

EMERGENCY_STEP = (
    "Call your local emergency number or go to the nearest emergency "
    "department IMMEDIATELY. Do not wait and do not drive yourself if you feel unwell."
)

# ---------------------------------------------------------------------------
# Medical disclaimers (shown in the sidebar, main area and results)
# ---------------------------------------------------------------------------
DISCLAIMER_SHORT = (
    "⚠️ **Educational AI prototype only.** MediGuide AI is NOT a doctor, "
    "NOT a diagnosis and NOT an emergency service. Always consult a qualified "
    "healthcare professional. In an emergency, call your local emergency number immediately."
)

DISCLAIMER_FULL = (
    "🩺 **IMPORTANT MEDICAL NOTICE** — This application is an educational AI "
    "prototype. It does **not** provide a confirmed diagnosis, does **not** replace "
    "a licensed doctor, professional diagnosis or medical treatment, and must "
    "**never** be used in an emergency. If you or someone else may be in danger, "
    "contact your local emergency services right away."
)

EMERGENCY_MESSAGE = (
    "🚨 **This may be a medical emergency.** Call your local emergency number or go "
    "to the nearest emergency department immediately. Do not wait for more "
    "information from this app."
)

# Plain-text disclaimer (used inside the PDF report, which cannot show emoji).
DISCLAIMER_PLAIN = (
    "IMPORTANT MEDICAL NOTICE: This report was produced by an educational AI "
    "prototype. It is NOT a diagnosis, does NOT replace a licensed doctor or "
    "professional medical treatment, and must NOT be used in an emergency. "
    "Always consult a qualified healthcare professional. In an emergency, "
    "call your local emergency number immediately."
)

# ---------------------------------------------------------------------------
# One-click demo scenarios (sidebar). They match the assignment's test table.
# The keys are the Streamlit widget keys used in app.py.
# ---------------------------------------------------------------------------
DEMO_SCENARIOS = {
    "1 · Mild cold (LOW)": {
        "age": "25", "gender": "Female",
        "symptoms": ["Runny nose", "Sore throat"], "extra_symptoms": "",
        "duration": "1-3 days", "severity": 2,
        "conditions": "", "medications": "", "notes": "", "language": "English",
    },
    "2 · Fever and cough (MEDIUM/HIGH)": {
        "age": "40", "gender": "Male",
        "symptoms": ["Fever", "Cough"], "extra_symptoms": "",
        "duration": "4-7 days", "severity": 6,
        "conditions": "", "medications": "", "notes": "", "language": "English",
    },
    "3 · Chest pain (EMERGENCY)": {
        "age": "55", "gender": "Male",
        "symptoms": ["Chest pain", "Shortness of breath"], "extra_symptoms": "",
        "duration": "Less than 24 hours", "severity": 9,
        "conditions": "High blood pressure", "medications": "", "notes": "Pain spreads to the left arm",
        "language": "English",
    },
    "4 · Toddler with fever (safety net)": {
        "age": "2", "gender": "Female",
        "symptoms": ["Fever", "Loss of appetite"], "extra_symptoms": "",
        "duration": "1-3 days", "severity": 5,
        "conditions": "", "medications": "", "notes": "", "language": "English",
    },
    "5 · Same as 2, Urdu answer": {
        "age": "40", "gender": "Male",
        "symptoms": ["Fever", "Cough"], "extra_symptoms": "",
        "duration": "4-7 days", "severity": 6,
        "conditions": "", "medications": "", "notes": "", "language": "Urdu",
    },
}

# Shown on the "About" tab.
TECH_STACK = [
    {"Layer": "Language", "Tool": "Python 3.10+"},
    {"Layer": "LLM framework", "Tool": "LangChain (prompts, LLMChain, messages, streaming, caching)"},
    {"Layer": "Model provider", "Tool": "OpenAI (gpt-4o-mini by default, Whisper for voice)"},
    {"Layer": "Validation", "Tool": "Pydantic v2 schema for the model's JSON"},
    {"Layer": "User interface", "Tool": "Streamlit + custom CSS"},
    {"Layer": "Charts", "Tool": "Plotly (gauges and analytics)"},
    {"Layer": "Reports", "Tool": "ReportLab (PDF) and Markdown export"},
    {"Layer": "Caching", "Tool": "InMemoryCache + SQLiteCache"},
    {"Layer": "Quality", "Tool": "pytest (offline tests) + GitHub Actions CI"},
    {"Layer": "Deployment", "Tool": "Docker + docker-compose"},
]
