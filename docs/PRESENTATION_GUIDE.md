# MediGuide AI – Presentation Guide

A ready-to-use plan for your final-year presentation / viva (about 10–12 minutes).

## 1. Suggested slide outline

| # | Slide | Key message |
|---|---|---|
| 1 | Title | MediGuide AI – safe, educational symptom guidance with LangChain |
| 2 | Problem | People search random websites → unreliable info, anxiety, confusion |
| 3 | Solution | Structured guidance: urgency level, next steps, doctor questions, warning signs |
| 4 | Safety first | No diagnosis, disclaimers everywhere, rule-based safety net |
| 5 | Architecture | Show the diagram from the app's *About* tab |
| 6 | LangChain concepts | ChatOpenAI, PromptTemplate, ChatPromptTemplate, messages, LLMChain, streaming, caching |
| 7 | Engineering quality | Pydantic validation, tests, CI, Docker, logging without patient data |
| 8 | Live demo | See script below |
| 9 | Results | Test scenarios table, cache speed-up |
| 10 | Limitations & future work | RAG over vetted sources, clinician evaluation, accounts |

## 2. Live demo script (≈ 5 minutes)

1. **Open the app** → point at the red disclaimers (sidebar, top of page).
2. Sidebar → **Demo scenarios → "1 · Mild cold"** → *Analyse symptoms*.
   Show: green LOW banner, **streaming** text typing live, gauges, tabs.
3. Click **"3 · Chest pain (EMERGENCY)"** → Analyse.
   Show: red EMERGENCY banner and the "🛡️ Safety check" message (rule-based safety net).
4. **Caching demo**: with *InMemoryCache* selected, run scenario 2 twice.
   First run: a few seconds. Second run: under a second, ⚡ "served from cache".
   Explain InMemory (RAM, lost on restart) vs SQLite (file, survives restart).
5. **Empty symptoms** → press Analyse → warning appears, **no API call is made**.
6. **Urdu**: choose scenario 5 → guidance returns in Urdu.
7. Open **Follow-up chat** and ask "Can I take paracetamol?" → answer stays safe (no dose, refers to doctor).
8. Show **PDF download** and the **Analytics** tab (urgency distribution, symptoms, average time).
9. Open **LangChain lab** tab → show the actual System/Human messages sent to the model.

> Tip: run each scenario once before the presentation so the cache is warm and the network cannot surprise you.

## 3. Talking points per concept

- **ChatOpenAI** – the model wrapper; temperature 0.2 for consistent, cautious answers; JSON mode for the assessment call.
- **PromptTemplate** – one string with variables (patient profile).
- **ChatPromptTemplate** – System message (role + safety rules + JSON schema) and Human message (patient data).
- **System / Human / AI messages** – the follow-up chat sends System + Human (patient) + AI (first answer) + new question, giving the model memory.
- **LLMChain** – reusable pipeline: prompt → model → text.
- **Structured JSON** – schema in the prompt, fences stripped, `json.loads` in try/except, then **Pydantic** repairs missing/wrong fields.
- **Streaming** – `llm.stream()` yields chunks; `st.write_stream` shows the typing effect.
- **Caching** – `set_llm_cache()` once; LangChain checks it before each call; we also reuse it for streamed text.

## 4. Likely viva questions (with short answers)

**Q: Why is this safe to show to users?**
It is labelled educational, never states a diagnosis, never gives doses, always escalates red flags and shows disclaimers on every screen. A rule-based layer can only *raise* urgency, so the LLM cannot silently under-triage obvious emergencies.

**Q: What if the model returns invalid JSON?**
JSON mode makes it rare; if it still happens, `safe_parse_json` catches the error, the app shows a friendly message and the raw output. If fields are missing or wrong, Pydantic fills safe defaults (unknown urgency → MEDIUM).

**Q: Difference between InMemoryCache and SQLiteCache?**
InMemory lives in RAM – fastest but lost on restart. SQLite is a file on disk – slightly slower but survives restarts and can be reused across sessions.

**Q: Why does streaming need special cache handling?**
LangChain's global cache is checked on `invoke`, not on `.stream()`. We read/write the same registered cache manually so repeated requests are instant too.

**Q: How do you handle prompt injection?**
The system prompt states that patient text is data and must not change the rules; output is validated by schema; the safety net runs outside the model.

**Q: Why LangChain instead of calling OpenAI directly?**
Reusable prompt templates, message abstractions, chains, streaming and caching behind one interface – and the model provider can be swapped with minimal code change.

**Q: How would you evaluate accuracy?**
Build a test set of cases reviewed by clinicians, measure urgency agreement (especially under-triage), and track failures. Currently there are scenario tests and unit tests for the safety logic.

**Q: Privacy?**
No database of patients, logs contain only timings, the API key is read from `.env`, users are told not to enter identifying details.

**Q: What are the limitations?**
It is not a medical device; LLMs can be wrong; PDF export supports Latin scripts only; no RAG over vetted medical sources yet.

## 5. Before you present – checklist

- [ ] `.env` contains a valid key and the account has credit
- [ ] Internet works; run every demo scenario once beforehand
- [ ] Browser zoomed to ~110 %, light or dark theme chosen
- [ ] Repo pushed to GitHub (`.env` NOT committed) and screen recording prepared as backup
