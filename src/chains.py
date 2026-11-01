"""
chains.py
=========
Everything that talks to the LLM:

    * build_llm()                  -> ChatOpenAI
    * build_assessment_chain()     -> reusable LLMChain (JSON assessment)
    * run_assessment()             -> run the chain, return the raw JSON text
    * stream_narrative()           -> generator using llm.stream()
    * demo_message_conversation()  -> raw SystemMessage / HumanMessage / AIMessage demo
"""

import logging
import re
import time
import warnings
from typing import Iterator

from langchain_core.globals import get_llm_cache
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import Generation
from langchain_openai import ChatOpenAI

from src import prompts

# LLMChain lives in `langchain` (0.3.x) or in `langchain-classic` (1.x and newer).
try:
    from langchain.chains import LLMChain
except ImportError:  # newer LangChain
    from langchain_classic.chains import LLMChain

# LLMChain still works but newer LangChain versions mark it as "legacy".
warnings.filterwarnings("ignore", message=".*LLMChain.*")


# ---------------------------------------------------------------------------
# 1) ChatOpenAI
# ---------------------------------------------------------------------------
def build_llm(
    model: str,
    temperature: float,
    api_key: str,
    json_mode: bool = False,
) -> ChatOpenAI:
    """
    Create the OpenAI chat model.

    json_mode=True asks OpenAI to guarantee syntactically valid JSON output
    (the prompt must mention "JSON", which ours does).
    """
    extra = {}
    if json_mode:
        extra["model_kwargs"] = {"response_format": {"type": "json_object"}}

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        timeout=60,      # seconds before giving up
        max_retries=2,   # retry temporary network / rate-limit errors
        **extra,
    )


# ---------------------------------------------------------------------------
# 2) LLMChain (prompt + model = reusable pipeline)
# ---------------------------------------------------------------------------
def build_assessment_chain(llm) -> LLMChain:
    """Reusable chain: patient variables -> ChatPromptTemplate -> LLM -> text."""
    return LLMChain(llm=llm, prompt=prompts.ASSESSMENT_CHAT_TEMPLATE)


def run_assessment(chain: LLMChain, inputs: dict) -> str:
    """Run the chain and return the model's raw text (should be JSON)."""
    result = chain.invoke(inputs)  # LLMChain returns a dict with key "text"
    return result["text"]


# ---------------------------------------------------------------------------
# 3) Streaming
# ---------------------------------------------------------------------------
def _messages_to_key(messages) -> str:
    """Turn a list of messages into one string usable as a cache key."""
    return "\n".join(f"{m.type}: {m.content}" for m in messages)


def _replay_text(text: str) -> Iterator[str]:
    """Re-play a cached answer in small pieces so the typing effect is kept."""
    pieces = re.findall(r"\S+\s*", text)
    for i in range(0, len(pieces), 3):
        yield "".join(pieces[i:i + 3])
        time.sleep(0.01)


def stream_narrative(llm, inputs: dict) -> Iterator[str]:
    """
    Stream the human-readable guidance with llm.stream().

    st.write_stream() prints every yielded piece, giving a typing effect.

    Caching note: LangChain's global cache is not consulted by .stream(), so we
    use the SAME registered cache (get_llm_cache) by hand. That way the second
    identical request is replayed instantly from InMemoryCache / SQLiteCache.
    """
    messages = prompts.NARRATIVE_CHAT_TEMPLATE.format_messages(**inputs)

    cache = get_llm_cache()
    cache_key = _messages_to_key(messages)
    llm_string = (
        f"narrative|{getattr(llm, 'model_name', 'model')}|"
        f"{getattr(llm, 'temperature', '')}"
    )

    # Cache hit -> replay the saved text.
    if cache is not None:
        cached = cache.lookup(cache_key, llm_string)
        if cached:
            yield from _replay_text(cached[0].text)
            return

    # Cache miss -> stream from the model and remember the final text.
    parts = []
    for chunk in llm.stream(messages):
        if chunk.content:
            parts.append(chunk.content)
            yield chunk.content

    if cache is not None and parts:
        cache.update(cache_key, llm_string, [Generation(text="".join(parts))])


# ---------------------------------------------------------------------------
# 4) Raw System / Human / AI message demo
# ---------------------------------------------------------------------------
def demo_message_conversation(llm, inputs: dict, previous_reply: str) -> str:
    """
    Show how a multi-turn chat is built from message objects:

        SystemMessage -> sets the role and safety rules
        HumanMessage  -> the patient's information (first turn)
        AIMessage     -> the assistant's EARLIER answer (conversation memory)
        HumanMessage  -> a follow-up question

    The model sees the whole list and answers the follow-up in context.
    """
    patient_text = prompts.PATIENT_PROFILE_TEMPLATE.format(
        **{k: inputs[k] for k in prompts.PATIENT_PROFILE_TEMPLATE.input_variables}
    )
    messages = [
        SystemMessage(content=prompts.DEMO_SYSTEM_MESSAGE),
        HumanMessage(content=patient_text),
        AIMessage(content=previous_reply),
        HumanMessage(content=prompts.DEMO_FOLLOW_UP_QUESTION),
    ]
    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# 5) Follow-up chat (real multi-turn memory with Human / AI messages)
# ---------------------------------------------------------------------------
MAX_HISTORY_TURNS = 10  # keep the prompt small and cheap


def build_followup_messages(
    inputs: dict,
    urgency: str,
    first_answer: str,
    chat_history: list,
    question: str,
) -> list:
    """
    Build the full message list for a follow-up question:

        SystemMessage  -> role + safety rules (+ the urgency already decided)
        HumanMessage   -> the patient's original information
        AIMessage      -> the guidance MediGuide already gave
        Human/AI ...   -> earlier follow-up questions and answers
        HumanMessage   -> the NEW question
    """
    patient_text = prompts.PATIENT_PROFILE_TEMPLATE.format(
        **{k: inputs[k] for k in prompts.PATIENT_PROFILE_TEMPLATE.input_variables}
    )
    messages = [
        SystemMessage(
            content=prompts.FOLLOWUP_SYSTEM_PROMPT.format(
                urgency=urgency, language=inputs["language"]
            )
        ),
        HumanMessage(content=patient_text + "\n\nPlease give me guidance."),
        AIMessage(content=first_answer),
    ]
    for turn in chat_history[-MAX_HISTORY_TURNS * 2:]:
        message_class = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(message_class(content=turn["content"]))
    messages.append(HumanMessage(content=question))
    return messages


def stream_followup(llm, messages: list) -> Iterator[str]:
    """Stream the answer to a follow-up question piece by piece."""
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


# ---------------------------------------------------------------------------
# 6) Voice input (OpenAI Whisper speech-to-text)
# ---------------------------------------------------------------------------
def transcribe_audio(audio_bytes: bytes, api_key: str) -> str:
    """Convert a recorded WAV clip into text with OpenAI's Whisper model."""
    from openai import OpenAI  # imported here so the app still starts without it

    client = OpenAI(api_key=api_key)
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=("voice.wav", audio_bytes, "audio/wav"),
    )
    logging.getLogger("mediguide").info("Voice clip transcribed (%d bytes).", len(audio_bytes))
    return (result.text or "").strip()
