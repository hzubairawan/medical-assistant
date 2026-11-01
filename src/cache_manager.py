"""
cache_manager.py
================
Switch LangChain's LLM cache between: off, InMemoryCache and SQLiteCache.

How it works (beginner explanation):
    set_llm_cache(cache) registers ONE global cache. From then on, LangChain
    checks it automatically before every model call:
        * cache hit  -> the saved answer is returned instantly (no API cost)
        * cache miss -> the model is called and the answer is saved

InMemoryCache vs SQLiteCache
    +------------------+----------------------+------------------------------+
    |                  | InMemoryCache        | SQLiteCache                  |
    +------------------+----------------------+------------------------------+
    | Stored in        | RAM (memory)         | a file on disk (.db)         |
    | Speed            | fastest              | fast, slightly slower        |
    | Survives restart | NO                   | YES                          |
    | Best for         | one session          | reusing answers across runs  |
    +------------------+----------------------+------------------------------+
"""

from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache

from src import config

# We keep ONE instance of each cache at module level. If we created a new
# InMemoryCache on every Streamlit rerun, it would always start empty!
_memory_cache = None
_sqlite_cache = None
_current_mode = config.CACHE_NONE


def _get_memory_cache() -> InMemoryCache:
    """Create the in-memory cache once and reuse it."""
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = InMemoryCache()
    return _memory_cache


def _get_sqlite_cache() -> SQLiteCache:
    """Create the SQLite cache once (and its folder) and reuse it."""
    global _sqlite_cache
    if _sqlite_cache is None:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _sqlite_cache = SQLiteCache(database_path=str(config.SQLITE_CACHE_PATH))
    return _sqlite_cache


def configure_cache(mode: str) -> str:
    """
    Register the chosen cache globally with set_llm_cache().

    mode: "none" | "memory" | "sqlite"
    Returns the mode that is now active.
    """
    global _current_mode

    if mode == config.CACHE_MEMORY:
        set_llm_cache(_get_memory_cache())
    elif mode == config.CACHE_SQLITE:
        set_llm_cache(_get_sqlite_cache())
    else:
        set_llm_cache(None)  # caching disabled
        mode = config.CACHE_NONE

    _current_mode = mode
    return mode


def get_current_mode() -> str:
    """Return the currently active cache mode."""
    return _current_mode


def clear_cache(mode: str) -> bool:
    """Empty the chosen cache. Returns True if something was cleared."""
    if mode == config.CACHE_MEMORY and _memory_cache is not None:
        _memory_cache.clear()
        return True
    if mode == config.CACHE_SQLITE and _sqlite_cache is not None:
        _sqlite_cache.clear()
        return True
    return False


def describe_cache_modes() -> list:
    """Rows for the comparison table shown in the UI."""
    return [
        {"": "Stored in", "InMemoryCache": "RAM (memory)", "SQLiteCache": "A file on disk (.db)"},
        {"": "Speed", "InMemoryCache": "Fastest", "SQLiteCache": "Fast, slightly slower"},
        {"": "Survives restart?", "InMemoryCache": "No", "SQLiteCache": "Yes"},
        {"": "Best for", "InMemoryCache": "One session", "SQLiteCache": "Reusing across sessions"},
    ]
