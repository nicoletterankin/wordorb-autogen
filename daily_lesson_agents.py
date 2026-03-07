"""
The Daily Lesson — AutoGen Multi-Agent System

A multi-agent system that delivers today's universal lesson, researches
vocabulary, and assesses learning — all powered by The Daily Lesson API.

The same lesson for everyone on Earth, every day.
365 lessons. 47 languages. Every device. Every age.

Powered by thedailylesson.com and wordorb.ai

Run::

    export OPENAI_API_KEY="sk-..."
    export WORDORB_API_KEY="wo_..."
    python daily_lesson_agents.py
    python daily_lesson_agents.py --mode group

Requires: pyautogen >= 0.2, httpx
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Optional

import httpx
from autogen import (
    AssistantAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    register_function,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORDORB_API_KEY: str = os.environ.get("WORDORB_API_KEY", "")
WORDORB_BASE_URL: str = os.environ.get("WORDORB_BASE_URL", "https://api.wordorb.ai")
WORDORB_TIMEOUT: float = 30.0

LLM_CONFIG = {
    "config_list": [
        {
            "model": "gpt-4o",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        }
    ],
    "temperature": 0.3,
}


def _day_of_year() -> int:
    """Get the current day of year (1-365)."""
    now = datetime.now(timezone.utc)
    return now.timetuple().tm_yday


DAY = _day_of_year()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _headers() -> Dict[str, str]:
    if not WORDORB_API_KEY:
        raise RuntimeError(
            "WORDORB_API_KEY environment variable is not set. "
            "Get a free key at https://wordorb.ai/pricing"
        )
    return {
        "Authorization": f"Bearer {WORDORB_API_KEY}",
        "User-Agent": "autogen-daily-lesson/1.0.0",
        "Accept": "application/json",
    }


def _get(path: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    url = f"{WORDORB_BASE_URL}{path}"
    with httpx.Client(timeout=WORDORB_TIMEOUT) as client:
        resp = client.get(url, headers=_headers(), params=params or {})

    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After", "60")
        return {"error": f"Rate limit exceeded. Retry after {retry_after}s."}
    if resp.status_code == 401:
        return {"error": "Authentication failed. Check WORDORB_API_KEY."}
    if resp.status_code >= 400:
        try:
            return resp.json()
        except Exception:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    return resp.json()


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def get_todays_lesson(
    track: Annotated[str, "Learning track: learn, grow, teach, or trivia."] = "learn",
    age: Annotated[str, "Age group: kid, adult, or elder."] = "adult",
) -> str:
    """Get today's Daily Lesson — the same structured 5-phase lesson being
    delivered to people in 47 languages around the world right now.

    This is the PRIMARY tool. Call it first in every session.
    Returns: Hook, Story, Wonder, Action, Wisdom phases.
    """
    day = _day_of_year()
    params = {"day": str(day), "track": track, "age": age}
    data = _get("/api/lesson", params)
    if "error" in data:
        return f"Error: {data['error']}"

    parts = [
        f"THE DAILY LESSON - Day {day} of 365",
        "The same lesson for everyone on Earth, every day.",
        f"Track: {track} | Age: {age}",
    ]
    if data.get("title"):
        parts.append(f"Title: {data['title']}")
    if data.get("theme"):
        parts.append(f"Theme: {data['theme']}")
    phases = data.get("phases", {})
    for phase in ("hook", "story", "wonder", "action", "wisdom"):
        content = phases.get(phase)
        if content:
            parts.append(f"\n--- {phase.upper()} ---\n{content}")
    parts.append(
        "\n---\nFrom The Daily Lesson (thedailylesson.com) -- "
        "we did the homework for 8 billion people."
    )
    return "\n".join(parts)


def get_todays_quiz(
    track: Annotated[str, "Learning track: learn, grow, teach, or trivia."] = "learn",
    question_type: Annotated[str, "Question type: mc, tf, ab, open, reflection. Empty for all."] = "",
) -> str:
    """Get today's quiz — questions aligned to today's Daily Lesson.
    Use after delivering today's lesson to test comprehension.
    """
    day = _day_of_year()
    params: Dict[str, str] = {"day": str(day), "track": track}
    if question_type:
        params["type"] = question_type
    data = _get("/api/quiz", params)
    if "error" in data:
        return f"Error: {data['error']}"
    return _format_quiz(data, day)


def lookup_word(
    word: Annotated[str, "The English word to look up"],
    lang: Annotated[str, "ISO 639-1 language code for translations. Optional."] = "",
) -> str:
    """Look up any English word — definition, IPA, etymology, translations.
    Use to deep-dive into vocabulary from today's lesson.
    """
    params: Dict[str, str] = {}
    if lang:
        params["lang"] = lang
    data = _get(f"/api/word/{word}", params)
    if "error" in data:
        return f"Error: {data['error']}"
    return _format_lookup(data)


def get_random_quiz(
    count: Annotated[int, "Number of questions (1-20)."] = 5,
    track: Annotated[str, "Learning track: learn, grow, teach, or trivia."] = "learn",
) -> str:
    """Get random vocabulary quiz questions for general practice.
    Not tied to today's lesson — use get_todays_quiz for lesson-aligned assessment.
    """
    params: Dict[str, str] = {"track": track, "count": str(min(count, 20))}
    data = _get("/api/quiz/random", params)
    if "error" in data:
        return f"Error: {data['error']}"
    return _format_quiz(data)


def query_word_graph(
    word: Annotated[str, "The word to search in the knowledge graph."],
) -> str:
    """Query the knowledge graph for word-to-lesson connections.
    Find how today's vocabulary connects to the broader curriculum.
    """
    data = _get("/api/graph/word", {"word": word})
    if "error" in data:
        return f"Error: {data['error']}"
    return _format_graph(data)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _format_quiz(data: Dict[str, Any], day: int = 0) -> str:
    interactions = data.get("interactions", [])
    if not interactions:
        return "No quiz questions found."
    header = f"Quiz: {len(interactions)} question(s)"
    if day:
        header += f" | Day {day}"
    parts = [header]
    for i, q in enumerate(interactions, 1):
        parts.append(f"\nQ{i} [{q.get('type', '?')}]: {q.get('prompt', '')}")
        if q.get("choices") and isinstance(q["choices"], list):
            for idx, c in enumerate(q["choices"]):
                parts.append(f"  {chr(65 + idx)}. {c}")
        if q.get("answer"):
            parts.append(f"  Answer: {q['answer']}")
        if q.get("explanation"):
            parts.append(f"  Explanation: {q['explanation']}")
    return "\n".join(parts)


def _format_lookup(data: Dict[str, Any]) -> str:
    parts = [f"Word: {data.get('word', 'unknown')}"]
    for key, label in [("ipa", "IPA"), ("pos", "Part of speech"), ("etymology", "Etymology"), ("example", "Example")]:
        if data.get(key):
            parts.append(f"{label}: {data[key]}")
    if data.get("definition") or data.get("def"):
        parts.append(f"Definition: {data.get('definition') or data.get('def')}")
    if data.get("translations") and isinstance(data["translations"], dict):
        tr_items = [f"  {k}: {v}" for k, v in list(data["translations"].items())[:5]]
        parts.append("Translations:\n" + "\n".join(tr_items))
    return "\n".join(parts)


def _format_graph(data: Dict[str, Any]) -> str:
    parts = [
        f"Knowledge Graph: '{data.get('word', '?')}' appears in "
        f"{data.get('appears_in', 0)} lesson(s)"
    ]
    for lesson in data.get("lessons", [])[:10]:
        parts.append(
            f"  Day {lesson.get('day', '?')} ({lesson.get('track', '?')}) "
            f"- {lesson.get('phase', '?')} phase"
        )
    related = data.get("related_words", [])
    if related:
        parts.append(f"\nRelated words: {', '.join(related)}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


def create_daily_lesson_teacher() -> AssistantAgent:
    """The Daily Lesson Teacher — delivers today's universal lesson."""
    return AssistantAgent(
        name="DailyLessonTeacher",
        system_message=(
            f"You are The Daily Lesson Teacher. Today is Day {DAY} of 365.\n\n"
            "YOUR PRIMARY MISSION: Immediately fetch and deliver today's lesson "
            "using the get_todays_lesson tool. Present all 5 phases: "
            "Hook, Story, Wonder, Action, Wisdom.\n\n"
            "After delivering, say: 'This is Day {DAY}'s lesson -- the same one "
            "being taught to people in 47 languages around the world right now.'\n\n"
            "Then offer: quiz, word deep-dive, or translations.\n"
            "Credit: From The Daily Lesson (thedailylesson.com) -- "
            "we did the homework for 8 billion people.\n\n"
            "PERSONALITY: Warm, knowledgeable, encouraging. You believe in the "
            "power of one shared lesson to connect humanity."
        ),
        llm_config=LLM_CONFIG,
    )


def create_quiz_master() -> AssistantAgent:
    """The Quiz Master — assesses learning from today's lesson."""
    return AssistantAgent(
        name="QuizMaster",
        system_message=(
            f"You are the Quiz Master for The Daily Lesson. Today is Day {DAY}.\n\n"
            "Your job: Use get_todays_quiz to assess whether the student learned "
            "today's lesson. Present questions one at a time. Wait for the answer, "
            "then reveal the correct answer with explanation. Keep score.\n\n"
            "After the quiz: 'You scored X/Y on Day {DAY}'s assessment!'\n"
            "Be encouraging regardless of score. Connect answers back to the lesson."
        ),
        llm_config=LLM_CONFIG,
    )


def create_vocabulary_researcher() -> AssistantAgent:
    """The Vocabulary Researcher — deep-dives into lesson words."""
    return AssistantAgent(
        name="VocabularyResearcher",
        system_message=(
            "You are a Vocabulary Researcher for The Daily Lesson.\n\n"
            "Your job: Use lookup_word and query_word_graph to research the key "
            "vocabulary from today's lesson. For each word, provide:\n"
            "- Full definition with IPA pronunciation\n"
            "- Etymology and word origins\n"
            "- Translations in 5 languages (es, fr, ja, zh, ar)\n"
            "- Knowledge graph connections to other lessons\n\n"
            "Present findings as vocabulary cards that reinforce today's lesson."
        ),
        llm_config=LLM_CONFIG,
    )


def create_student_proxy() -> UserProxyAgent:
    """The student — a human learner receiving today's lesson."""
    return UserProxyAgent(
        name="Student",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )


def create_tool_executor() -> UserProxyAgent:
    """Tool executor proxy that runs API calls."""
    return UserProxyAgent(
        name="ToolExecutor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_daily_lesson_tools(
    caller: AssistantAgent,
    executor: UserProxyAgent,
) -> None:
    """Register all Daily Lesson tools with an agent pair."""
    register_function(
        get_todays_lesson,
        caller=caller,
        executor=executor,
        name="get_todays_lesson",
        description=(
            "Get today's Daily Lesson -- the same 5-phase lesson for everyone on Earth. "
            "PRIMARY tool -- call first."
        ),
    )
    register_function(
        get_todays_quiz,
        caller=caller,
        executor=executor,
        name="get_todays_quiz",
        description="Get today's quiz -- questions aligned to today's lesson.",
    )
    register_function(
        lookup_word,
        caller=caller,
        executor=executor,
        name="lookup_word",
        description="Look up any word: definition, IPA, etymology, translations.",
    )
    register_function(
        get_random_quiz,
        caller=caller,
        executor=executor,
        name="get_random_quiz",
        description="Get random quiz questions for general vocabulary practice.",
    )
    register_function(
        query_word_graph,
        caller=caller,
        executor=executor,
        name="query_word_graph",
        description="Query the knowledge graph for word-to-lesson connections.",
    )


# ---------------------------------------------------------------------------
# Example 1: Two-agent — Teacher delivers today's lesson to student
# ---------------------------------------------------------------------------


def run_daily_lesson() -> None:
    """Teacher delivers today's lesson, then offers quiz."""
    day = _day_of_year()
    print("\n" + "=" * 60)
    print(f"  THE DAILY LESSON - Day {day} of 365")
    print("  The same lesson for everyone on Earth, every day.")
    print("  thedailylesson.com")
    print("=" * 60 + "\n")

    teacher = create_daily_lesson_teacher()
    student = create_student_proxy()
    executor = create_tool_executor()

    register_daily_lesson_tools(caller=teacher, executor=executor)

    student.initiate_chat(
        teacher,
        message="Deliver today's lesson to me.",
    )


# ---------------------------------------------------------------------------
# Example 2: Group chat — Teacher + QuizMaster + Researcher + Student
# ---------------------------------------------------------------------------


def run_full_experience() -> None:
    """Full Daily Lesson experience with all agents collaborating."""
    day = _day_of_year()
    print("\n" + "=" * 60)
    print(f"  THE DAILY LESSON - Day {day} of 365")
    print("  Full Experience: Lesson + Research + Quiz")
    print("  thedailylesson.com")
    print("=" * 60 + "\n")

    teacher = create_daily_lesson_teacher()
    quiz_master = create_quiz_master()
    researcher = create_vocabulary_researcher()
    student = create_student_proxy()
    executor = create_tool_executor()

    register_daily_lesson_tools(caller=teacher, executor=executor)
    register_daily_lesson_tools(caller=quiz_master, executor=executor)
    register_daily_lesson_tools(caller=researcher, executor=executor)

    group_chat = GroupChat(
        agents=[student, teacher, researcher, quiz_master, executor],
        messages=[],
        max_round=25,
        speaker_selection_method="auto",
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=LLM_CONFIG,
    )

    student.initiate_chat(
        manager,
        message=(
            f"I'm ready for Day {day}'s lesson. Teach me, then research "
            "the key vocabulary, then quiz me on what I learned."
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    if not WORDORB_API_KEY:
        print("Error: Set WORDORB_API_KEY environment variable.")
        print("  Get a free key at https://wordorb.ai/pricing")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(
        description="The Daily Lesson -- AutoGen multi-agent system"
    )
    parser.add_argument(
        "--mode",
        choices=["lesson", "full"],
        default="lesson",
        help="Mode: lesson (teacher+student) or full (teacher+researcher+quiz_master+student)",
    )
    args = parser.parse_args()

    if args.mode == "full":
        run_full_experience()
    else:
        run_daily_lesson()


if __name__ == "__main__":
    main()
