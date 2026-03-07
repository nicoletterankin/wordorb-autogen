"""
AutoGen agent setup with WordOrb.ai tool functions.

Provides:
  - Tool function definitions for ``register_for_llm``
  - VocabularyTutor AssistantAgent with all WordOrb tools registered
  - GroupChat example: tutor + student + quiz_master
  - Tool execution functions

Run::

    export OPENAI_API_KEY="sk-..."
    export WORDORB_API_KEY="wo_..."
    python wordorb_agent.py

Requires: pyautogen >= 0.2, httpx
"""

from __future__ import annotations

import json
import os
import sys
from typing import Annotated, Any, Dict, List, Literal, Optional

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


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _headers() -> Dict[str, str]:
    """Build authorization headers for the WordOrb API."""
    if not WORDORB_API_KEY:
        raise RuntimeError(
            "WORDORB_API_KEY environment variable is not set. "
            "Get a free key at https://wordorb.ai/pricing"
        )
    return {
        "Authorization": f"Bearer {WORDORB_API_KEY}",
        "User-Agent": "autogen-wordorb/1.0.0",
        "Accept": "application/json",
    }


def _get(path: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Make a GET request to the WordOrb API."""
    url = f"{WORDORB_BASE_URL}{path}"
    with httpx.Client(timeout=WORDORB_TIMEOUT) as client:
        resp = client.get(url, headers=_headers(), params=params or {})

    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After", "60")
        return {
            "error": f"Rate limit exceeded. Retry after {retry_after}s. "
            "Upgrade at https://wordorb.ai/pricing"
        }
    if resp.status_code == 401:
        return {"error": "Authentication failed. Check WORDORB_API_KEY."}
    if resp.status_code >= 400:
        try:
            return resp.json()
        except Exception:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    return resp.json()


# ---------------------------------------------------------------------------
# Tool functions (registered with AutoGen agents)
# ---------------------------------------------------------------------------


def lookup_word(
    word: Annotated[str, "The English word to look up (e.g. 'courage', 'serendipity')"],
    lang: Annotated[str, "ISO 639-1 language code for translations. Optional."] = "",
) -> str:
    """Look up any English word via the WordOrb dictionary API.

    Returns IPA pronunciation, definition, etymology, part of speech,
    example sentences, and translations in 47 languages.
    Backed by a 162K-word corpus.
    """
    params: Dict[str, str] = {}
    if lang:
        params["lang"] = lang

    data = _get(f"/api/word/{word}", params)
    if "error" in data:
        return f"Error: {data['error']}"
    return _format_lookup(data)


def get_lesson(
    day: Annotated[int, "Day of the year (1-365). Each day has unique content."] = 1,
    track: Annotated[
        str, "Learning track: learn, grow, teach, or trivia."
    ] = "learn",
    age: Annotated[str, "Age group: kid, adult, or elder."] = "adult",
) -> str:
    """Get a structured 5-phase vocabulary lesson from the WordOrb platform.

    Each lesson has five phases: hook, story, wonder, action, wisdom.
    365 days x 4 tracks x 3 age groups = 4,380 unique lessons.
    """
    params = {"day": str(day), "track": track, "age": age}
    data = _get("/api/lesson", params)
    if "error" in data:
        return f"Error: {data['error']}"
    return _format_lesson(data)


def get_quiz(
    day: Annotated[int, "Day (1-365), or 0 for random questions."] = 0,
    track: Annotated[str, "Learning track: learn, grow, teach, or trivia."] = "learn",
    question_type: Annotated[
        str, "Question type: mc, tf, ab, open, reflection. Empty for all types."
    ] = "",
    count: Annotated[int, "Number of random questions (1-20). Only for day=0."] = 5,
) -> str:
    """Get vocabulary quiz questions from the WordOrb platform.

    Supports six question types: mc (multiple choice), tf (true/false),
    ab (A/B comparison), open (free response), reflection, and quiz.
    """
    if day > 0:
        params: Dict[str, str] = {"day": str(day), "track": track}
        if question_type:
            params["type"] = question_type
        path = "/api/quiz"
    else:
        params = {"track": track, "count": str(min(count, 20))}
        if question_type:
            params["type"] = question_type
        path = "/api/quiz/random"

    data = _get(path, params)
    if "error" in data:
        return f"Error: {data['error']}"
    return _format_quiz(data)


def query_word_graph(
    word: Annotated[str, "The word to look up in the knowledge graph."],
) -> str:
    """Query the WordOrb knowledge graph for word-to-lesson connections.

    Returns all lessons where a word appears, related words, and importance
    scores. Useful for building learning paths.
    """
    data = _get("/api/graph/word", {"word": word})
    if "error" in data:
        return f"Error: {data['error']}"
    return _format_graph(data)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _format_lookup(data: Dict[str, Any]) -> str:
    parts = [f"Word: {data.get('word', 'unknown')}"]
    if data.get("ipa"):
        parts.append(f"IPA: {data['ipa']}")
    if data.get("pos"):
        parts.append(f"Part of speech: {data['pos']}")
    if data.get("definition") or data.get("def"):
        parts.append(f"Definition: {data.get('definition') or data.get('def')}")
    if data.get("etymology"):
        parts.append(f"Etymology: {data['etymology']}")
    if data.get("example"):
        parts.append(f"Example: {data['example']}")
    if data.get("synonyms"):
        syns = data["synonyms"]
        if isinstance(syns, list):
            syns = ", ".join(syns)
        parts.append(f"Synonyms: {syns}")
    if data.get("translations"):
        translations = data["translations"]
        if isinstance(translations, dict):
            tr_items = [f"  {k}: {v}" for k, v in list(translations.items())[:5]]
            parts.append("Translations:\n" + "\n".join(tr_items))
    return "\n".join(parts)


def _format_lesson(data: Dict[str, Any]) -> str:
    parts = [
        f"Lesson: Day {data.get('day', '?')} | Track: {data.get('track', '?')} "
        f"| Age: {data.get('age_group', '?')}",
    ]
    if data.get("title"):
        parts.append(f"Title: {data['title']}")
    if data.get("theme"):
        parts.append(f"Theme: {data['theme']}")
    phases = data.get("phases", {})
    for phase_name in ("hook", "story", "wonder", "action", "wisdom"):
        content = phases.get(phase_name)
        if content:
            parts.append(f"\n--- {phase_name.upper()} ---\n{content}")
    return "\n".join(parts)


def _format_quiz(data: Dict[str, Any]) -> str:
    interactions = data.get("interactions", [])
    if not interactions:
        return "No quiz questions found for the given parameters."
    parts = [
        f"Quiz: {len(interactions)} question(s) | Track: {data.get('track', '?')}"
    ]
    for i, q in enumerate(interactions, 1):
        parts.append(f"\nQ{i} [{q.get('type', '?')}]: {q.get('prompt', '')}")
        if q.get("choices"):
            choices = q["choices"]
            if isinstance(choices, list):
                for idx, c in enumerate(choices):
                    parts.append(f"  {chr(65 + idx)}. {c}")
        if q.get("answer"):
            parts.append(f"  Answer: {q['answer']}")
        if q.get("explanation"):
            parts.append(f"  Explanation: {q['explanation']}")
    return "\n".join(parts)


def _format_graph(data: Dict[str, Any]) -> str:
    parts = [
        f"Knowledge Graph: '{data.get('word', '?')}' appears in "
        f"{data.get('appears_in', 0)} lesson(s)"
    ]
    lessons = data.get("lessons", [])
    for lesson in lessons[:10]:
        parts.append(
            f"  Day {lesson.get('day', '?')} ({lesson.get('track', '?')}) "
            f"- {lesson.get('phase', '?')} phase, "
            f"importance={lesson.get('importance', '?')}"
        )
        if lesson.get("context"):
            parts.append(f"    Context: {lesson['context']}")
    related = data.get("related_words", [])
    if related:
        parts.append(f"\nRelated words: {', '.join(related)}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


def create_vocabulary_tutor() -> AssistantAgent:
    """Create a VocabularyTutor assistant agent with all WordOrb tools."""
    tutor = AssistantAgent(
        name="VocabularyTutor",
        system_message=(
            "You are an expert vocabulary tutor powered by the WordOrb platform. "
            "You can look up any word, retrieve structured lessons, generate quizzes, "
            "and explore the knowledge graph. Always use the tools for accurate data -- "
            "never invent definitions. Include IPA pronunciation and examples. "
            "After teaching a word, offer a quiz to reinforce learning."
        ),
        llm_config=LLM_CONFIG,
    )
    return tutor


def create_quiz_master() -> AssistantAgent:
    """Create a QuizMaster assistant agent specialized in assessments."""
    quiz_master = AssistantAgent(
        name="QuizMaster",
        system_message=(
            "You are a quiz master who assesses vocabulary knowledge. "
            "Use the get_quiz tool to retrieve questions from the WordOrb question bank. "
            "Present questions one at a time, wait for the student's answer, then "
            "reveal the correct answer with an explanation. Keep score and provide "
            "feedback at the end. Be encouraging but accurate."
        ),
        llm_config=LLM_CONFIG,
    )
    return quiz_master


def create_student_proxy() -> UserProxyAgent:
    """Create a student user proxy that executes tool calls."""
    student = UserProxyAgent(
        name="Student",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    return student


def create_tool_executor() -> UserProxyAgent:
    """Create a tool executor proxy that runs WordOrb tool functions."""
    executor = UserProxyAgent(
        name="ToolExecutor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
    )
    return executor


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def register_wordorb_tools(
    caller: AssistantAgent,
    executor: UserProxyAgent,
) -> None:
    """Register all WordOrb tool functions with an agent pair.

    The ``caller`` agent (assistant) will be able to invoke these functions,
    and the ``executor`` agent (user proxy) will execute them.

    Parameters
    ----------
    caller : AssistantAgent
        The assistant agent that will call the tools.
    executor : UserProxyAgent
        The user proxy that will execute the tool calls.
    """
    register_function(
        lookup_word,
        caller=caller,
        executor=executor,
        name="lookup_word",
        description=(
            "Look up any English word. Returns IPA pronunciation, definition, "
            "etymology, part of speech, and translations in 47 languages."
        ),
    )

    register_function(
        get_lesson,
        caller=caller,
        executor=executor,
        name="get_lesson",
        description=(
            "Get a structured vocabulary lesson with 5 phases "
            "(hook, story, wonder, action, wisdom). "
            "365 days x 4 tracks x 3 age groups."
        ),
    )

    register_function(
        get_quiz,
        caller=caller,
        executor=executor,
        name="get_quiz",
        description=(
            "Get vocabulary quiz questions. Specify a day for lesson-aligned "
            "questions, or day=0 for random questions. "
            "6 question types: mc, tf, ab, open, reflection, quiz."
        ),
    )

    register_function(
        query_word_graph,
        caller=caller,
        executor=executor,
        name="query_word_graph",
        description=(
            "Query the knowledge graph for a word. Returns all lessons "
            "where it appears, related words, and importance scores."
        ),
    )


# ---------------------------------------------------------------------------
# Example 1: Two-agent chat (tutor + student)
# ---------------------------------------------------------------------------


def run_two_agent_chat() -> None:
    """Run a simple two-agent chat: tutor teaches, student asks questions."""
    print("=" * 60)
    print("  WordOrb Vocabulary Tutor (AutoGen Two-Agent)")
    print("  Type your questions. The tutor will use WordOrb tools.")
    print("=" * 60)
    print()

    tutor = create_vocabulary_tutor()
    student = create_student_proxy()
    executor = create_tool_executor()

    # Register tools: tutor can call them, executor runs them
    register_wordorb_tools(caller=tutor, executor=executor)

    # Start the conversation
    student.initiate_chat(
        tutor,
        message=(
            "I want to learn about the word 'resilience'. "
            "Can you look it up and then give me a quiz?"
        ),
    )


# ---------------------------------------------------------------------------
# Example 2: Group chat (tutor + quiz_master + student)
# ---------------------------------------------------------------------------


def run_group_chat() -> None:
    """Run a group chat with three agents collaborating."""
    print("=" * 60)
    print("  WordOrb Vocabulary Tutor (AutoGen Group Chat)")
    print("  Agents: VocabularyTutor, QuizMaster, Student")
    print("=" * 60)
    print()

    tutor = create_vocabulary_tutor()
    quiz_master = create_quiz_master()
    student = create_student_proxy()
    executor = create_tool_executor()

    # Register tools for both assistants
    register_wordorb_tools(caller=tutor, executor=executor)
    register_wordorb_tools(caller=quiz_master, executor=executor)

    # Group chat setup
    group_chat = GroupChat(
        agents=[student, tutor, quiz_master, executor],
        messages=[],
        max_round=20,
        speaker_selection_method="auto",
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=LLM_CONFIG,
    )

    # Start the conversation
    student.initiate_chat(
        manager,
        message=(
            "I want to learn vocabulary today. Start by teaching me an "
            "interesting word from day 42 of the learn track, then quiz me on it."
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: choose between two-agent and group chat modes."""
    if not WORDORB_API_KEY:
        print("Error: Set the WORDORB_API_KEY environment variable.")
        print("  Get a free key at https://wordorb.ai/pricing")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    import argparse

    parser = argparse.ArgumentParser(description="WordOrb + AutoGen vocabulary tutor")
    parser.add_argument(
        "--mode",
        choices=["two-agent", "group"],
        default="two-agent",
        help="Chat mode: two-agent (tutor+student) or group (tutor+quiz_master+student)",
    )
    args = parser.parse_args()

    if args.mode == "group":
        run_group_chat()
    else:
        run_two_agent_chat()


if __name__ == "__main__":
    main()
