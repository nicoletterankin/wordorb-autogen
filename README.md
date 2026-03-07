# AutoGen + WordOrb Integration

Use the [WordOrb.ai](https://wordorb.ai) vocabulary and lesson API as tool
functions inside [Microsoft AutoGen](https://microsoft.github.io/autogen/) agents.

## Setup

```bash
pip install pyautogen httpx
export WORDORB_API_KEY="wo_live_abc123"
export OPENAI_API_KEY="sk-..."
```

Get a free WordOrb API key at [wordorb.ai/pricing](https://wordorb.ai/pricing).

## Tool functions

| Function | Description |
|----------|-------------|
| `lookup_word` | Dictionary lookup -- IPA, definition, etymology, 47 languages |
| `get_lesson` | 5-phase lesson (hook/story/wonder/action/wisdom) |
| `get_quiz` | Day-aligned or random quiz questions (6 types) |
| `query_word_graph` | Knowledge graph: word-to-lesson connections |

## Quick start -- register tools with any agent

```python
from autogen import AssistantAgent, UserProxyAgent
from wordorb_agent import register_wordorb_tools

tutor = AssistantAgent(
    name="VocabularyTutor",
    system_message="You are a vocabulary tutor using WordOrb tools.",
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": "sk-..."}]},
)

executor = UserProxyAgent(
    name="ToolExecutor",
    human_input_mode="NEVER",
    code_execution_config=False,
)

# Register all four WordOrb tools
register_wordorb_tools(caller=tutor, executor=executor)
```

## Two-agent chat example

A tutor agent teaches vocabulary using WordOrb tools, while the student asks
questions interactively:

```bash
python wordorb_agent.py --mode two-agent
```

## Group chat example

Three agents collaborate: a vocabulary tutor researches words, a quiz master
administers assessments, and the student interacts:

```bash
python wordorb_agent.py --mode group
```

| Agent | Role | Tools |
|-------|------|-------|
| VocabularyTutor | Looks up words, retrieves lessons, explores graph | All 4 |
| QuizMaster | Generates and evaluates quiz questions | All 4 |
| Student | User proxy for interactive input | None (human) |
| ToolExecutor | Executes tool function calls | All 4 (executor) |

## Using individual tool functions

The tool functions can also be called directly without agents:

```python
from wordorb_agent import lookup_word, get_lesson, get_quiz, query_word_graph

# Look up a word
print(lookup_word("serendipity"))

# Get a lesson
print(get_lesson(day=42, track="learn", age="adult"))

# Get random quiz questions
print(get_quiz(day=0, track="learn", count=5))

# Explore the knowledge graph
print(query_word_graph("courage"))
```

## AutoGen tool registry guide

To register these tools with the AutoGen ecosystem:

1. **Package**: Publish as a PyPI package (e.g. `autogen-wordorb`).

2. **Tool registry**: AutoGen tools are registered using `register_function()`.
   The `register_wordorb_tools()` helper handles this for all four functions.

3. **Documentation**: Open a PR to `microsoft/autogen` under
   `website/docs/topics/tool-use/` with:
   - Tool function signatures and descriptions
   - Setup instructions (API key, environment variables)
   - Working two-agent and group chat examples
   - Link to WordOrb API docs at `https://wordorb.ai/docs`

4. **Notebook**: Include a Jupyter notebook in `notebook/` demonstrating
   the integration end-to-end.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `WORDORB_API_KEY` | Yes | WordOrb API key (starts with `wo_`) |
| `OPENAI_API_KEY` | Yes | OpenAI API key for the LLM |
| `WORDORB_BASE_URL` | No | Override API URL (default: `https://api.wordorb.ai`) |

## License

MIT
