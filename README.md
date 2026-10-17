# Self-Learning AI Agent

A small terminal assistant that uses OpenAI for responses and mem0 for persistent, user-specific memory. It retrieves relevant memories before each answer and offers only the user's message to mem0 under an explicit policy for selecting durable memories. Assistant-generated answers and temporary conversation details are not sent as durable-memory input. This improves personalization; it does not fine-tune or update the language model's weights.

The memory policy prioritizes stable preferences, ongoing research or learning goals, recurring instructions, and user-stated or confirmed facts that can help in future conversations. Questions, hypotheticals, one-off requests, and unsupported inferences are not durable facts. Finance-related memories are retained only when the user's message states both the source or provenance and exact reporting period; both are kept with the fact.

## Setup

Python 3.10 or newer is required.

```bash
cd self-learning-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `.env`, then start the agent:

```bash
python agent.py
```

Memories are isolated by `MEM0_USER_ID` and stored in `data/qdrant/`. Keep that directory to retain memory between runs. The LLM and embedding requests are sent to OpenAI; mem0's memory extraction uses the configured OpenAI model.

## Try it

Tell the agent something it should remember, such as `I prefer concise answers and I'm learning Python.` Then ask a related question. On a later run with the same `MEM0_USER_ID`, it can retrieve that preference and personalize its answer. Type `/quit` to exit.

Set a different `MEM0_USER_ID` in `.env` to use an independent memory namespace. Don't share an ID between people.

## Commands

- `/memories` lists up to 100 memories for the current `MEM0_USER_ID`.
- `/search <query>` searches that user's memories.
- `/forget <memory_id>` deletes one of that user's memories. Use `/memories` or `/search` to find its ID.
- `/feedback good` or `/feedback bad` rates the most recent answer. Feedback is stored separately in `data/feedback.sqlite3` and is not added to Mem0.
- `/help` lists available commands, and `/quit` exits.