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

Add your OpenAI API key to `.env`, then start the terminal agent:

```bash
python agent.py
```

### Ingest Reliance annual reports

Place Reliance annual report PDFs in `data/reports/reliance/`, with the financial year in each filename (for example, `RIL_FY2024-25.pdf`). From the project root, run:

```bash
python ingest.py
```

The script reports discovered files and each initialization stage before embedding. It reads PDFs locally, sends extracted text to OpenAI to create embeddings, and stores vectors in the local `data/financial_qdrant/` directory. Keep the terminal open while it processes; API embedding calls can take time.

### Extract reported financial tables for review

Run the table extractor from the project root:

```bash
python extract_financials.py
```

It writes one raw JSON file per PDF to `financial_data/`. Each extracted table includes its source PDF, financial year, PDF page, table index, and cleaned rows. The extractor does not classify tables as financial statements, verify figures, or calculate metrics; review the extracted tables before using any values. It reads from `data/reports/reliance/`; when that folder has no PDFs, it explicitly reports and falls back to the PDFs in `static/reliance/`.

### Research with cited evidence

Use `research.py` for a non-streaming research answer with structured source metadata:

```bash
python research.py "What was Reliance's consolidated revenue?" --year FY2024-25
```

The Python API is `research(question, financial_year=None, top_k=8)` and returns an `answer` plus `sources` containing citation ID, report filename, financial year, PDF page, and retrieved passage text. It requires the report collection to have been built with `ingest.py`. The chat UI uses the same evidence-grounded research workflow in streaming mode and displays the retrieved report filenames and pages beneath each answer.

To start the chat UI locally, run this from the project directory:

```bash
python web_app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The UI streams answer tokens as they arrive, displays the report sources and PDF pages used, and lets you rate an answer or submit an explanation and confirmed correction. Answers are grounded in retrieved Reliance report passages; user memories and confirmed feedback may personalize responses but are not treated as financial evidence.

### Running on a server

Copy the project to a private Linux server, install its requirements, and put `.env` on that server (do not commit or upload your API key to source control). Run the production WSGI server from the project directory:

```bash
gunicorn --workers 1 --bind 127.0.0.1:8000 web_app:app
```

Put an HTTPS reverse proxy with authentication in front of `127.0.0.1:8000`. Do not expose the app directly to the public internet: this initial UI has no user login and uses the single `MEM0_USER_ID` configured for the process, so everyone reaching it would share that user's memories and feedback. Keep `data/` persistent and backed up; it contains the local Qdrant data and feedback database. For a private single-user server, access it through the authenticated proxy or a VPN.

Memories are isolated by `MEM0_USER_ID` and stored in `data/qdrant/`. Keep that directory to retain memory between runs. The LLM and embedding requests are sent to OpenAI; mem0's memory extraction uses the configured OpenAI model.

## Try it

Tell the agent something it should remember, such as `I prefer concise answers and I'm learning Python.` Then ask a related question. On a later run with the same `MEM0_USER_ID`, it can retrieve that preference and personalize its answer. Type `/quit` to exit.

Set a different `MEM0_USER_ID` in `.env` to use an independent memory namespace. Don't share an ID between people.

## Commands

- `/memories` lists up to 100 memories for the current `MEM0_USER_ID`.
- `/search <query>` searches that user's memories.
- `/forget <memory_id>` deletes one of that user's memories. Use `/memories` or `/search` to find its ID.
- `/feedback good` or `/feedback bad` rates the most recent answer. For a bad answer, the agent can save an explanation and optional correction in `data/feedback.sqlite3`. It asks you to confirm those details before using them as lessons in future prompts; this confirmation is user-provided, not independent fact-checking. Ratings alone are never retrieved as verified lessons.
- `/help` lists available commands, and `/quit` exits.