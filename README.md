# Substack Newsletter Chatbot

This project ingests Substack newsletters from RSS, stores articles in PostgreSQL (Supabase), indexes them in Qdrant, and exposes a chat-first interface powered by a LangGraph agent.

## What It Does

- Fetches newsletter articles from RSS feeds
- Stores article metadata and content in PostgreSQL
- Splits and embeds article content into Qdrant
- Exposes a FastAPI backend for search and chat
- Provides a Gradio chatbot UI for interactive exploration

## Tech Stack

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- FastAPI
- Gradio
- Prefect
- Qdrant
- PostgreSQL / Supabase
- LangChain / LangGraph
- OpenRouter

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Create the environment file

```bash
cp .env.example .env
```

At minimum, configure:

- `SUPABASE_DB__*`
- `QDRANT__*`
- `OPENROUTER__API_KEY`
- `BACKEND_URL`

Optional but useful:

- `LANGCHAIN_API_KEY`
- `LANGCHAIN_PROJECT`
- `LANGCHAIN_TRACING_V2`

### 3. Configure RSS feeds

Edit `src/configs/feeds_rss.yaml`.

Example:

```yaml
feeds:
  - name: "AI Echoes"
    author: "Benito Martin"
    url: "https://aiechoes.substack.com/feed"
```

## Run the App

### Start the API

```bash
make run-api
```

### Start the chatbot UI

```bash
make run-gradio
```

By default:

- API: `http://localhost:8080`
- Gradio UI: `http://localhost:7860`

## API Endpoints

- `GET /`
- `GET /health`
- `GET /ready`
- `POST /unique-titles`
- `POST /chat`
- `POST /chat/stream`

## Data Pipeline

### Create storage resources

```bash
make supabase-create
make qdrant-create-collection
make qdrant-create-index
```

### Ingest RSS articles into PostgreSQL

```bash
make ingest-rss-articles-flow
```

### Ingest embeddings into Qdrant

```bash
make ingest-embeddings-flow
```

Or from a specific date:

```bash
make ingest-embeddings-flow FROM_DATE=2025-01-01
```

## Prefect Deployment

### Deploy to Prefect Cloud

```bash
make deploy-cloud-flows
```

Notes:

- The target work pool must already exist in Prefect Cloud
- The deployment reads database and Qdrant values from environment variables

## Project Structure

```text
src/
├── api/              # FastAPI app, routes, request/response models
├── infrastructure/   # Qdrant and PostgreSQL integrations
├── models/           # Domain and SQLAlchemy models
├── pipelines/        # Prefect flows and tasks
├── utils/            # Shared utilities
└── config.py         # Application settings

frontend/
└── app.py            # Gradio chatbot UI
```