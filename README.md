# Substack Newsletter Search Pipeline

Ingest Substack newsletters via RSS, store in Supabase (PostgreSQL), and provide vector search via Qdrant.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for package management
- Supabase PostgreSQL database
- Qdrant vector store (local or cloud)

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables:
- `SUPABASE_DB__*` - Database connection (host, user, password, port)
- `QDRANT__URL` - Qdrant server URL
- `QDRANT__API_KEY` - Qdrant API key

Optional variables (for embeddings):
- `JINA__API_KEY` - Jina AI for embeddings
- `HUGGING_FACE__API_KEY` - HuggingFace for embeddings

### 3. Configure RSS Feeds

Edit `src/configs/feeds_rss.yaml` to add/remove Substack feeds:

```yaml
feeds:
  - name: "AI Echoes"
    author: "Benito Martin"
    url: "https://aiechoes.substack.com/feed"
```

## Running the Pipeline

### Create Database Table

```bash
make supabase-create
```

### Create Qdrant Collection

```bash
make qdrant-create-collection
```

### Create Qdrant Indexes (HNSW, text indexes)

```bash
make qdrant-create-index
```

### Ingest RSS Articles to Database

```bash
make ingest-rss-articles-flow
```

Or directly:
```bash
uv run python -m src.pipelines.flows.rss_ingestion_flow
```

### Generate Embeddings and Ingest to Qdrant

```bash
make ingest-embeddings-flow
```

With specific date:
```bash
make ingest-embeddings-flow FROM_DATE=2025-01-01
```

Or directly:
```bash
uv run python -m src.pipelines.flows.embeddings_ingestion_flow --from-date 2025-01-01
```

## Prefect Deployment (Optional)

Deploy flows to Prefect Cloud:
```bash
make deploy-cloud-flows
```

Deploy to local Prefect server:
```bash
make deploy-local-flows
```

## Quick Start (All-in-One)

```bash
# 1. Create database and Qdrant resources
make supabase-create
make qdrant-create-collection
make qdrant-create-index

# 2. Run pipelines
make ingest-rss-articles-flow
make ingest-embeddings-flow
```

## Project Structure

```
src/
├── config.py              # Pydantic settings
├── configs/               # YAML configurations (feeds)
├── models/                # Data models
├── pipelines/
│   ├── flows/            # Prefect flows
│   └── tasks/            # Prefect tasks
├── infrastructure/       # External integrations
│   ├── supabase/         # Database operations
│   └── qdrant/           # Vector store operations
└── utils/               # Utilities
```