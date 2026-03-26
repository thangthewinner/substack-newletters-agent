# Check if .env exists 
ifeq (,$(wildcard .env))
$(error .env file is not found)
endif 

# Load environment from .env file 
include .env 

.PHONY: tests mypy clean help ruff-check ruff-check-fix ruff-format ruff-format-fix all-check all-fix 

## Supabase commands 
supabase-create: 
	@echo "Creating Supabase database..."
	uv run python -m src.infrastructure.supabase.create_db

supabase-delete:
	@echo "Deleting Supabase database..."
	uv run python -m src.pipelines.flows.delete_db


## Prefect Flow Commands 
ingest-rss-articles-flow: ## Ingest RSS articles flow
	@echo "Running ingest RSS articles flow..."
	uv run python -m src.pipelines.flows.rss_ingestion_flow
	@echo "Ingest RSS articles flow completed."