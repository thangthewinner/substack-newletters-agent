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


## Qdrant commands
qdrant-create-collection: ## Create Qdrant collection
	@echo "Creating Qdrant collection..."
	uv run python -m src.infrastructure.qdrant.create_collection

qdrant-delete-collection: ## Delete Qdrant collection
	@echo "Deleting Qdrant collection..."
	uv run python -m src.infrastructure.qdrant.delete_collection

qdrant-create-index: ## Create Qdrant index
	@echo "Updating HNSW and creating Qdrant indexes..."
	uv run python -m src.infrastructure.qdrant.create_indexes

qdrant-ingest-from-sql: ## Ingest data from SQL to Qdrant
	@echo "Ingesting data from SQL to Qdrant..."
	uv run python -m src.infrastructure.qdrant.ingest_from_sql
	@echo "Data ingestion complete."

## Prefect Flow Commands 
ingest-rss-articles-flow: ## Ingest RSS articles flow
	@echo "Running ingest RSS articles flow..."
	uv run python -m src.pipelines.flows.rss_ingestion_flow
	@echo "Ingest RSS articles flow completed."

ingest-embeddings-flow: ## Ingest embeddings flow
	@echo "Running ingest embeddings flow..."
	$(if $(FROM_DATE), \
		uv run python -m src.pipelines.flows.embeddings_ingestion_flow --from-date $(FROM_DATE), \
		uv run python -m src.pipelines.flows.embeddings_ingestion_flow)
	@echo "Ingest embeddings flow completed."


## Prefect Deployment Commands
deploy-cloud-flows: ## Deploy Prefect flows to Prefect Cloud
	@echo "Deploying Prefect flows to Prefect Cloud..."
	prefect deploy --prefect-file prefect-cloud.yaml
	@echo "Prefect Cloud deployment complete."

deploy-local-flows: ## Deploy Prefect flows to Prefect Local Server
	@echo "Deploying Prefect flows to Prefect Local Server..."
	prefect deploy --prefect-file prefect-local.yaml
	@echo "Prefect Local deployment complete."

## Recreate Commands 
recreate-supabase: supabase-delete supabase-create ## Recreate Supabase resources

recreate-qdrant: qdrant-delete-collection qdrant-create-collection ## Recreate Qdrant resources

recreate-all: supabase-delete qdrant-delete-collection supabase-create qdrant-create-collection ## Recreate Qdrant and Supabase resources