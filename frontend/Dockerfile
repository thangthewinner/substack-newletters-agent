# ---------- Build Stage ----------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Configure UV for optimal performance
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never

# Copy dependency files and sync dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy source code for Streamlit
COPY frontend ./frontend
COPY src/config.py ./src/config.py

# Also copy pyproject.toml and uv.lock for the final sync
COPY pyproject.toml uv.lock ./

# Install project dependencies into virtualenv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


# ---------- Runtime Stage ----------
FROM python:3.12-slim-bookworm

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy built application and virtualenv from builder
COPY --from=builder /app /app

# Set working directory and ownership
WORKDIR /app
RUN chown -R appuser:appuser /app

# Set Python path and environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
