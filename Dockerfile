# 1. Use an official Python runtime as a parent image
# Using the slim-bullseye version for a smaller image size
FROM python:3.11-slim-bullseye AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV UV_CACHE_DIR=/opt/uv-cache
ENV UV_LINK_MODE=copy

# Set work directory
WORKDIR /app

# --- Builder Stage --- #
FROM base AS builder

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create cache directory
RUN mkdir -p /opt/uv-cache

# Copy pyproject.toml and uv.lock first (for better Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY . .

# --- Final Stage --- #
FROM base AS final

# Install uv in final stage for runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy the virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Ensure alembic directory and ini are copied if they exist (they will soon)
# COPY ./alembic.ini /app/  # Temporarily commented out until alembic init is run
# COPY ./alembic /app/alembic # Temporarily commented out until alembic init is run

# Add virtual environment to path
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Command to run the FastAPI application using uvicorn
# Use --host 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]