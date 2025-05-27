# 1. Use an official Python runtime as a parent image
# Using the slim-bullseye version for a smaller image size
FROM python:3.11-slim-bullseye AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100
# Poetry env vars (uv uses some of these)
# Or your desired Poetry version
ENV POETRY_VERSION=1.7.1
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set work directory
WORKDIR /app

# --- Builder Stage --- #
FROM base AS builder

RUN pip install uvicorn

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy the entire project first
COPY . .

# Install dependencies using uv
RUN uv pip install --system --no-cache .

# --- Final Stage --- #
FROM base AS final

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Ensure alembic directory and ini are copied if they exist (they will soon)
# COPY ./alembic.ini /app/  # Temporarily commented out until alembic init is run
# COPY ./alembic /app/alembic # Temporarily commented out until alembic init is run

EXPOSE 8501

# Command to run the application using uvicorn
# Use --host 0.0.0.0 to make it accessible from outside the container
CMD ["streamlit", "run", "app.py"]