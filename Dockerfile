FROM python:3.14-slim

# Install uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uv /uvx /bin/

WORKDIR /app

# Install compilation tools needed for C dependencies (e.g. greenlet, sentence-transformers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy python project settings to container
COPY pyproject.toml uv.lock ./

# Tell uv to create the virtual environment at /venv instead of .venv (outside project mount)
ENV UV_PROJECT_ENVIRONMENT=/venv

# Synchronize dependencies with uv (frozen to lock down dependencies, no-dev to keep it light)
# Since package = false in tool.uv, it won't try to build this app as a package.
RUN uv sync --frozen --no-dev --no-install-project

# Copy remaining code files
COPY . .

# Set environment PATH to prioritize uv virtual environment
ENV PATH="/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Run FastAPI API Gateway
CMD ["python", "-m", "api.main"]



