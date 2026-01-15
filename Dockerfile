FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy project files
COPY . .

# Install Python dependencies using UV
RUN uv pip install -r requirements.txt --system

# Create directories for data and results
RUN mkdir -p data/raw data/processed results/models results/metrics

EXPOSE 8000

CMD ["python", "scripts/train.py"]
