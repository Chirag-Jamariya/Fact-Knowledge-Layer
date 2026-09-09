# ==============================================================================
# Production Dockerfile for Cross-Document Fact Knowledge Layer
# ==============================================================================
FROM python:3.12-slim-bookworm

# System dependencies for PDF parsing, OCR, and native C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tesseract-ocr \
    poppler-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=120 \
    UV_CONCURRENT_DOWNLOADS=4 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Install Python dependencies
COPY pyproject.toml uv.lock* ./
RUN uv pip install --system -r pyproject.toml

# Copy application source code
COPY src/ ./src/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/
COPY README.md ./

# Create data directories if needed
RUN mkdir -p /app/data/knowledge_store /app/data/uploads

# Expose Streamlit port
EXPOSE 8501

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch application
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
