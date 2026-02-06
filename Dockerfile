# HuggingFace Spaces Docker image for Customer Journey GraphRAG
# Runs FastAPI backend + Streamlit frontend

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy pre-built graph and index artifacts
COPY graph/ ./graph/

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Run startup script
CMD ["./start.sh"]
