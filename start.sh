#!/bin/bash
# Startup script for HuggingFace Spaces
# Runs FastAPI backend and Streamlit frontend concurrently

set -e

echo "🚀 Starting Customer Journey GraphRAG..."

# Start FastAPI backend in background
echo "📡 Starting FastAPI backend on port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for FastAPI to be ready
echo "⏳ Waiting for FastAPI to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ FastAPI is ready!"
        break
    fi
    sleep 1
done

# Start Streamlit frontend (foreground, on HF Spaces port)
echo "🎨 Starting Streamlit frontend on port 7860..."
exec streamlit run src/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
