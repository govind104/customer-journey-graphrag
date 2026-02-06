# HuggingFace Spaces Deployment Guide

Deploy Customer Journey GraphRAG to HuggingFace Spaces using Docker.

## Prerequisites

- HuggingFace account
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Git installed locally

## File Structure for HF Spaces

Upload these files to your HF Space repository:

```
your-space/
├── README.md           # HF Spaces version (README_HFSpaces.md, with YAML frontmatter)
├── Dockerfile          # Docker build configuration
├── requirements.txt    # Python dependencies
├── start.sh            # Startup script
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── app.py
│   ├── build_graph.py
│   ├── generate_data.py
│   ├── llm.py
│   ├── naive_rag.py
│   └── retrieval.py
└── graph/
    ├── journey_graph.pkl      # Pre-built graph (~50MB)
    └── naive_rag_index.pkl    # Pre-built FAISS index (~100MB)
```

> **Note:** `README.md` is for your GitHub repository, not HF Spaces.

## Deployment Steps

### 1. Create HuggingFace Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Configure:
   - **Space name:** `customer-journey-graphrag`
   - **SDK:** Docker
   - **Visibility:** Public or Private
4. Click **Create Space**

### 2. Set Up Secrets

1. Go to Space **Settings** → **Repository secrets**
2. Add secret:
   - **Name:** `GROQ_API_KEY`
   - **Value:** Your Groq API key

### 3. Clone and Push Files

```bash
# Clone your new space
git clone https://huggingface.co/spaces/YOUR_USERNAME/customer-journey-graphrag
cd customer-journey-graphrag

# Copy deployment files from your project
cp /path/to/project/README.md .           # HF version
cp /path/to/project/Dockerfile .
cp /path/to/project/requirements.txt .
cp /path/to/project/start.sh .
cp -r /path/to/project/src .
cp -r /path/to/project/graph .

# Push to HF Spaces
git add .
git commit -m "Initial deployment"
git push
```

### 4. Monitor Build

1. Go to your Space page
2. Click **Logs** tab to watch Docker build
3. Build typically takes 5-10 minutes
4. Once ready, app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/customer-journey-graphrag`

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           HuggingFace Space Container               │
│                                                     │
│   ┌─────────────────┐     ┌───────────────────┐     │
│   │   Streamlit     │────▶│    FastAPI        │    │
│   │   (port 7860)   │     │    (port 8000)    │     │
│   │   [Frontend]    │     │    [Backend]      │     │
│   └────────┬────────┘     └─────────┬─────────┘     │
│            │                        │               │
│            ▼                        ▼               │
│   ┌─────────────────────────────────────────────┐   │
│   │           Graph + FAISS Artifacts           │   │
│   │   journey_graph.pkl  │  naive_rag_index.pkl │   │
│   └─────────────────────────────────────────────┘   │
│                                                     │
│                        │                            │
│                        ▼                            │
│               ┌─────────────────┐                   │
│               │   Groq API      │                   │
│               │   (Llama 3.1)   │                   │
│               └─────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

## Troubleshooting

### Build Fails

- Check Docker logs for errors
- Ensure `graph/` directory contains both `.pkl` files
- Verify `requirements.txt` has all dependencies

### App Shows Error

- Verify `GROQ_API_KEY` secret is set correctly
- Check container logs for API connection errors

### Slow Startup

- First load takes ~60s to initialize models
- Subsequent loads are faster

## Local Docker Testing

Test the Docker build locally before pushing:

```bash
cd "Graph-RAG for Customer Journey Intelligence"

# Build image
docker build -t graphrag-journey .

# Run with API key
docker run -e GROQ_API_KEY=$GROQ_API_KEY -p 7860:7860 graphrag-journey

# Open http://localhost:7860
```

## File Responsibilities

| File | Purpose |
|------|---------|
| `README.md` | HF Spaces metadata (YAML frontmatter) + short description |
| `README_GITHUB.md` | Full documentation for GitHub repository |
| `Dockerfile` | Container build instructions |
| `start.sh` | Launches FastAPI + Streamlit on container start |
| `requirements.txt` | Python dependencies for pip |
| `graph/*.pkl` | Pre-computed graph and index artifacts |
