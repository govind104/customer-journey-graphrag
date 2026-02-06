---
title: Customer Journey GraphRAG
emoji: 🔀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: Graph-enhanced RAG for e-commerce customer journey analytics
---

# Customer Journey GraphRAG

**Graph-enhanced RAG vs Vector RAG comparison for product analytics**

This demo showcases how **Graph-enhanced Retrieval Augmented Generation (GraphRAG)** outperforms traditional vector-based RAG for customer journey analytics. Built on synthetic e-commerce clickstream data (5,000 users, 85K+ events), it lets you compare both approaches side-by-side on real product analytics questions.

## Tech Stack

- **Graph Engine:** NetworkX (temporal user behavior graph)
- **LLM:** Llama 3.1 8B via Groq API
- **Vector Store:** FAISS with sentence-transformers
- **Backend:** FastAPI
- **Frontend:** Streamlit

## Key Results

| Metric | Naive RAG | GraphRAG |
|--------|-----------|----------|
| Statistics per query | 0–1 (10-sample) | 3–5 (cohort-level) |
| Pattern identification | Qualitative | Explicit sequences with % |
| Cohort comparison | No | Yes (e.g., 5.72 vs 2.31 events/session) |

## How to Use

1. **Select a preset query** from the dropdown (e.g., "What's the typical journey of churned users?")
2. **Click Compare** to run both GraphRAG and Naive RAG
3. **View side-by-side results** with response times and retrieved context
4. **Notice the difference:** GraphRAG provides quantitative path-based insights; Naive RAG gives generic descriptions

## Example Queries

- Journey patterns of churned users
- Pre-purchase behavior for electronics
- High-LTV vs Low-LTV browsing differences
- Fashion category exit analysis
- Conversion funnel drop-off points

---

*Source: [GitHub Repository](https://github.com/govind104/customer-journey-graphrag)*
