"""
FastAPI REST API for Customer Journey GraphRAG.

Provides endpoints for querying journey data using both GraphRAG
and naive vector RAG approaches, with comparison capabilities.
"""

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .build_graph import get_graph_stats, load_or_build_graph
from .llm import JourneyLLM
from .naive_rag import load_or_build_naive_rag
from .retrieval import (
    query_category_exit_analysis,
    query_churned_user_journeys,
    query_high_vs_low_ltv,
    query_pre_purchase_behavior,
)


# ============================================================================
# Application State
# ============================================================================


class AppState:
    """Global application state for loaded resources."""

    graph = None
    naive_rag = None
    llm = None
    initialized = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Loads graph, naive RAG index, and LLM client on startup.
    """
    print("=" * 60)
    print("Initializing Customer Journey GraphRAG API...")
    print("=" * 60)

    try:
        # Load graph
        print("\n[1/3] Loading journey graph...")
        state.graph = load_or_build_graph()
        stats = get_graph_stats(state.graph)
        print(f"  ✓ Graph loaded: {stats['total_nodes']:,} nodes, {stats['total_edges']:,} edges")

        # Load naive RAG
        print("\n[2/3] Loading naive RAG index...")
        state.naive_rag = load_or_build_naive_rag()
        print(f"  ✓ Naive RAG loaded: {len(state.naive_rag.documents):,} documents")

        # Initialize LLM
        print("\n[3/3] Initializing LLM client...")
        state.llm = JourneyLLM()
        print(f"  ✓ LLM initialized: {state.llm.model}")

        state.initialized = True
        print("\n" + "=" * 60)
        print("API ready! Available at http://localhost:8000")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Initialization error: {e}")
        print("Some features may not be available.")

    yield

    # Cleanup
    print("Shutting down API...")


# ============================================================================
# FastAPI App
# ============================================================================


app = FastAPI(
    title="Customer Journey GraphRAG API",
    description="Graph-enhanced RAG for e-commerce customer journey analysis",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================


class QueryRequest(BaseModel):
    """Request model for query endpoints."""

    query: str = Field(..., description="Natural language product/analytics question")
    category: str | None = Field(None, description="Optional category filter")


class QueryResponse(BaseModel):
    """Response model for single-method queries."""

    query: str
    method: str
    context: str
    response: str
    latency_ms: float


class ComparisonResponse(BaseModel):
    """Response model for comparison queries."""

    query: str
    graphrag: QueryResponse
    naive_rag: QueryResponse


class StatsResponse(BaseModel):
    """Response model for graph statistics."""

    total_nodes: int
    total_edges: int
    node_types: dict[str, int]
    edge_types: dict[str, int]
    naive_rag_documents: int


# ============================================================================
# Preset Queries
# ============================================================================

PRESET_QUERIES = [
    {
        "id": "churned_journeys",
        "name": "Churned User Journeys",
        "query": "What's the typical journey of churned users? What patterns lead to churn?",
        "description": "Analyzes journey patterns of users who churned",
    },
    {
        "id": "pre_purchase_electronics",
        "name": "Pre-Purchase Behavior (Electronics)",
        "query": "Which products do users view before purchasing electronics? What's the typical path to conversion?",
        "category": "Electronics",
        "description": "Examines browsing behavior before electronics purchases",
    },
    {
        "id": "high_vs_low_ltv",
        "name": "High-LTV vs Low-LTV Comparison",
        "query": "How do high-LTV users browse differently from low-LTV users? What behaviors distinguish them?",
        "description": "Compares journey patterns between value segments",
    },
    {
        "id": "fashion_exit",
        "name": "Fashion Category Exit Analysis",
        "query": "Why do users drop off after viewing fashion products? What are common exit patterns?",
        "category": "Fashion",
        "description": "Analyzes drop-off points in fashion category journeys",
    },
    {
        "id": "conversion_funnel",
        "name": "Conversion Funnel Analysis",
        "query": "What does the conversion funnel look like? Where do most users drop off in the purchase journey?",
        "description": "Examines the overall conversion funnel",
    },
]


# ============================================================================
# Helper Functions
# ============================================================================


def get_graphrag_context(query: str, category: str | None = None) -> str:
    """
    Get context using GraphRAG retrieval based on query intent.

    Args:
        query: User query string.
        category: Optional category filter.

    Returns:
        Context string for LLM.
    """
    query_lower = query.lower()

    # Route to appropriate retrieval function based on query intent
    if "churn" in query_lower or "drop" in query_lower:
        return query_churned_user_journeys(state.graph, sample_size=30)

    elif "high" in query_lower and "low" in query_lower:
        return query_high_vs_low_ltv(state.graph)

    elif "before" in query_lower and "purchase" in query_lower:
        return query_pre_purchase_behavior(state.graph, category)

    elif "exit" in query_lower and category:
        return query_category_exit_analysis(state.graph, category)

    elif category:
        # Default to category exit analysis if category specified
        return query_category_exit_analysis(state.graph, category)

    else:
        # Default to churned user analysis
        return query_churned_user_journeys(state.graph, sample_size=30)


def get_naive_context(query: str, top_k: int = 10) -> str:
    """
    Get context using naive vector RAG.

    Args:
        query: User query string.
        top_k: Number of documents to retrieve.

    Returns:
        Context string for LLM.
    """
    return state.naive_rag.retrieve_context(query, top_k=top_k)


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy" if state.initialized else "initializing",
        "graph_loaded": state.graph is not None,
        "naive_rag_loaded": state.naive_rag is not None,
        "llm_loaded": state.llm is not None,
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get graph and index statistics."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="API not fully initialized")

    stats = get_graph_stats(state.graph)
    return StatsResponse(
        total_nodes=stats["total_nodes"],
        total_edges=stats["total_edges"],
        node_types=stats["node_types"],
        edge_types=stats["edge_types"],
        naive_rag_documents=len(state.naive_rag.documents),
    )


@app.get("/presets")
async def get_presets() -> list[dict[str, Any]]:
    """Get available preset queries."""
    return PRESET_QUERIES


@app.post("/query/graphrag", response_model=QueryResponse)
async def query_graphrag(request: QueryRequest) -> QueryResponse:
    """
    Query using GraphRAG retrieval.

    Uses graph traversal and path analysis to retrieve contextually
    relevant journey information for the LLM.
    """
    if not state.initialized:
        raise HTTPException(status_code=503, detail="API not fully initialized")

    start_time = time.time()

    # Get GraphRAG context
    context = get_graphrag_context(request.query, request.category)

    # Generate response
    response = state.llm.generate(request.query, context)

    latency_ms = (time.time() - start_time) * 1000

    return QueryResponse(
        query=request.query,
        method="graphrag",
        context=context,
        response=response,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/query/naive", response_model=QueryResponse)
async def query_naive(request: QueryRequest) -> QueryResponse:
    """
    Query using naive vector RAG retrieval.

    Uses semantic similarity search over session documents
    without understanding of graph structure.
    """
    if not state.initialized:
        raise HTTPException(status_code=503, detail="API not fully initialized")

    start_time = time.time()

    # Get naive RAG context
    context = get_naive_context(request.query)

    # Generate response
    response = state.llm.generate(request.query, context)

    latency_ms = (time.time() - start_time) * 1000

    return QueryResponse(
        query=request.query,
        method="naive",
        context=context,
        response=response,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/query/compare", response_model=ComparisonResponse)
async def query_compare(request: QueryRequest) -> ComparisonResponse:
    """
    Compare GraphRAG and naive RAG for the same query.

    Runs both retrieval methods and returns side-by-side results
    for quality comparison.
    """
    if not state.initialized:
        raise HTTPException(status_code=503, detail="API not fully initialized")

    # GraphRAG
    graphrag_start = time.time()
    graphrag_context = get_graphrag_context(request.query, request.category)
    graphrag_response = state.llm.generate(request.query, graphrag_context)
    graphrag_latency = (time.time() - graphrag_start) * 1000

    # Naive RAG
    naive_start = time.time()
    naive_context = get_naive_context(request.query)
    naive_response = state.llm.generate(request.query, naive_context)
    naive_latency = (time.time() - naive_start) * 1000

    return ComparisonResponse(
        query=request.query,
        graphrag=QueryResponse(
            query=request.query,
            method="graphrag",
            context=graphrag_context,
            response=graphrag_response,
            latency_ms=round(graphrag_latency, 2),
        ),
        naive_rag=QueryResponse(
            query=request.query,
            method="naive",
            context=naive_context,
            response=naive_response,
            latency_ms=round(naive_latency, 2),
        ),
    )


# ============================================================================
# CLI Entry Point
# ============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
