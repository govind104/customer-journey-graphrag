"""
Streamlit UI for Customer Journey GraphRAG.

Interactive dashboard for comparing GraphRAG vs naive vector RAG
on product analytics questions.
"""

import os
import time

import httpx
import streamlit as st

# ============================================================================
# Configuration
# ============================================================================

# Allow API URL override for Docker deployment (default: localhost:8000)
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Customer Journey GraphRAG",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .method-card {
        background: rgba(100, 100, 100, 0.1);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: inherit;
        line-height: 1.6;
    }
    .graphrag-card {
        border-left: 4px solid #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    .naive-card {
        border-left: 4px solid #f093fb;
        background: rgba(240, 147, 251, 0.1);
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        color: white;
    }
    .context-expander {
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        color: inherit;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# API Client
# ============================================================================


def check_api_health() -> dict | None:
    """Check if API is available and healthy."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        return response.json()
    except Exception:
        return None


def get_stats() -> dict | None:
    """Get graph statistics from API."""
    try:
        response = httpx.get(f"{API_BASE_URL}/stats", timeout=10.0)
        return response.json()
    except Exception:
        return None


def get_presets() -> list[dict]:
    """Get preset queries from API."""
    try:
        response = httpx.get(f"{API_BASE_URL}/presets", timeout=5.0)
        return response.json()
    except Exception:
        return []


def query_compare(query: str, category: str | None = None) -> dict | None:
    """Run comparison query through API."""
    try:
        payload = {"query": query}
        if category:
            payload["category"] = category

        response = httpx.post(
            f"{API_BASE_URL}/query/compare",
            json=payload,
            timeout=60.0,
        )
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# ============================================================================
# UI Components
# ============================================================================


def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">🛒 Customer Journey GraphRAG</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Compare Graph-enhanced RAG vs Vector RAG for product analytics</p>',
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with statistics and info."""
    with st.sidebar:
        st.header("📊 System Status")

        # Check API health
        health = check_api_health()

        if health and health.get("status") == "healthy":
            st.success("✅ API Connected")

            # Show stats
            stats = get_stats()
            if stats:
                st.metric("Graph Nodes", f"{stats['total_nodes']:,}")
                st.metric("Graph Edges", f"{stats['total_edges']:,}")
                st.metric("RAG Documents", f"{stats['naive_rag_documents']:,}")

                with st.expander("Node Types"):
                    for node_type, count in stats["node_types"].items():
                        st.text(f"{node_type}: {count:,}")

                with st.expander("Edge Types"):
                    for edge_type, count in stats["edge_types"].items():
                        st.text(f"{edge_type}: {count:,}")
        else:
            st.error("❌ API Not Available")
            st.info(
                """
                Start the API server:
                ```
                uv run uvicorn src.api:app
                ```
                """
            )

        st.divider()

        st.header("ℹ️ About")
        st.markdown(
            """
            **GraphRAG** uses graph traversal and path analysis
            to retrieve journey patterns with temporal structure.

            **Naive RAG** uses semantic similarity over text
            documents without graph awareness.

            Compare responses to see how graph structure
            improves product insights!
            """
        )


def render_query_section():
    """Render the query input section."""
    st.header("🔍 Query")

    # Get presets
    presets = get_presets()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Preset selection
        preset_options = ["Custom Query"] + [p["name"] for p in presets]
        selected_preset = st.selectbox("Select a preset query or enter custom:", preset_options)

        if selected_preset == "Custom Query":
            query = st.text_area(
                "Enter your product/analytics question:",
                placeholder="e.g., What's the typical journey of churned users?",
                height=100,
            )
            category = st.text_input(
                "Category filter (optional):",
                placeholder="e.g., Electronics, Fashion",
            )
        else:
            # Find selected preset
            preset = next((p for p in presets if p["name"] == selected_preset), None)
            if preset:
                query = preset["query"]
                category = preset.get("category", "")
                st.info(f"📝 {preset['description']}")
                st.text_area("Query:", value=query, disabled=True, height=80)
                if category:
                    st.text(f"Category: {category}")

    with col2:
        st.markdown("### Quick Tips")
        st.markdown(
            """
            - Ask about **churned users** to see dropout patterns
            - Ask about **high-LTV vs low-LTV** for cohort comparison
            - Specify a **category** for focused analysis
            - Ask about **pre-purchase behavior** for funnel insights
            """
        )

    return query, category if category else None


def escape_latex(text: str) -> str:
    """Escape dollar signs to prevent Streamlit LaTeX interpretation."""
    return text.replace("$", "&#36;")


def render_results(results: dict):
    """Render comparison results."""
    st.header("📈 Results")

    # Metrics row
    col1, col2, col3 = st.columns(3)

    graphrag_latency = results["graphrag"]["latency_ms"]
    naive_latency = results["naive_rag"]["latency_ms"]

    with col1:
        st.metric(
            "GraphRAG Latency",
            f"{graphrag_latency:.0f}ms",
        )
    with col2:
        st.metric(
            "Naive RAG Latency",
            f"{naive_latency:.0f}ms",
        )
    with col3:
        diff = naive_latency - graphrag_latency
        st.metric(
            "Latency Difference",
            f"{abs(diff):.0f}ms",
            delta=f"{'faster' if diff > 0 else 'slower'} GraphRAG",
            delta_color="normal" if diff > 0 else "inverse",
        )

    st.divider()

    # Escape dollar signs to prevent LaTeX interpretation
    graphrag_response = escape_latex(results["graphrag"]["response"])
    graphrag_context = escape_latex(results["graphrag"]["context"])
    naive_response = escape_latex(results["naive_rag"]["response"])
    naive_context = escape_latex(results["naive_rag"]["context"])

    # Side-by-side responses
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔷 GraphRAG Response")
        st.markdown(
            f'<div class="method-card graphrag-card">{graphrag_response}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("📄 View Retrieved Context"):
            st.markdown(
                f'<div class="context-expander">{graphrag_context}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### 🔶 Naive RAG Response")
        st.markdown(
            f'<div class="method-card naive-card">{naive_response}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("📄 View Retrieved Context"):
            st.markdown(
                f'<div class="context-expander">{naive_context}</div>',
                unsafe_allow_html=True,
            )


def render_comparison_insights():
    """Render explanation of comparison benefits."""
    st.header("💡 Why GraphRAG Wins")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### GraphRAG Advantages

            ✅ **Temporal Understanding**
            - Understands event sequences (A → B → C)
            - Can identify patterns like "users who did X then Y"

            ✅ **Structural Queries**
            - Path analysis (journey patterns)
            - Cohort comparisons with statistics

            ✅ **Specific Counts**
            - "82% of churned users..."
            - "Avg 4.2 events per session"
            """
        )

    with col2:
        st.markdown(
            """
            ### Naive RAG Limitations

            ❌ **No Temporal Order**
            - Can't distinguish "before" vs "after"
            - Loses event sequence information

            ❌ **No Aggregation**
            - Returns individual documents
            - Can't compute cohort statistics

            ❌ **Keyword Matching**
            - Semantic similarity only
            - Miss structurally relevant sessions
            """
        )


# ============================================================================
# Main App
# ============================================================================


def main():
    """Main application entry point."""
    render_header()
    render_sidebar()

    # Query section
    query, category = render_query_section()

    # Run query button
    if st.button("🚀 Compare Methods", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a query or select a preset.")
            return

        with st.spinner("Running comparison..."):
            start = time.time()
            results = query_compare(query, category)
            elapsed = time.time() - start

        if results:
            st.success(f"Comparison complete in {elapsed:.1f}s")
            render_results(results)
        else:
            st.error("Failed to get results. Is the API running?")

    # Always show comparison insights
    st.divider()
    render_comparison_insights()


if __name__ == "__main__":
    main()
