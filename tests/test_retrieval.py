"""
Unit tests for Customer Journey GraphRAG core retrieval logic.

Tests the key functions in retrieval.py and naive_rag.py to ensure
correct behavior of graph traversal and pattern extraction.
"""

import networkx as nx
import pandas as pd
import pytest

from src.naive_rag import session_to_document
from src.retrieval import (
    extract_journey_pattern,
    extract_user_journeys,
    find_common_patterns,
    get_user_context,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_graph() -> nx.DiGraph:
    """Create a minimal test graph with user, session, events, and products."""
    G = nx.DiGraph()

    # Add user node
    G.add_node(
        "user_1",
        node_type="User",
        user_id=1,
        segment="high_value",
        ltv=500.0,
        churned=False,
    )

    # Add session node
    G.add_node(
        "session_1",
        node_type="Session",
        session_id=1,
        start_time="2026-01-01T10:00:00",
    )
    G.add_edge("user_1", "session_1", edge_type="STARTED")

    # Add event nodes
    G.add_node(
        "event_1",
        node_type="Event",
        event_id=1,
        event_type="page_view",
        timestamp="2026-01-01T10:00:00",
        page_url="/home",
    )
    G.add_node(
        "event_2",
        node_type="Event",
        event_id=2,
        event_type="click",
        timestamp="2026-01-01T10:01:00",
        page_url="/products",
    )
    G.add_node(
        "event_3",
        node_type="Event",
        event_id=3,
        event_type="purchase",
        timestamp="2026-01-01T10:02:00",
        page_url="/checkout",
    )

    # Connect session to events
    G.add_edge("session_1", "event_1", edge_type="CONTAINS")
    G.add_edge("session_1", "event_2", edge_type="CONTAINS")
    G.add_edge("session_1", "event_3", edge_type="CONTAINS")

    # Connect events in sequence
    G.add_edge("event_1", "event_2", edge_type="NEXT")
    G.add_edge("event_2", "event_3", edge_type="NEXT")

    # Add product node
    G.add_node(
        "product_1",
        node_type="Product",
        product_id=1,
        name="Test Product",
        category="Electronics",
        price=99.99,
    )
    G.add_edge("event_2", "product_1", edge_type="INVOLVES")

    return G


@pytest.fixture
def churned_user_graph() -> nx.DiGraph:
    """Create a graph with a churned user for pattern testing."""
    G = nx.DiGraph()

    # Add churned user
    G.add_node(
        "user_2",
        node_type="User",
        user_id=2,
        segment="low",
        ltv=30.0,
        churned=True,
    )

    # Add session
    G.add_node(
        "session_2",
        node_type="Session",
        session_id=2,
        start_time="2026-01-01T11:00:00",
    )
    G.add_edge("user_2", "session_2", edge_type="STARTED")

    # Add short exit journey: search → exit
    G.add_node(
        "event_4",
        node_type="Event",
        event_id=4,
        event_type="search",
        timestamp="2026-01-01T11:00:00",
        page_url="/search",
    )
    G.add_node(
        "event_5",
        node_type="Event",
        event_id=5,
        event_type="exit",
        timestamp="2026-01-01T11:01:00",
        page_url="/exit",
    )

    G.add_edge("session_2", "event_4", edge_type="CONTAINS")
    G.add_edge("session_2", "event_5", edge_type="CONTAINS")
    G.add_edge("event_4", "event_5", edge_type="NEXT")

    return G


@pytest.fixture
def sample_dataframes():
    """Create sample DataFrames for naive RAG testing."""
    users_df = pd.DataFrame({
        "user_id": [1],
        "segment": ["high_value"],
        "ltv": [500.0],
        "churned": [False],
    })

    events_df = pd.DataFrame({
        "event_id": [1, 2, 3],
        "session_id": [1, 1, 1],
        "user_id": [1, 1, 1],
        "event_type": ["page_view", "click", "purchase"],
        "timestamp": ["2026-01-01T10:00:00", "2026-01-01T10:01:00", "2026-01-01T10:02:00"],
        "product_id": [None, 1, 1],
    })

    products_df = pd.DataFrame({
        "product_id": [1],
        "name": ["Test Product"],
        "category": ["Electronics"],
        "price": [99.99],
    })

    return users_df, events_df, products_df


# ============================================================================
# GraphRAG Retrieval Tests
# ============================================================================


class TestUserJourneyExtraction:
    """Tests for extract_user_journeys function."""

    def test_extracts_journey_for_existing_user(self, sample_graph):
        """Should extract journey for a valid user."""
        journeys = extract_user_journeys(sample_graph, user_id=1, max_sessions=10)

        assert len(journeys) == 1
        assert journeys[0]["session_id"] == 1
        assert journeys[0]["event_count"] == 3

    def test_returns_empty_for_nonexistent_user(self, sample_graph):
        """Should return empty list for user not in graph."""
        journeys = extract_user_journeys(sample_graph, user_id=999, max_sessions=10)

        assert journeys == []

    def test_events_sorted_by_timestamp(self, sample_graph):
        """Should return events in chronological order."""
        journeys = extract_user_journeys(sample_graph, user_id=1, max_sessions=10)
        events = journeys[0]["events"]

        assert events[0]["event_type"] == "page_view"
        assert events[1]["event_type"] == "click"
        assert events[2]["event_type"] == "purchase"


class TestUserContext:
    """Tests for get_user_context function."""

    def test_returns_user_attributes(self, sample_graph):
        """Should return correct user attributes."""
        context = get_user_context(sample_graph, user_id=1)

        assert context is not None
        assert context["segment"] == "high_value"
        assert context["ltv"] == 500.0
        assert context["churned"] is False

    def test_returns_none_for_nonexistent_user(self, sample_graph):
        """Should return None for user not in graph."""
        context = get_user_context(sample_graph, user_id=999)

        assert context is None


class TestJourneyPatternExtraction:
    """Tests for extract_journey_pattern function."""

    def test_creates_arrow_pattern(self):
        """Should create arrow-separated pattern string."""
        events = [
            {"event_type": "page_view"},
            {"event_type": "click"},
            {"event_type": "purchase"},
        ]

        pattern = extract_journey_pattern(events)

        assert pattern == "page_view → click → purchase"

    def test_empty_events_returns_empty_string(self):
        """Should return empty string for empty event list."""
        pattern = extract_journey_pattern([])

        assert pattern == ""


class TestCommonPatterns:
    """Tests for find_common_patterns function."""

    def test_finds_patterns_for_churned_users(self, churned_user_graph):
        """Should find patterns for filtered users."""
        patterns = find_common_patterns(
            churned_user_graph,
            user_filter={"churned": True},
            limit=10,
        )

        assert len(patterns) >= 1
        # Should find the "search → exit" pattern
        pattern_strings = [p[0] for p in patterns]
        assert "search → exit" in pattern_strings


# ============================================================================
# Naive RAG Tests
# ============================================================================


class TestSessionToDocument:
    """Tests for session_to_document function."""

    def test_creates_document_with_user_context(self, sample_dataframes):
        """Should include user context in document."""
        users_df, events_df, products_df = sample_dataframes
        session_events = events_df[events_df["session_id"] == 1]
        user_info = {"segment": "high_value", "ltv": 500.0, "churned": False}

        doc = session_to_document(session_events, user_info, products_df)

        assert "high_value" in doc
        assert "$500.00" in doc

    def test_includes_event_types(self, sample_dataframes):
        """Should include event types in document."""
        users_df, events_df, products_df = sample_dataframes
        session_events = events_df[events_df["session_id"] == 1]
        user_info = {"segment": "high_value", "ltv": 500.0, "churned": False}

        doc = session_to_document(session_events, user_info, products_df)

        assert "page_view" in doc
        assert "click" in doc
        assert "purchase" in doc

    def test_includes_product_category(self, sample_dataframes):
        """Should include product category when available."""
        users_df, events_df, products_df = sample_dataframes
        session_events = events_df[events_df["session_id"] == 1]
        user_info = {"segment": "high_value", "ltv": 500.0, "churned": False}

        doc = session_to_document(session_events, user_info, products_df)

        assert "Electronics" in doc


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
