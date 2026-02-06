"""
GraphRAG retrieval engine for customer journey analysis.

Implements journey-aware graph traversal and pattern extraction to provide
contextually relevant information for LLM-based product insights.
"""

from collections import Counter
from typing import Any, Literal

import networkx as nx

# ============================================================================
# User Journey Extraction
# ============================================================================


def extract_user_journeys(
    G: nx.DiGraph,
    user_id: int,
    max_sessions: int = 10,
) -> list[dict[str, Any]]:
    """
    Extract all journey paths for a specific user.

    Traverses the graph from user node through sessions to events,
    maintaining temporal ordering within each session.

    Args:
        G: The journey graph.
        user_id: The user ID to extract journeys for.
        max_sessions: Maximum number of sessions to return.

    Returns:
        List of session dictionaries, each containing:
        - session_id: Session identifier
        - events: List of event dicts with type, timestamp, products
    """
    user_node = f"user_{user_id}"

    if user_node not in G:
        return []

    journeys = []

    # Get all sessions for this user
    sessions = [
        n for n in G.successors(user_node) if G.nodes[n].get("node_type") == "Session"
    ]

    for session_node in sessions[:max_sessions]:
        session_data = G.nodes[session_node]

        # Get all events in this session
        events = []
        for event_node in G.successors(session_node):
            if G.nodes[event_node].get("node_type") != "Event":
                continue

            event_data = G.nodes[event_node]

            # Find connected products
            products = []
            for product_node in G.successors(event_node):
                if G.nodes[product_node].get("node_type") == "Product":
                    products.append(G.nodes[product_node])

            events.append(
                {
                    "event_id": event_data.get("event_id"),
                    "event_type": event_data.get("event_type"),
                    "timestamp": event_data.get("timestamp"),
                    "page_url": event_data.get("page_url"),
                    "products": products,
                }
            )

        # Sort events by timestamp
        events.sort(key=lambda x: x["timestamp"])

        journeys.append(
            {
                "session_id": session_data.get("session_id"),
                "start_time": session_data.get("start_time"),
                "event_count": len(events),
                "events": events,
            }
        )

    return journeys


def get_user_context(G: nx.DiGraph, user_id: int) -> dict[str, Any] | None:
    """
    Get user attributes from the graph.

    Args:
        G: The journey graph.
        user_id: The user ID to look up.

    Returns:
        Dictionary with user attributes (segment, ltv, churned) or None if not found.
    """
    user_node = f"user_{user_id}"

    if user_node not in G:
        return None

    data = G.nodes[user_node]
    return {
        "user_id": user_id,
        "segment": data.get("segment"),
        "ltv": data.get("ltv"),
        "churned": data.get("churned"),
    }


# ============================================================================
# Path Pattern Analysis
# ============================================================================


def extract_journey_pattern(events: list[dict[str, Any]]) -> str:
    """
    Convert event sequence to a pattern string for aggregation.

    Args:
        events: List of event dictionaries.

    Returns:
        Pattern string like "page_view → search → click → add_to_cart → purchase".
    """
    return " → ".join(e["event_type"] for e in events)


def find_sessions_by_outcome(
    G: nx.DiGraph,
    outcome: Literal["purchase", "exit", "add_to_cart"],
    limit: int = 100,
) -> list[str]:
    """
    Find session nodes that end with a specific event type.

    Args:
        G: The journey graph.
        outcome: The event type to filter by (purchase, exit, add_to_cart).
        limit: Maximum number of sessions to return.

    Returns:
        List of session node IDs ending with the specified outcome.
    """
    matching_sessions = []

    # Find all event nodes with the target type
    for node, data in G.nodes(data=True):
        if data.get("node_type") != "Event":
            continue
        if data.get("event_type") != outcome:
            continue

        # Check if this is the last event in its session (no NEXT edge)
        has_next = any(
            G.edges[node, succ].get("edge_type") == "NEXT" for succ in G.successors(node)
        )

        if not has_next:
            # Find the session this event belongs to
            for pred in G.predecessors(node):
                if G.nodes[pred].get("node_type") == "Session":
                    matching_sessions.append(pred)
                    break

        if len(matching_sessions) >= limit:
            break

    return matching_sessions


def find_conversion_paths(G: nx.DiGraph, limit: int = 50) -> list[dict[str, Any]]:
    """
    Find journey paths that end in a purchase.

    Args:
        G: The journey graph.
        limit: Maximum number of paths to return.

    Returns:
        List of journey dictionaries for sessions ending in purchase.
    """
    purchase_sessions = find_sessions_by_outcome(G, "purchase", limit)
    paths = []

    for session_node in purchase_sessions:
        session_data = G.nodes[session_node]
        session_id = session_data.get("session_id")

        # Get user for this session
        user_node = None
        for pred in G.predecessors(session_node):
            if G.nodes[pred].get("node_type") == "User":
                user_node = pred
                break

        if user_node:
            user_id = G.nodes[user_node].get("user_id")
            journeys = extract_user_journeys(G, user_id, max_sessions=1)
            # Find the specific session
            for j in journeys:
                if j["session_id"] == session_id:
                    paths.append(
                        {
                            "user_id": user_id,
                            "user_context": get_user_context(G, user_id),
                            **j,
                        }
                    )
                    break

    return paths


def find_churn_paths(G: nx.DiGraph, limit: int = 50) -> list[dict[str, Any]]:
    """
    Find journey paths that end in exit without purchase.

    Args:
        G: The journey graph.
        limit: Maximum number of paths to return.

    Returns:
        List of journey dictionaries for sessions ending in exit.
    """
    exit_sessions = find_sessions_by_outcome(G, "exit", limit)
    paths = []

    for session_node in exit_sessions:
        session_data = G.nodes[session_node]
        session_id = session_data.get("session_id")

        # Get user for this session
        user_node = None
        for pred in G.predecessors(session_node):
            if G.nodes[pred].get("node_type") == "User":
                user_node = pred
                break

        if user_node:
            user_id = G.nodes[user_node].get("user_id")
            user_context = get_user_context(G, user_id)

            # Only include churned users
            if user_context and user_context.get("churned"):
                journeys = extract_user_journeys(G, user_id, max_sessions=1)
                for j in journeys:
                    if j["session_id"] == session_id:
                        paths.append(
                            {
                                "user_id": user_id,
                                "user_context": user_context,
                                **j,
                            }
                        )
                        break

    return paths


def find_common_patterns(
    G: nx.DiGraph,
    user_filter: dict[str, Any] | None = None,
    limit: int = 100,
) -> list[tuple[str, int, float]]:
    """
    Find common journey patterns for a filtered set of users.

    Aggregates event sequences into pattern strings and counts occurrences.

    Args:
        G: The journey graph.
        user_filter: Optional dict with filters like {"churned": True, "segment": "high_value"}.
        limit: Maximum number of sessions to analyze.

    Returns:
        List of tuples: (pattern_string, count, percentage).
    """
    pattern_counts: Counter[str] = Counter()
    sessions_analyzed = 0

    # Iterate through all users
    for node, data in G.nodes(data=True):
        if data.get("node_type") != "User":
            continue

        # Apply filters
        if user_filter:
            match = True
            for key, value in user_filter.items():
                if data.get(key) != value:
                    match = False
                    break
            if not match:
                continue

        user_id = data.get("user_id")
        journeys = extract_user_journeys(G, user_id, max_sessions=5)

        for journey in journeys:
            if sessions_analyzed >= limit:
                break

            pattern = extract_journey_pattern(journey["events"])
            pattern_counts[pattern] += 1
            sessions_analyzed += 1

        if sessions_analyzed >= limit:
            break

    # Calculate percentages
    total = sum(pattern_counts.values())
    results = [
        (pattern, count, count / total * 100 if total > 0 else 0)
        for pattern, count in pattern_counts.most_common(10)
    ]

    return results


# ============================================================================
# Cohort Comparison
# ============================================================================


def compare_cohorts(
    G: nx.DiGraph,
    cohort_a_filter: dict[str, Any],
    cohort_b_filter: dict[str, Any],
    cohort_a_name: str = "Cohort A",
    cohort_b_name: str = "Cohort B",
    sample_size: int = 50,
) -> dict[str, Any]:
    """
    Compare journey patterns between two user cohorts.

    Extracts and compares metrics like average events per session,
    conversion rates, and common patterns for two different user groups.

    Args:
        G: The journey graph.
        cohort_a_filter: Filter dict for first cohort (e.g., {"segment": "high_value"}).
        cohort_b_filter: Filter dict for second cohort.
        cohort_a_name: Display name for first cohort.
        cohort_b_name: Display name for second cohort.
        sample_size: Number of sessions to analyze per cohort.

    Returns:
        Dictionary with comparison metrics for both cohorts.
    """

    def analyze_cohort(
        user_filter: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze a single cohort."""
        users = []
        sessions = []
        event_counts = []
        purchase_count = 0
        cart_add_count = 0

        for node, data in G.nodes(data=True):
            if data.get("node_type") != "User":
                continue

            # Apply filter
            match = True
            for key, value in user_filter.items():
                if data.get(key) != value:
                    match = False
                    break
            if not match:
                continue

            users.append(data)
            user_id = data.get("user_id")
            journeys = extract_user_journeys(G, user_id, max_sessions=3)

            for journey in journeys:
                sessions.append(journey)
                event_counts.append(journey["event_count"])

                # Check for purchase/cart events
                for event in journey["events"]:
                    if event["event_type"] == "purchase":
                        purchase_count += 1
                    elif event["event_type"] == "add_to_cart":
                        cart_add_count += 1

            if len(sessions) >= sample_size:
                break

        avg_events = sum(event_counts) / len(event_counts) if event_counts else 0
        avg_ltv = sum(u["ltv"] for u in users) / len(users) if users else 0

        return {
            "user_count": len(users),
            "session_count": len(sessions),
            "avg_events_per_session": round(avg_events, 2),
            "avg_ltv": round(avg_ltv, 2),
            "purchase_events": purchase_count,
            "cart_add_events": cart_add_count,
            "conversion_rate": round(
                purchase_count / len(sessions) * 100 if sessions else 0, 1
            ),
        }

    cohort_a_stats = analyze_cohort(cohort_a_filter)
    cohort_b_stats = analyze_cohort(cohort_b_filter)

    return {
        cohort_a_name: cohort_a_stats,
        cohort_b_name: cohort_b_stats,
        "comparison": {
            "events_diff": round(
                cohort_a_stats["avg_events_per_session"]
                - cohort_b_stats["avg_events_per_session"],
                2,
            ),
            "ltv_diff": round(cohort_a_stats["avg_ltv"] - cohort_b_stats["avg_ltv"], 2),
            "conversion_diff": round(
                cohort_a_stats["conversion_rate"] - cohort_b_stats["conversion_rate"], 1
            ),
        },
    }


# ============================================================================
# Product-Centric Queries
# ============================================================================


def find_products_before_purchase(
    G: nx.DiGraph,
    category_filter: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Find which products users view before making a purchase.

    Walks backward from purchase events to find preceding product views.

    Args:
        G: The journey graph.
        category_filter: Optional category to filter purchase events.
        limit: Maximum number of purchase events to analyze.

    Returns:
        Dictionary with product categories and counts viewed before purchase.
    """
    categories_viewed: Counter[str] = Counter()
    products_viewed: Counter[str] = Counter()
    total_purchases = 0

    for node, data in G.nodes(data=True):
        if data.get("node_type") != "Event":
            continue
        if data.get("event_type") != "purchase":
            continue

        # Check category filter if specified
        if category_filter:
            # Get the purchased product category
            purchased_product = None
            for succ in G.successors(node):
                if G.nodes[succ].get("node_type") == "Product":
                    purchased_product = G.nodes[succ]
                    break

            if not purchased_product or purchased_product.get("category") != category_filter:
                continue

        total_purchases += 1

        # Walk backward through the session to find viewed products
        current_node = node
        for _ in range(20):  # Max 20 steps back
            # Find predecessor event
            prev_event = None
            for pred in G.predecessors(current_node):
                edge_data = G.edges[pred, current_node]
                if edge_data.get("edge_type") == "NEXT":
                    prev_event = pred
                    break

            if not prev_event:
                break

            # Check if this event viewed a product
            event_data = G.nodes[prev_event]
            if event_data.get("event_type") in ["page_view", "click"]:
                for succ in G.successors(prev_event):
                    if G.nodes[succ].get("node_type") == "Product":
                        prod_data = G.nodes[succ]
                        categories_viewed[prod_data.get("category", "Unknown")] += 1
                        products_viewed[prod_data.get("name", "Unknown")] += 1

            current_node = prev_event

        if total_purchases >= limit:
            break

    return {
        "total_purchases_analyzed": total_purchases,
        "categories_viewed_before_purchase": dict(categories_viewed.most_common(10)),
        "top_products_viewed": dict(products_viewed.most_common(10)),
    }


def find_exit_points_after_category(
    G: nx.DiGraph,
    category: str,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Find where users exit after viewing products in a specific category.

    Args:
        G: The journey graph.
        category: Product category to analyze.
        limit: Maximum number of exit events to analyze.

    Returns:
        Dictionary with exit patterns and statistics.
    """
    exit_after_category = 0
    exit_patterns: Counter[str] = Counter()
    events_before_exit: list[int] = []

    for node, data in G.nodes(data=True):
        if data.get("node_type") != "Event":
            continue
        if data.get("event_type") != "exit":
            continue

        # Walk backward to check if user viewed the category
        viewed_category = False
        events_count = 0
        current_node = node

        for _ in range(20):
            prev_event = None
            for pred in G.predecessors(current_node):
                edge_data = G.edges[pred, current_node]
                if edge_data.get("edge_type") == "NEXT":
                    prev_event = pred
                    break

            if not prev_event:
                break

            events_count += 1
            event_data = G.nodes[prev_event]

            # Check if product in this category was viewed
            for succ in G.successors(prev_event):
                if G.nodes[succ].get("node_type") == "Product":
                    if G.nodes[succ].get("category") == category:
                        viewed_category = True
                        exit_patterns[event_data.get("event_type", "unknown")] += 1
                        break

            current_node = prev_event

        if viewed_category:
            exit_after_category += 1
            events_before_exit.append(events_count)

        if exit_after_category >= limit:
            break

    avg_events_before_exit = (
        sum(events_before_exit) / len(events_before_exit) if events_before_exit else 0
    )

    return {
        "category": category,
        "exits_after_viewing": exit_after_category,
        "avg_events_before_exit": round(avg_events_before_exit, 2),
        "last_event_before_exit": dict(exit_patterns.most_common(5)),
    }


# ============================================================================
# Context Serialization for LLM
# ============================================================================


def serialize_journey_to_text(journey: dict[str, Any]) -> str:
    """
    Convert a single journey to human-readable text.

    Args:
        journey: Journey dictionary with events and user context.

    Returns:
        Formatted string describing the journey.
    """
    parts = []

    # User context if available
    if "user_context" in journey and journey["user_context"]:
        ctx = journey["user_context"]
        parts.append(
            f"User (segment: {ctx['segment']}, LTV: ${ctx['ltv']:.2f}, churned: {ctx['churned']})"
        )

    # Journey path
    path_parts = []
    for event in journey.get("events", []):
        event_desc = event["event_type"]
        if event.get("products"):
            prod = event["products"][0]
            event_desc += f" ({prod.get('category', 'Unknown')} - ${prod.get('price', 0):.2f})"
        path_parts.append(event_desc)

    if path_parts:
        parts.append("Journey: " + " → ".join(path_parts))

    return "\n".join(parts)


def serialize_journeys_for_llm(
    journeys: list[dict[str, Any]],
    max_journeys: int = 10,
    include_stats: bool = True,
) -> str:
    """
    Convert multiple journeys to LLM-friendly context text.

    Creates a formatted string with journey paths and optional statistics
    suitable for inclusion in LLM prompts.

    Args:
        journeys: List of journey dictionaries.
        max_journeys: Maximum number of journeys to include.
        include_stats: Whether to include summary statistics.

    Returns:
        Formatted string with journey context.
    """
    if not journeys:
        return "No journeys found matching the criteria."

    lines = ["## Customer Journey Context\n"]

    # Add journeys
    lines.append("### Sample Journeys:")
    for i, journey in enumerate(journeys[:max_journeys], 1):
        lines.append(f"\n**Journey {i}:**")
        lines.append(serialize_journey_to_text(journey))

    # Add statistics if requested
    if include_stats and len(journeys) > 1:
        event_counts = [len(j.get("events", [])) for j in journeys]
        avg_events = sum(event_counts) / len(event_counts)

        # Count event types
        event_types: Counter[str] = Counter()
        for j in journeys:
            for e in j.get("events", []):
                event_types[e["event_type"]] += 1

        lines.append("\n### Statistics:")
        lines.append(f"- Total journeys analyzed: {len(journeys)}")
        lines.append(f"- Average events per session: {avg_events:.1f}")
        lines.append("- Event distribution: " + ", ".join(
            f"{k}: {v}" for k, v in event_types.most_common(5)
        ))

    return "\n".join(lines)


def serialize_patterns_for_llm(
    patterns: list[tuple[str, int, float]],
    context_description: str = "journey patterns",
) -> str:
    """
    Convert pattern analysis results to LLM-friendly text.

    Args:
        patterns: List of (pattern_string, count, percentage) tuples.
        context_description: Description of what patterns represent.

    Returns:
        Formatted string describing the patterns.
    """
    if not patterns:
        return f"No {context_description} found."

    lines = [f"## Common {context_description.title()}\n"]

    for i, (pattern, count, pct) in enumerate(patterns, 1):
        lines.append(f"{i}. **{pattern}** - {count} occurrences ({pct:.1f}%)")

    return "\n".join(lines)


def serialize_comparison_for_llm(comparison: dict[str, Any]) -> str:
    """
    Convert cohort comparison to LLM-friendly text.

    Args:
        comparison: Comparison dictionary from compare_cohorts().

    Returns:
        Formatted string with comparison data.
    """
    lines = ["## Cohort Comparison\n"]

    for key, value in comparison.items():
        if key == "comparison":
            lines.append("\n### Key Differences:")
            for metric, diff in value.items():
                direction = "higher" if diff > 0 else "lower"
                lines.append(f"- {metric}: {abs(diff)} {direction}")
        elif isinstance(value, dict):
            lines.append(f"\n### {key}:")
            for metric, stat in value.items():
                lines.append(f"- {metric}: {stat}")

    return "\n".join(lines)


# ============================================================================
# High-Level Query Functions
# ============================================================================


def query_churned_user_journeys(G: nx.DiGraph, sample_size: int = 20) -> str:
    """
    Analyze and serialize journeys of churned users.

    Args:
        G: The journey graph.
        sample_size: Number of journeys to analyze.

    Returns:
        LLM-ready context string about churned user behavior.
    """
    paths = find_churn_paths(G, limit=sample_size)
    patterns = find_common_patterns(G, {"churned": True}, limit=100)

    journey_text = serialize_journeys_for_llm(paths, max_journeys=5)
    pattern_text = serialize_patterns_for_llm(patterns, "churned user journey patterns")

    return f"{journey_text}\n\n{pattern_text}"


def query_high_vs_low_ltv(G: nx.DiGraph) -> str:
    """
    Compare high-value and low-value user journeys.

    Args:
        G: The journey graph.

    Returns:
        LLM-ready context string comparing cohorts.
    """
    comparison = compare_cohorts(
        G,
        cohort_a_filter={"segment": "high_value"},
        cohort_b_filter={"segment": "low"},
        cohort_a_name="High-Value Users",
        cohort_b_name="Low-Value Users",
        sample_size=50,
    )

    high_patterns = find_common_patterns(G, {"segment": "high_value"}, limit=50)
    low_patterns = find_common_patterns(G, {"segment": "low"}, limit=50)

    comparison_text = serialize_comparison_for_llm(comparison)
    high_pattern_text = serialize_patterns_for_llm(
        high_patterns, "high-value user patterns"
    )
    low_pattern_text = serialize_patterns_for_llm(low_patterns, "low-value user patterns")

    return f"{comparison_text}\n\n{high_pattern_text}\n\n{low_pattern_text}"


def query_pre_purchase_behavior(G: nx.DiGraph, category: str | None = None) -> str:
    """
    Analyze what users do before making a purchase.

    Args:
        G: The journey graph.
        category: Optional category to filter purchases.

    Returns:
        LLM-ready context string about pre-purchase behavior.
    """
    pre_purchase = find_products_before_purchase(G, category, limit=50)
    conversion_paths = find_conversion_paths(G, limit=20)

    lines = ["## Pre-Purchase Behavior Analysis\n"]

    if category:
        lines.append(f"**Filtered by category: {category}**\n")

    lines.append(f"Purchases analyzed: {pre_purchase['total_purchases_analyzed']}")
    lines.append("\n### Categories viewed before purchase:")
    for cat, count in pre_purchase["categories_viewed_before_purchase"].items():
        lines.append(f"- {cat}: {count} views")

    lines.append("\n### Sample conversion journeys:")

    journey_text = serialize_journeys_for_llm(conversion_paths, max_journeys=5)

    return "\n".join(lines) + "\n\n" + journey_text


def query_category_exit_analysis(G: nx.DiGraph, category: str) -> str:
    """
    Analyze why users exit after viewing a specific category.

    Args:
        G: The journey graph.
        category: Product category to analyze.

    Returns:
        LLM-ready context string about exit patterns.
    """
    exit_data = find_exit_points_after_category(G, category, limit=50)

    lines = [f"## Exit Analysis for {category} Category\n"]
    lines.append(f"- Users who exited after viewing {category}: {exit_data['exits_after_viewing']}")
    lines.append(
        f"- Average events before exit: {exit_data['avg_events_before_exit']}"
    )

    lines.append("\n### Last event type before exit:")
    for event_type, count in exit_data["last_event_before_exit"].items():
        lines.append(f"- {event_type}: {count}")

    return "\n".join(lines)
