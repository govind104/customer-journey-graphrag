"""
NetworkX graph construction for customer journey data.

Builds a directed temporal graph from clickstream events with nodes for
Users, Sessions, Events, and Products, connected by typed edges.
"""

import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

# ============================================================================
# Graph Construction
# ============================================================================


def build_journey_graph(
    users_df: pd.DataFrame,
    products_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> nx.DiGraph:
    """
    Construct a directed temporal graph from clickstream data.

    Creates a graph with the following structure:
    - User nodes: Contains segment, LTV, churn status
    - Session nodes: Contains session timing info
    - Event nodes: Contains event type, timestamp, page URL
    - Product nodes: Contains category, price, popularity

    Edges:
    - User -[STARTED]-> Session
    - Session -[CONTAINS]-> Event (with order attribute)
    - Event -[NEXT]-> Event (temporal sequence)
    - Event -[INVOLVES]-> Product (when product_id present)

    Args:
        users_df: DataFrame with user_id, segment, ltv, churned.
        products_df: DataFrame with product_id, category, price, popularity_score.
        events_df: DataFrame with event_id, user_id, session_id, timestamp,
                   event_type, page_url, product_id.

    Returns:
        NetworkX DiGraph with all nodes and edges constructed.
    """
    G = nx.DiGraph()

    print("Building journey graph...")

    # -------------------------------------------------------------------------
    # Add User nodes
    # -------------------------------------------------------------------------
    print("  [1/4] Adding user nodes...")
    for user in users_df.itertuples():
        G.add_node(
            f"user_{user.user_id}",
            node_type="User",
            user_id=user.user_id,
            segment=user.segment,
            ltv=user.ltv,
            churned=user.churned,
        )

    # -------------------------------------------------------------------------
    # Add Product nodes
    # -------------------------------------------------------------------------
    print("  [2/4] Adding product nodes...")
    for prod in products_df.itertuples():
        G.add_node(
            f"product_{prod.product_id}",
            node_type="Product",
            product_id=prod.product_id,
            name=prod.name,
            category=prod.category,
            price=prod.price,
            popularity_score=prod.popularity_score,
        )

    # -------------------------------------------------------------------------
    # Add Session and Event nodes with edges
    # -------------------------------------------------------------------------
    print("  [3/4] Adding session and event nodes...")

    # Group events by session for efficient processing
    session_groups = events_df.groupby("session_id")
    num_sessions = len(session_groups)

    for idx, (session_id, session_events) in enumerate(session_groups):
        # Get user for this session
        first_event = session_events.iloc[0]
        user_id = first_event["user_id"]

        # Create session node
        session_node_id = f"session_{session_id}"
        session_start = session_events["timestamp"].min()
        session_end = session_events["timestamp"].max()

        G.add_node(
            session_node_id,
            node_type="Session",
            session_id=session_id,
            start_time=str(session_start),
            end_time=str(session_end),
            event_count=len(session_events),
        )

        # Edge: User -> Session
        G.add_edge(f"user_{user_id}", session_node_id, edge_type="STARTED")

        # Sort events by timestamp for sequential ordering
        sorted_events = session_events.sort_values("timestamp")
        prev_event_node: str | None = None

        for order, event in enumerate(sorted_events.itertuples()):
            event_node_id = f"event_{event.event_id}"

            # Create event node
            G.add_node(
                event_node_id,
                node_type="Event",
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp=str(event.timestamp),
                page_url=event.page_url,
            )

            # Edge: Session -> Event (with order)
            G.add_edge(session_node_id, event_node_id, edge_type="CONTAINS", order=order)

            # Edge: Event -> Event (temporal sequence)
            if prev_event_node is not None:
                G.add_edge(prev_event_node, event_node_id, edge_type="NEXT")
            prev_event_node = event_node_id

            # Edge: Event -> Product (if product involved)
            if pd.notna(event.product_id):
                product_node_id = f"product_{int(event.product_id)}"
                if product_node_id in G:
                    G.add_edge(event_node_id, product_node_id, edge_type="INVOLVES")

        # Progress indicator
        if (idx + 1) % 5000 == 0:
            print(f"    Processed {idx + 1}/{num_sessions} sessions...")

    print("  [4/4] Graph construction complete!")

    return G


def get_graph_stats(G: nx.DiGraph) -> dict[str, Any]:
    """
    Calculate summary statistics for the journey graph.

    Args:
        G: The journey graph.

    Returns:
        Dictionary with node counts by type, edge counts by type,
        and overall graph metrics.
    """
    # Count nodes by type
    node_types: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        node_type = data.get("node_type", "Unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1

    # Count edges by type
    edge_types: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        edge_type = data.get("edge_type", "Unknown")
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types": node_types,
        "edge_types": edge_types,
    }


def print_graph_stats(G: nx.DiGraph) -> None:
    """
    Print formatted graph statistics to console.

    Args:
        G: The journey graph.
    """
    stats = get_graph_stats(G)

    print("\n" + "=" * 50)
    print("Journey Graph Statistics")
    print("=" * 50)
    print(f"Total Nodes: {stats['total_nodes']:,}")
    print(f"Total Edges: {stats['total_edges']:,}")

    print("\nNodes by Type:")
    for node_type, count in sorted(stats["node_types"].items()):
        print(f"  - {node_type}: {count:,}")

    print("\nEdges by Type:")
    for edge_type, count in sorted(stats["edge_types"].items()):
        print(f"  - {edge_type}: {count:,}")
    print("=" * 50)


# ============================================================================
# Persistence
# ============================================================================


def save_graph(G: nx.DiGraph, path: str | Path = "graph/journey_graph.pkl") -> None:
    """
    Serialize and save the graph to disk.

    Args:
        G: The journey graph to save.
        path: File path for the pickle file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(G, f)

    print(f"Graph saved to {path}")


def load_graph(path: str | Path = "graph/journey_graph.pkl") -> nx.DiGraph:
    """
    Load a previously saved graph from disk.

    Args:
        path: File path to the pickle file.

    Returns:
        The loaded NetworkX DiGraph.

    Raises:
        FileNotFoundError: If the graph file doesn't exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    with open(path, "rb") as f:
        G = pickle.load(f)

    print(f"Graph loaded from {path}")
    return G


def load_or_build_graph(
    data_dir: str | Path = "data",
    graph_path: str | Path = "graph/journey_graph.pkl",
    force_rebuild: bool = False,
) -> nx.DiGraph:
    """
    Load existing graph or build from data if not found.

    Convenience function that checks for existing graph file,
    and builds from CSVs if not present or force_rebuild is True.

    Args:
        data_dir: Directory containing CSV data files.
        graph_path: Path to save/load the graph pickle.
        force_rebuild: If True, rebuild even if graph exists.

    Returns:
        The journey graph (loaded or newly built).
    """
    graph_path = Path(graph_path)

    if graph_path.exists() and not force_rebuild:
        return load_graph(graph_path)

    # Load data and build graph
    data_dir = Path(data_dir)
    print(f"Loading data from {data_dir}...")

    users_df = pd.read_csv(data_dir / "users.csv")
    products_df = pd.read_csv(data_dir / "products.csv")
    events_df = pd.read_csv(data_dir / "events.csv")

    print(f"  Users: {len(users_df)}, Products: {len(products_df)}, Events: {len(events_df)}")

    G = build_journey_graph(users_df, products_df, events_df)
    print_graph_stats(G)
    save_graph(G, graph_path)

    return G


# ============================================================================
# CLI Entry Point
# ============================================================================


if __name__ == "__main__":
    G = load_or_build_graph(force_rebuild=True)
