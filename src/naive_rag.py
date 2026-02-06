"""
Naive vector RAG baseline for comparison with GraphRAG.

Implements a simple text-chunk based retrieval using sentence-transformers
and FAISS, treating sessions as documents without graph structure awareness.
"""

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ============================================================================
# Document Generation
# ============================================================================


def session_to_document(
    session_events: pd.DataFrame,
    user_info: dict[str, Any],
    products_df: pd.DataFrame,
) -> str:
    """
    Convert a session's events to a text document.

    Creates a natural language description of user actions suitable
    for embedding-based retrieval.

    Args:
        session_events: DataFrame of events for a single session.
        user_info: Dictionary with user attributes (segment, ltv, churned).
        products_df: Product catalog for enriching event descriptions.

    Returns:
        Text document describing the session.
    """
    # Create product lookup
    product_lookup = products_df.set_index("product_id").to_dict("index")

    # Build document parts
    parts = []

    # User context
    parts.append(
        f"User (segment: {user_info['segment']}, "
        f"LTV: ${user_info['ltv']:.2f}, "
        f"churned: {user_info['churned']})"
    )

    # Session actions
    actions = []
    for event in session_events.sort_values("timestamp").itertuples():
        event_desc = event.event_type

        if pd.notna(event.product_id):
            prod = product_lookup.get(int(event.product_id), {})
            if prod:
                event_desc += f" {prod.get('category', 'Unknown')}"
                event_desc += f" (${prod.get('price', 0):.2f})"

        actions.append(event_desc)

    parts.append("Actions: " + ", ".join(actions))

    return " | ".join(parts)


def generate_documents(
    users_df: pd.DataFrame,
    events_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Generate text documents from all sessions.

    Each session becomes one document containing user context
    and a description of all events.

    Args:
        users_df: User DataFrame.
        events_df: Events DataFrame.
        products_df: Products DataFrame.

    Returns:
        List of document dictionaries with 'text', 'session_id', 'user_id'.
    """
    # Create user lookup
    user_lookup = users_df.set_index("user_id").to_dict("index")

    documents = []
    session_groups = events_df.groupby("session_id")

    print(f"Generating documents for {len(session_groups)} sessions...")

    for session_id, session_events in session_groups:
        user_id = session_events.iloc[0]["user_id"]
        user_info = user_lookup.get(user_id, {})

        if not user_info:
            continue

        doc_text = session_to_document(session_events, user_info, products_df)

        documents.append(
            {
                "text": doc_text,
                "session_id": session_id,
                "user_id": user_id,
                "segment": user_info.get("segment"),
                "churned": user_info.get("churned"),
            }
        )

    print(f"  ✓ Generated {len(documents)} documents")
    return documents


# ============================================================================
# Vector Index
# ============================================================================


class NaiveVectorRAG:
    """
    Simple vector RAG system using sentence-transformers and FAISS.

    Provides basic semantic search over session documents without
    understanding of temporal or graph structure.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "graph/naive_rag_index.pkl",
    ):
        """
        Initialize the vector RAG system.

        Args:
            model_name: Sentence transformer model to use.
            index_path: Path to save/load the index.
        """
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.model: SentenceTransformer | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[dict[str, Any]] = []
        self._initialized = False

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("  ✓ Model loaded")

    def build_index(
        self,
        users_df: pd.DataFrame,
        events_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> None:
        """
        Build the vector index from data.

        Generates documents from sessions, embeds them, and creates
        a FAISS index for efficient similarity search.

        Args:
            users_df: User DataFrame.
            events_df: Events DataFrame.
            products_df: Products DataFrame.
        """
        self._load_model()

        # Generate documents
        self.documents = generate_documents(users_df, events_df, products_df)

        # Embed documents
        print("Embedding documents...")
        texts = [doc["text"] for doc in self.documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"  ✓ Indexed {self.index.ntotal} documents")
        self._initialized = True

        # Save index
        self.save()

    def save(self) -> None:
        """Save index and documents to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "documents": self.documents,
            "index": faiss.serialize_index(self.index) if self.index else None,
        }

        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Index saved to {self.index_path}")

    def load(self) -> bool:
        """
        Load index and documents from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.index_path.exists():
            return False

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        if data["index"] is not None:
            self.index = faiss.deserialize_index(data["index"])

        self._initialized = True
        print(f"Index loaded from {self.index_path}")
        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search for documents similar to the query.

        Args:
            query: Natural language query.
            top_k: Number of results to return.

        Returns:
            List of matching documents with similarity scores.
        """
        if not self._initialized:
            raise RuntimeError("Index not initialized. Call build_index() or load() first.")

        self._load_model()

        # Embed query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(score)
                results.append(doc)

        return results

    def retrieve_context(self, query: str, top_k: int = 10) -> str:
        """
        Retrieve context for LLM from matching documents.

        Args:
            query: Natural language query.
            top_k: Number of documents to retrieve.

        Returns:
            Formatted context string for LLM prompts.
        """
        results = self.search(query, top_k)

        if not results:
            return "No relevant sessions found."

        lines = ["## Retrieved Session Context (Vector Search)\n"]

        for i, doc in enumerate(results, 1):
            lines.append(f"**Session {i}** (similarity: {doc['score']:.3f}):")
            lines.append(f"  {doc['text']}\n")

        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================


def build_naive_rag(
    data_dir: str | Path = "data",
    index_path: str = "graph/naive_rag_index.pkl",
) -> NaiveVectorRAG:
    """
    Build naive RAG index from data files.

    Args:
        data_dir: Directory containing CSV files.
        index_path: Path to save the index.

    Returns:
        Initialized NaiveVectorRAG instance.
    """
    data_dir = Path(data_dir)

    users_df = pd.read_csv(data_dir / "users.csv")
    events_df = pd.read_csv(data_dir / "events.csv")
    products_df = pd.read_csv(data_dir / "products.csv")

    rag = NaiveVectorRAG(index_path=index_path)
    rag.build_index(users_df, events_df, products_df)

    return rag


def load_or_build_naive_rag(
    data_dir: str | Path = "data",
    index_path: str = "graph/naive_rag_index.pkl",
    force_rebuild: bool = False,
) -> NaiveVectorRAG:
    """
    Load existing naive RAG or build if not found.

    Args:
        data_dir: Directory containing CSV files.
        index_path: Path to the index file.
        force_rebuild: If True, rebuild even if index exists.

    Returns:
        Initialized NaiveVectorRAG instance.
    """
    rag = NaiveVectorRAG(index_path=index_path)

    if not force_rebuild and rag.load():
        return rag

    return build_naive_rag(data_dir, index_path)


# ============================================================================
# CLI Entry Point
# ============================================================================


if __name__ == "__main__":
    rag = load_or_build_naive_rag(force_rebuild=True)

    # Test query
    test_query = "What do churned users typically do?"
    print(f"\nTest query: {test_query}")
    print("-" * 50)
    context = rag.retrieve_context(test_query, top_k=5)
    print(context)
