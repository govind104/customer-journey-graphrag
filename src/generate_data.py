"""
Synthetic e-commerce clickstream data generator.

Generates realistic user journey patterns including conversion paths,
churn journeys, and browse behaviors for different user segments.
"""

import random
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

# Initialize generators with seed for reproducibility
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================================
# Configuration
# ============================================================================

NUM_USERS = 5000
NUM_PRODUCTS = 800
NUM_SESSIONS = 20000
TARGET_EVENTS = 100000  # Target ~100k events

CATEGORIES = ["Electronics", "Fashion", "Home", "Books", "Sports", "Beauty"]

SEGMENTS = {
    "high_value": {
        "ratio": 0.15,
        "ltv_mean": 500,
        "ltv_std": 150,
        "churn_rate": 0.10,
        "events_range": (5, 12),
        "purchase_prob": 0.60,
    },
    "medium": {
        "ratio": 0.50,
        "ltv_mean": 150,
        "ltv_std": 50,
        "churn_rate": 0.30,
        "events_range": (3, 7),
        "purchase_prob": 0.30,
    },
    "low": {
        "ratio": 0.35,
        "ltv_mean": 30,
        "ltv_std": 15,
        "churn_rate": 0.60,
        "events_range": (1, 4),
        "purchase_prob": 0.10,
    },
}


# ============================================================================
# Data Generation Functions
# ============================================================================


def generate_products(num_products: int = NUM_PRODUCTS) -> pd.DataFrame:
    """
    Generate synthetic product catalog.

    Args:
        num_products: Number of products to generate.

    Returns:
        DataFrame with columns: product_id, name, category, price, popularity_score.
    """
    products = []
    for i in range(num_products):
        category = random.choice(CATEGORIES)
        # Price follows log-normal distribution (realistic price spread)
        price = round(np.random.lognormal(mean=4, sigma=1.2), 2)
        price = max(5.0, min(price, 2000.0))  # Clamp to realistic range

        products.append(
            {
                "product_id": i,
                "name": f"{category}_{i}",
                "category": category,
                "price": price,
                "popularity_score": round(np.random.beta(2, 5), 4),
            }
        )

    return pd.DataFrame(products)


def generate_users(num_users: int = NUM_USERS) -> pd.DataFrame:
    """
    Generate synthetic user base with segmentation.

    Creates users with realistic segment distribution, LTV values,
    and churn flags based on segment characteristics.

    Args:
        num_users: Number of users to generate.

    Returns:
        DataFrame with columns: user_id, registration_date, segment, ltv, churned.
    """
    users = []
    segment_names = list(SEGMENTS.keys())
    segment_weights = [SEGMENTS[s]["ratio"] for s in segment_names]

    for i in range(num_users):
        segment = random.choices(segment_names, weights=segment_weights)[0]
        seg_config = SEGMENTS[segment]

        # LTV with segment-specific distribution
        ltv = max(0, np.random.normal(seg_config["ltv_mean"], seg_config["ltv_std"]))

        # Churn probability based on segment
        churned = random.random() < seg_config["churn_rate"]

        users.append(
            {
                "user_id": i,
                "registration_date": fake.date_between(start_date="-2y", end_date="-1m"),
                "segment": segment,
                "ltv": round(ltv, 2),
                "churned": churned,
            }
        )

    return pd.DataFrame(users)


def generate_session_events(
    session_id: int,
    user_id: int,
    segment: str,
    churned: bool,
    products_df: pd.DataFrame,
    start_event_id: int,
) -> tuple[list[dict[str, Any]], int]:
    """
    Generate events for a single session with realistic journey patterns.

    Creates event sequences that model real user behavior:
    - High-value users: more events, higher purchase probability
    - Churned users: rarely complete purchases
    - Journey patterns: browse → search → view → cart → purchase (or exit)

    Args:
        session_id: Unique session identifier.
        user_id: User who owns this session.
        segment: User segment (high_value, medium, low).
        churned: Whether user has churned.
        products_df: Product catalog for sampling.
        start_event_id: Starting event ID for this session.

    Returns:
        Tuple of (list of event dicts, next available event_id).
    """
    seg_config = SEGMENTS[segment]
    events = []
    event_id = start_event_id

    # Session characteristics based on segment
    min_events, max_events = seg_config["events_range"]
    num_events = random.randint(min_events, max_events)
    purchase_prob = seg_config["purchase_prob"]

    if churned:
        purchase_prob *= 0.1  # Churned users rarely purchase

    session_start = fake.date_time_between(start_date="-6m", end_date="now")
    current_time = session_start

    viewed_products: list[int] = []
    cart_products: list[int] = []

    # First event: home or search
    first_event_type = random.choice(["page_view", "search"])
    events.append(
        {
            "event_id": event_id,
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": current_time,
            "event_type": first_event_type,
            "page_url": "home" if first_event_type == "page_view" else "search",
            "product_id": None,
        }
    )
    event_id += 1
    current_time += timedelta(seconds=random.randint(5, 30))

    # Generate remaining events
    session_ended = False
    for _ in range(num_events - 1):
        if session_ended:
            break

        # Event type probabilities depend on session state
        if cart_products:
            # More likely to checkout if items in cart
            event_weights = [0.3, 0.2, 0.1, 0.2, 0.2]  # view, click, cart, exit, checkout_view
        else:
            event_weights = [0.45, 0.30, 0.15, 0.10, 0.0]  # view, click, cart, exit, checkout

        event_type = random.choices(
            ["page_view", "click", "add_to_cart", "exit", "checkout_view"],
            weights=event_weights,
        )[0]

        if event_type in ["page_view", "click"]:
            # Sample product weighted by popularity
            product = products_df.sample(1, weights=products_df["popularity_score"]).iloc[0]
            viewed_products.append(int(product["product_id"]))

            events.append(
                {
                    "event_id": event_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": current_time,
                    "event_type": event_type,
                    "page_url": f"product/{product['product_id']}",
                    "product_id": int(product["product_id"]),
                }
            )

        elif event_type == "add_to_cart" and viewed_products:
            product_id = random.choice(viewed_products)
            cart_products.append(product_id)

            events.append(
                {
                    "event_id": event_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": current_time,
                    "event_type": "add_to_cart",
                    "page_url": "cart",
                    "product_id": product_id,
                }
            )

        elif event_type == "exit":
            events.append(
                {
                    "event_id": event_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": current_time,
                    "event_type": "exit",
                    "page_url": "exit",
                    "product_id": None,
                }
            )
            session_ended = True

        elif event_type == "checkout_view" and cart_products:
            events.append(
                {
                    "event_id": event_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": current_time,
                    "event_type": "page_view",
                    "page_url": "checkout",
                    "product_id": None,
                }
            )

        event_id += 1
        current_time += timedelta(seconds=random.randint(10, 60))

    # Decide on purchase (only if cart has items and session didn't end)
    if cart_products and not session_ended and random.random() < purchase_prob:
        # Purchase primary cart item
        events.append(
            {
                "event_id": event_id,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": current_time,
                "event_type": "purchase",
                "page_url": "checkout/success",
                "product_id": cart_products[0],
            }
        )
        event_id += 1

    return events, event_id


def generate_events(
    users_df: pd.DataFrame,
    products_df: pd.DataFrame,
    num_sessions: int = NUM_SESSIONS,
) -> pd.DataFrame:
    """
    Generate all session events across users.

    Distributes sessions across users with segment-appropriate frequency,
    then generates realistic event sequences for each session.

    Args:
        users_df: User DataFrame with segment and churn info.
        products_df: Product catalog for event generation.
        num_sessions: Target number of sessions to generate.

    Returns:
        DataFrame with columns: event_id, user_id, session_id, timestamp,
        event_type, page_url, product_id.
    """
    all_events: list[dict[str, Any]] = []
    event_id = 0

    for session_id in range(num_sessions):
        # Sample user (high-value users have more sessions)
        segment_weights = users_df["segment"].map(
            {"high_value": 3.0, "medium": 1.5, "low": 1.0}
        )
        user_row = users_df.sample(1, weights=segment_weights).iloc[0]

        user_id = int(user_row["user_id"])
        segment = user_row["segment"]
        churned = user_row["churned"]

        session_events, event_id = generate_session_events(
            session_id=session_id,
            user_id=user_id,
            segment=segment,
            churned=churned,
            products_df=products_df,
            start_event_id=event_id,
        )

        all_events.extend(session_events)

        # Progress indicator
        if (session_id + 1) % 5000 == 0:
            print(f"  Generated {session_id + 1}/{num_sessions} sessions...")

    return pd.DataFrame(all_events)


def generate_all(output_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    """
    Generate complete synthetic dataset and save to CSV files.

    Creates users, products, and events CSVs with realistic e-commerce
    clickstream patterns suitable for journey graph construction.

    Args:
        output_dir: Directory to save CSV files (created if not exists).

    Returns:
        Dictionary with 'users', 'products', 'events' DataFrames.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Synthetic E-commerce Clickstream Data")
    print("=" * 60)

    # Generate products
    print("\n[1/3] Generating products...")
    products_df = generate_products()
    print(f"  ✓ Created {len(products_df)} products across {len(CATEGORIES)} categories")

    # Generate users
    print("\n[2/3] Generating users...")
    users_df = generate_users()
    segment_counts = users_df["segment"].value_counts()
    print(f"  ✓ Created {len(users_df)} users:")
    for seg, count in segment_counts.items():
        print(f"    - {seg}: {count} ({count/len(users_df)*100:.1f}%)")
    print(f"  ✓ Churn rate: {users_df['churned'].mean()*100:.1f}%")

    # Generate events
    print("\n[3/3] Generating events...")
    events_df = generate_events(users_df, products_df)
    print(f"  ✓ Created {len(events_df)} events across {NUM_SESSIONS} sessions")

    # Event type distribution
    event_counts = events_df["event_type"].value_counts()
    print("\n  Event distribution:")
    for event_type, count in event_counts.items():
        print(f"    - {event_type}: {count} ({count/len(events_df)*100:.1f}%)")

    # Calculate metrics
    conversion_sessions = events_df[events_df["event_type"] == "purchase"]["session_id"].nunique()
    conversion_rate = conversion_sessions / NUM_SESSIONS * 100
    print(f"\n  Session conversion rate: {conversion_rate:.1f}%")

    # Save to CSV
    print("\n[Saving] Writing CSV files...")
    users_df.to_csv(output_path / "users.csv", index=False)
    products_df.to_csv(output_path / "products.csv", index=False)
    events_df.to_csv(output_path / "events.csv", index=False)

    print(f"  ✓ Saved to {output_path.absolute()}/")
    print("=" * 60)
    print("Data generation complete!")
    print("=" * 60)

    return {"users": users_df, "products": products_df, "events": events_df}


# ============================================================================
# CLI Entry Point
# ============================================================================


if __name__ == "__main__":
    generate_all()
