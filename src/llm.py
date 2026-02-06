"""
LLM integration via Groq API for product insights.

Provides prompt construction and inference for customer journey
analysis using Llama 3.1 8B model.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are an expert product analyst specializing in customer journey analysis for e-commerce/marketplace platforms.

Your role is to:
- Analyze user journey patterns from graph-based behavioral data
- Identify conversion drivers, drop-off points, and optimization opportunities
- Provide actionable product insights backed by specific journey examples
- Compare cohorts (high-LTV vs low-LTV, converters vs churners, etc.)

Guidelines:
- Be precise and cite specific patterns, percentages, and counts from the provided data
- Frame insights in product/business terms that would be actionable for a product manager
- When comparing cohorts, highlight meaningful differences in behavior
- Suggest concrete next steps or experiments when appropriate

Always base your analysis on the journey data provided. If the data is insufficient, acknowledge limitations."""


# ============================================================================
# LLM Client
# ============================================================================


class JourneyLLM:
    """
    LLM client for customer journey analysis.

    Wraps Groq API for Llama 3.1 inference with journey-specific
    prompt construction.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
            model: Model identifier to use.

        Raises:
            ValueError: If no API key is provided or found.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Set it in .env file or pass directly.")

        self.model = model
        self.client = Groq(api_key=self.api_key)

    def generate(
        self,
        query: str,
        context: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate analysis from query and context.

        Args:
            query: User's product/analytics question.
            context: Journey context from retrieval (GraphRAG or naive).
            temperature: Sampling temperature (lower = more focused).
            max_tokens: Maximum response length.

        Returns:
            LLM-generated analysis text.
        """
        user_message = f"""## Customer Journey Context:
{context}

## Product Question:
{query}

## Analysis & Insight:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    def analyze_with_method(
        self,
        query: str,
        context: str,
        method: Literal["graphrag", "naive"] = "graphrag",
        temperature: float = 0.3,
    ) -> dict[str, str]:
        """
        Generate analysis with method metadata.

        Args:
            query: User's product/analytics question.
            context: Journey context from retrieval.
            method: Retrieval method used ("graphrag" or "naive").
            temperature: Sampling temperature.

        Returns:
            Dictionary with 'method', 'query', 'context', and 'response'.
        """
        response = self.generate(query, context, temperature=temperature)

        return {
            "method": method,
            "query": query,
            "context": context,
            "response": response,
        }


# ============================================================================
# Convenience Functions
# ============================================================================


def get_llm(api_key: str | None = None) -> JourneyLLM:
    """
    Get an initialized LLM client.

    Args:
        api_key: Optional API key override.

    Returns:
        Configured JourneyLLM instance.
    """
    return JourneyLLM(api_key=api_key)


def quick_analyze(
    query: str,
    context: str,
    api_key: str | None = None,
) -> str:
    """
    Quick one-shot analysis without persistent client.

    Args:
        query: Product/analytics question.
        context: Journey context string.
        api_key: Optional API key override.

    Returns:
        LLM analysis response.
    """
    llm = get_llm(api_key)
    return llm.generate(query, context)


# ============================================================================
# CLI Entry Point
# ============================================================================


if __name__ == "__main__":
    # Test with sample context
    sample_context = """## Customer Journey Context

### Sample Journeys:

**Journey 1:**
User (segment: high_value, LTV: $523.45, churned: False)
Journey: page_view → search → click (Electronics - $299.99) → click (Electronics - $49.99) → add_to_cart → purchase

**Journey 2:**
User (segment: low, LTV: $28.50, churned: True)
Journey: page_view → click (Fashion - $45.00) → exit

### Statistics:
- Total journeys analyzed: 50
- Average events per session: 4.2
- Event distribution: page_view: 120, click: 85, add_to_cart: 25, purchase: 15, exit: 35"""

    sample_query = "What are the key differences between high-value and churned users?"

    try:
        llm = get_llm()
        response = llm.generate(sample_query, sample_context)
        print("Query:", sample_query)
        print("-" * 50)
        print("Response:", response)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set GROQ_API_KEY in .env file to test LLM functionality.")
