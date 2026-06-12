import pandas as pd
import re


def parse_messages(messages):
    """Extract user questions and assistant responses from conversations.

    Args:
        messages (list[dict]): Conversation messages containing role
            and content fields.

    Returns:
        pd.Series: Series containing:
            - question (str): User product description.
            - response (str): Assistant HTS classification response.
    """

    user_text = None
    assistant_text = None

    # Extract message content by role
    for msg in messages:
        if msg["role"] == "user":
            user_text = msg["content"]

        elif msg["role"] == "assistant":
            assistant_text = msg["content"]

    return pd.Series({
        "question": user_text,
        "response": assistant_text
    })


def extract_fields(row):
    """Extract product description, HTS code, and reasoning text.

    Parses the assistant response and separates the HTS code from
    the classification reasoning.

    Args:
        row (pd.Series): Row containing question and response fields.

    Returns:
        pd.Series: Series containing:
            - product_description (str)
            - hts_codes (str | None)
            - reasoning (str | None)
    """

    response = row["response"]

    # Extract HTS code section
    hts_match = re.search(
        r"HTS US Code ->\s*(.*?)\s*Reasoning ->",
        response,
        flags=re.DOTALL
    )

    # Extract reasoning section
    reasoning_match = re.search(
        r"Reasoning ->\s*(.*)",
        response,
        flags=re.DOTALL
    )

    return pd.Series({
        "product_description": row["question"],
        "hts_codes": (
            hts_match.group(1).strip()
            if hts_match
            else None
        ),
        "reasoning": (
            reasoning_match.group(1).strip()
            if reasoning_match
            else None
        )
    })


def clean_text(text):
    """Normalize product descriptions for NLP processing.

    Converts text to lowercase, removes boilerplate wording,
    strips punctuation, and normalizes whitespace.

    Args:
        text (str): Raw product description.

    Returns:
        str: Cleaned product description.
    """

    if pd.isnull(text):
        return text

    # Convert text to lowercase
    text = text.lower()

    # Remove repeated prompt wording
    text = re.sub(
        r"what is the hts us code for\s*",
        "",
        text
    )

    # Remove punctuation
    text = re.sub(r"[?.,;:]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
