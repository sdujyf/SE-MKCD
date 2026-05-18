"""Text normalization and matching utilities for experience retrieval."""

import re
from typing import Sequence


def normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, strip, collapse whitespace.

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_token_overlap(
    query: str,
    candidate: str,
    normalize: bool = True,
) -> float:
    """Compute token overlap score between query and candidate strings.

    Uses Jaccard-like overlap: |intersection| / max(|query_tokens|, |candidate_tokens|).

    Args:
        query: Query text.
        candidate: Candidate text to match against.
        normalize: Whether to normalize texts first.

    Returns:
        Score between 0.0 and 1.0.
    """
    if normalize:
        query = normalize_text(query)
        candidate = normalize_text(candidate)

    q_tokens = set(query.split())
    c_tokens = set(candidate.split())

    if not q_tokens or not c_tokens:
        return 0.0

    intersection = q_tokens & c_tokens
    return len(intersection) / max(len(q_tokens), len(c_tokens))


def match_entity_name(
    query_name: str,
    entity_key: str,
    aliases: Sequence[str],
) -> bool:
    """Check if a query entity name matches an experience bank entity.

    Compares against the entity key and all aliases after normalization.

    Args:
        query_name: The entity name from the query.
        entity_key: The primary entity key in the bank.
        aliases: List of alias strings for the entity.

    Returns:
        True if the query matches the entity.
    """
    query_norm = normalize_text(query_name)
    if query_norm == normalize_text(entity_key):
        return True
    for alias in aliases:
        if query_norm == normalize_text(alias):
            return True
    return False
