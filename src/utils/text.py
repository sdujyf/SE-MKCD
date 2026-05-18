"""Text processing utilities for normalization and JSON parsing."""

import json
import re


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison.

    Strips whitespace, lowercases, and removes extra spaces.

    Args:
        text: The raw answer text.

    Returns:
        Normalized answer string.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".")
    return text.strip()


def parse_json_from_output(raw_output: str) -> dict:
    """Attempt to extract and parse a JSON object from model output.

    Tries to find JSON between ```json fences, or the first { ... } block.

    Args:
        raw_output: Raw model output string.

    Returns:
        Parsed JSON dictionary, or empty dict on failure.
    """
    if not raw_output:
        return {}

    # Try ```json fence
    match = re.search(r"```json\s*(.*?)\s*```", raw_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try first { ... } block
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try parsing the whole string
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {}
