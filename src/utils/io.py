"""IO utilities for reading configs, JSON, and managing output files."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config


def read_json(json_path: str) -> Any:
    """Read a JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        FileNotFoundError: If json_path does not exist.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, json_path: str) -> None:
    """Write data to a JSON file with UTF-8 encoding and indentation.

    Args:
        data: Data to serialize.
        json_path: Destination file path.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug("Written JSON to %s", json_path)


def write_jsonl_line(record: dict, output_path: str) -> None:
    """Append a single JSON line to a JSONL file.

    Args:
        record: Dictionary to write as a JSON line.
        output_path: Path to the JSONL file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_completed_ids(output_path: str) -> set[int]:
    """Read already-completed sample IDs from an existing JSONL output file.

    Args:
        output_path: Path to the JSONL results file.

    Returns:
        Set of completed sample IDs.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return set()

    completed = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "ID" in record:
                    completed.add(record["ID"])
            except json.JSONDecodeError:
                continue
    logger.info("Found %d completed samples in %s", len(completed), output_path)
    return completed
