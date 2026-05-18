"""Dataset loading and validation utilities."""

import json
import logging
from pathlib import Path
from typing import Optional

from .schema import SampleSchema

logger = logging.getLogger(__name__)


def load_dataset(
    data_path: str,
    image_root: str = "./",
    max_images: int = 2,
    max_samples: Optional[int] = None,
) -> list[SampleSchema]:
    """Load and validate a dataset from a JSON file.

    Args:
        data_path: Path to the JSON data file.
        image_root: Root directory for resolving relative image paths.
        max_images: Maximum number of images to use per sample.
        max_samples: If set, only load the first N samples.

    Returns:
        List of validated SampleSchema instances.

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError: If the JSON structure is invalid.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected a JSON array at top level, got {type(raw_data).__name__}")

    image_root_path = Path(image_root)
    samples = []

    for idx, item in enumerate(raw_data):
        if max_samples is not None and idx >= max_samples:
            break

        try:
            validate_sample(item)
            # Resolve image paths
            raw_paths = item["image_path"]
            if isinstance(raw_paths, str):
                raw_paths = [raw_paths]
            elif not isinstance(raw_paths, list):
                raise ValueError(f"image_path must be str or list, got {type(raw_paths).__name__}")

            resolved = []
            for p in raw_paths[:max_images]:
                img_path = Path(p)
                if not img_path.is_absolute():
                    img_path = image_root_path / img_path
                resolved.append(str(img_path))

            sample = SampleSchema(
                ID=item.get("ID", idx),
                instance=item.get("instance", ""),
                category=item.get("category", ""),
                knowledge=item.get("knowledge", ""),
                image_path=resolved,
                conflict_knowledge=item.get("conflict_knowledge", ""),
                open_query=item.get("open_query", ""),
                open_groundtruth=item.get("open_groundtruth", ""),
                conflict_openanswer=item.get("conflict_openanswer", ""),
                mcq_query=item.get("mcq_query", ""),
                mcq_groundtruth=item.get("mcq_groundtruth", ""),
                conflict_mcqanswer=item.get("conflict_mcqanswer", ""),
            )
            samples.append(sample)
        except Exception as e:
            logger.warning("Skipping sample at index %d: %s", idx, e)

    logger.info("Loaded %d samples from %s", len(samples), data_path)
    return samples


def validate_sample(item: dict) -> None:
    """Validate required fields in a sample dictionary.

    Args:
        item: Raw sample dictionary.

    Raises:
        ValueError: If required fields are missing or have invalid types.
    """
    required = ["instance", "open_query", "conflict_knowledge"]
    for field in required:
        if field not in item:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(item[field], str):
            raise ValueError(f"Field '{field}' must be str, got {type(item[field]).__name__}")

    if "image_path" not in item:
        raise ValueError("Missing required field: image_path")
    if not isinstance(item["image_path"], (str, list)):
        raise ValueError(f"Field 'image_path' must be str or list, got {type(item['image_path']).__name__}")
