"""Metrics and summary computation utilities."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compute_summary(results: list[dict], groundtruth_available: bool = True) -> dict:
    """Compute summary statistics from pipeline results.

    Args:
        results: List of pipeline result dictionaries.
        groundtruth_available: Whether groundtruth is present for accuracy calculation.

    Returns:
        Dictionary of summary statistics.
    """
    total = len(results)
    success = sum(1 for r in results if r.get("status") == "success")
    failed = total - success
    retrieval_hit_count = sum(1 for r in results if r.get("retrieval_hit", False))
    conflict_detected_count = sum(
        1 for r in results if r.get("conflict_detection", {}).get("conflict_detected", False)
    )
    experience_updated_count = sum(1 for r in results if r.get("experience_updated", False))

    summary = {
        "total": total,
        "success": success,
        "failed": failed,
        "retrieval_hit_count": retrieval_hit_count,
        "conflict_detected_count": conflict_detected_count,
        "experience_updated_count": experience_updated_count,
    }

    if groundtruth_available:
        correct_count = sum(1 for r in results if r.get("correct", False) is True)
        total_with_gt = sum(1 for r in results if r.get("correct") is not None)
        summary["correct_count"] = correct_count
        summary["total_with_groundtruth"] = total_with_gt
        if total_with_gt > 0:
            summary["accuracy"] = correct_count / total_with_gt
        else:
            summary["accuracy"] = 0.0

    return summary
