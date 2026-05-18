"""Data schemas for input samples and pipeline results."""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class SampleSchema:
    """Schema for a single input sample.

    Attributes:
        ID: Unique sample identifier.
        instance: Entity name (e.g., "Sunflower").
        category: Category label (e.g., "ER").
        knowledge: Knowledge text describing the entity.
        image_path: List of image file paths (always a list after parsing).
        conflict_knowledge: External conflicting knowledge text.
        open_query: The open-ended question.
        open_groundtruth: Ground truth answer for open-ended question.
        conflict_openanswer: The answer implied by conflicting knowledge.
        mcq_query: Multiple-choice question string.
        mcq_groundtruth: Ground truth MCQ answer label.
        conflict_mcqanswer: Conflicting MCQ answer label.
    """
    ID: int
    instance: str
    category: str = ""
    knowledge: str = ""
    image_path: list[str] = field(default_factory=list)
    conflict_knowledge: str = ""
    open_query: str = ""
    open_groundtruth: str = ""
    conflict_openanswer: str = ""
    mcq_query: str = ""
    mcq_groundtruth: str = ""
    conflict_mcqanswer: str = ""


@dataclass
class PipelineResultSchema:
    """Schema for a single pipeline result.

    Attributes:
        ID: Sample identifier.
        instance: Entity name.
        category: Category label.
        question: The question asked.
        image_paths: List of image paths used.
        external_document: External conflicting knowledge used.
        groundtruth: Ground truth answer.
        retrieval_hit: Whether experience retrieval succeeded.
        retrieved_results: List of retrieved experience results.
        answer1_no_external: Answer without external document.
        answer2_with_external: Answer with external document.
        conflict_detection: Conflict detection result dict.
        conflict_summary: Conflict summary dict.
        final_answer: Final resolved answer.
        final_reason: Reasoning for final answer.
        pipeline_mode: One of "experience_guided_inference", "no_conflict", "conflict_detected_and_resolved".
        experience_updated: Whether experience bank was updated.
        correct: Whether final answer matches groundtruth.
        status: "success" or "failed".
        error: Error message if failed.
    """
    ID: int
    instance: str = ""
    category: str = ""
    question: str = ""
    image_paths: list[str] = field(default_factory=list)
    external_document: str = ""
    groundtruth: str = ""

    retrieval_hit: bool = False
    retrieved_results: list[dict] = field(default_factory=list)

    answer1_no_external: str = ""
    answer2_with_external: str = ""
    conflict_detection: dict = field(default_factory=dict)
    conflict_summary: dict = field(default_factory=dict)
    final_answer: str = ""
    final_reason: str = ""

    pipeline_mode: str = ""
    experience_updated: bool = False
    correct: Optional[bool] = None
    status: str = "success"
    error: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
