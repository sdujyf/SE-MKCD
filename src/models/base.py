"""Base class and prompt templates for main multimodal models."""

from abc import ABC, abstractmethod
from typing import Any


# ============================================================
# Prompt Templates
# ============================================================

PROMPT_NO_EXTERNAL = """
Question: {question}

"""


PROMPT_WITH_EXTERNAL = """
External Document: {external_document}
Question: {question}
"""


PROMPT_RECONCILE  = """You are a knowledgeable multimodal assistant. You are given two conflicting answers and need to determine which one is more reliable.

Image(s): provided
Question: {question}

External Document: {external_document}

Answer 1 (without external document): {answer1_no_external}
Answer 2 (with external document): {answer2_with_external}

Conflict Summary: {conflict_summary}

Instructions:
- Compare Answer 1 and Answer 2.
- Consider: Which is more consistent with the visual evidence in the image(s)?
- Consider: Is the external document likely to be incorrect or misleading?
- Consider: The conflict summary provides additional analysis of the disagreement.
- Determine the most reliable final answer.
- Output a JSON object with the following keys:
  - "answer": Your final answer.
  - "reason": A brief explanation of why you chose this answer.
  - "confidence": A float between 0.0 and 1.0.

Output JSON only, no additional text."""


PROMPT_WITH_EXPERIENCE = """You are a knowledgeable multimodal assistant with access to historical conflict resolution experiences. Analyze the provided image(s), read the external document, and consider the retrieved experiences.

Image(s): provided
Question: {question}

External Document: {external_document}

Retrieved Experiences:
{experiences_text}

Instructions:
- Carefully observe the image(s).
- Read the external document.
- Review the historical experiences: these describe past situations where the model encountered similar conflicts.
- Use the experiences to guide your judgment — if a similar conflict was resolved in a particular way, consider that.
- Prioritize consistency with visual evidence and reliable historical resolutions.
- Determine the most reliable final answer.
-You must strictly follow this output format\n"
    "Answer: your answer\n"
    "Reason: <brief explanation>\n\n"
"""


# ============================================================
# Base Class
# ============================================================

class BaseMultimodalModel(ABC):
    """Abstract base class for multimodal models.

    Subclasses must implement all four generation methods.
    """

    def __init__(self, model_name_or_path: str = "", device: str = "cuda", **kwargs: Any):
        """Initialize the model.

        Args:
            model_name_or_path: Model identifier or local path.
            device: Device to run inference on.
            **kwargs: Additional model-specific arguments.
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self._model = None
        self._processor = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor into memory."""
        ...

    @abstractmethod
    def generate_no_external(
        self,
        image_paths: list[str],
        question: str,
        **kwargs: Any,
    ) -> dict:
        """Generate an answer without external document.

        Args:
            image_paths: List of paths to image files.
            question: The question to answer.
            **kwargs: Additional arguments (e.g., entity_name).

        Returns:
            Dict with keys: recognized_entity, answer, reason, confidence, raw_output.
        """
        ...

    @abstractmethod
    def generate_with_external(
        self,
        image_paths: list[str],
        question: str,
        external_document: str,
        **kwargs: Any,
    ) -> dict:
        """Generate an answer with external document.

        Args:
            image_paths: List of paths to image files.
            question: The question to answer.
            external_document: The external knowledge document.
            **kwargs: Additional arguments.

        Returns:
            Dict with keys: recognized_entity, answer, external_claim, reason, confidence, raw_output.
        """
        ...

    @abstractmethod
    def generate_with_experience(
        self,
        image_paths: list[str],
        question: str,
        external_document: str,
        experiences: list[dict],
        **kwargs: Any,
    ) -> dict:
        """Generate an answer using historical experiences.

        Args:
            image_paths: List of paths to image files.
            question: The question to answer.
            external_document: The external knowledge document.
            experiences: List of experience dicts from the experience bank.
            **kwargs: Additional arguments.

        Returns:
            Dict with keys: recognized_entity, answer, reason, confidence, raw_output.
        """
        ...

    @abstractmethod
    def reconcile(
        self,
        image_paths: list[str],
        question: str,
        external_document: str,
        answer1_no_external: str,
        answer2_with_external: str,
        conflict_summary: str,
        **kwargs: Any,
    ) -> dict:
        """Reconcile conflicting answers to produce a final answer.

        Args:
            image_paths: List of paths to image files.
            question: The question to answer.
            external_document: The external knowledge document.
            answer1_no_external: Answer without external document.
            answer2_with_external: Answer with external document.
            conflict_summary: Summary of the detected conflict.
            **kwargs: Additional arguments.

        Returns:
            Dict with keys: answer, reason, confidence, raw_output.
        """
        ...
