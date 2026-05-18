"""Dummy multimodal model for testing the pipeline without real model dependencies."""

import logging
import re
import time
from typing import Any

from .base import BaseMultimodalModel

logger = logging.getLogger(__name__)


class DummyMultimodalModel(BaseMultimodalModel):
    """A dummy model that returns controlled responses for pipeline testing.

    Behavior:
    - generate_no_external: Returns the entity_name (from kwargs) or a placeholder.
    - generate_with_external: Extracts "This is an image of XXX." from external_document
      and returns XXX as the answer, creating a simulated conflict.
    - generate_with_experience: Prefers the experience-guided answer.
    - reconcile: Defaults to answer1_no_external, simulating visual-evidence preference.
    """

    def __init__(self, model_name_or_path: str = "", device: str = "cpu", **kwargs: Any):
        super().__init__(model_name_or_path=model_name_or_path, device=device, **kwargs)
        logger.info("Initialized DummyMultimodalModel")

    def load_model(self) -> None:
        """No-op for dummy model."""
        logger.info("DummyMultimodalModel: load_model() called (no-op)")

    def _extract_external_entity(self, external_document: str) -> str:
        """Extract entity name from external document using pattern matching.

        Looks for patterns like 'This is an image of XXX.' or 'The XXX is ...'

        Args:
            external_document: The external knowledge text.

        Returns:
            Extracted entity name, or empty string.
        """
        # Pattern: "This is an image of XXX."
        match = re.search(r"this is an image of\s+([\w\s\-]+?)[\.\n]", external_document, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern: "The common XXX is ..."
        match = re.search(r"the common\s+([\w\-]+)\s+is", external_document, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern: first proper noun after "of" near start
        match = re.search(r"of\s+([\w\-]+)", external_document, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return ""

    def generate_no_external(
        self,
        image_paths: list[str],
        question: str,
        **kwargs: Any,
    ) -> dict:
        """Return answer based on provided entity_name (from kwargs)."""
        entity_name = kwargs.get("entity_name", "Unknown")
        time.sleep(0.01)  # Simulate tiny delay
        return {
            "recognized_entity": entity_name,
            "answer": entity_name,
            "reason": "Dummy no-external inference uses the provided entity name based on visual analysis.",
            "confidence": 0.95,
            "raw_output": f'{{"recognized_entity": "{entity_name}", "answer": "{entity_name}", ...}}',
        }

    def generate_with_external(
        self,
        image_paths: list[str],
        question: str,
        external_document: str,
        **kwargs: Any,
    ) -> dict:
        """Return answer based on external document, potentially conflicting."""
        entity_name = kwargs.get("entity_name", "")
        external_entity = self._extract_external_entity(external_document)
        answer = external_entity if external_entity else entity_name
        external_claim = external_document.strip()

        # Truncate claim to reasonable length
        if len(external_claim) > 200:
            external_claim = external_claim[:200] + "..."

        time.sleep(0.01)
        return {
            "recognized_entity": entity_name,
            "answer": answer,
            "external_claim": external_claim,
            "reason": f"Dummy with-external inference found external claim: '{external_entity}'. Answer uses external document.",
            "confidence": 0.85,
            "raw_output": f'{{"recognized_entity": "{entity_name}", "answer": "{answer}", ...}}',
        }

    def generate_with_experience(
        self,
        image_paths: list[str],
        question: str,
        external_document: str,
        experiences: list[dict],
        **kwargs: Any,
    ) -> dict:
        """Return answer guided by historical experiences."""
        entity_name = kwargs.get("entity_name", "Unknown")

        # Use the first experience's final_decision if available
        final_answer = entity_name
        if experiences:
            first_exp = experiences[0]
            if isinstance(first_exp, dict) and "final_decision" in first_exp:
                final_answer = first_exp["final_decision"]
            elif isinstance(first_exp, dict) and "answer3_final" in first_exp:
                final_answer = first_exp["answer3_final"]

        time.sleep(0.01)
        return {
            "recognized_entity": entity_name,
            "answer": final_answer,
            "reason": f"Dummy experience-guided inference: using {len(experiences)} historical experience(s). Final answer: {final_answer}.",
            "confidence": 0.90,
            "raw_output": f'{{"recognized_entity": "{entity_name}", "answer": "{final_answer}", ...}}',
        }

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
        """Reconcile by trusting answer1 (visual evidence without external doc)."""
        time.sleep(0.01)
        return {
            "answer": answer1_no_external,
            "reason": f"Dummy reconcile: external document conflicts with visual evidence. Retaining answer without external: {answer1_no_external}.",
            "confidence": 0.90,
            "raw_output": f'{{"answer": "{answer1_no_external}", ...}}',
        }
