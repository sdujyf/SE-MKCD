"""Dummy auxiliary model using simple rule-based logic for testing."""

import logging
from datetime import date
from typing import Any

from .base import BaseAuxiliaryModel
from ..utils.text import normalize_answer

logger = logging.getLogger(__name__)


class DummyAuxiliaryModel(BaseAuxiliaryModel):
    """A rule-based auxiliary model for testing the pipeline.

    Conflict detection:
    - If answer1 and answer2 (normalized) are different, conflict_detected = True.
    - If they are the same, conflict_detected = False.

    Conflict summarization:
    - Generates a simple template-based summary.

    Experience summarization:
    - Creates a structured experience record from the conflict episode.
    """

    def __init__(self, model_name_or_path: str = "", **kwargs: Any):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        logger.info("Initialized DummyAuxiliaryModel")

    def detect_conflict(
        self,
        question: str,
        answer1_no_external: str,
        answer2_with_external: str,
        external_document: str = "",
        **kwargs: Any,
    ) -> dict:
        """Detect conflict by comparing normalized answers.

        Args:
            question: The original question.
            answer1_no_external: Answer without external document.
            answer2_with_external: Answer with external document.
            external_document: The external knowledge document.
            **kwargs: Additional arguments.

        Returns:
            Conflict detection result dict.
        """
        a1 = normalize_answer(answer1_no_external)
        a2 = normalize_answer(answer2_with_external)

        conflict_detected = a1 != a2

        if conflict_detected:
            # Determine conflict dimension
            question_lower = question.lower()

            if any(w in question_lower for w in ["who", "what", "which entity", "identify"]):
                conflict_type = "entity_recognition_conflict"
                conflict_dimension = "entity"
            elif any(w in question_lower for w in ["when", "born", "year", "date", "age"]):
                conflict_type = "attribute_value_conflict"
                conflict_dimension = "attribute"
            else:
                conflict_type = "factual_contradiction"
                conflict_dimension = "fact"

            explanation = (
                f"Conflict detected: Answer 1 is '{answer1_no_external}' but "
                f"Answer 2 is '{answer2_with_external}'. The external document "
                f"contradicts the visual evidence."
            )
            confidence = 1.0
        else:
            conflict_type = "no_conflict"
            conflict_dimension = "none"
            explanation = (
                f"No conflict: both answers agree on '{answer1_no_external}'. "
                f"The external document does not introduce contradictions."
            )
            confidence = 0.95

        return {
            "conflict_detected": conflict_detected,
            "conflict_type": conflict_type,
            "conflict_dimension": conflict_dimension,
            "explanation": explanation,
            "confidence": confidence,
        }

    def summarize_conflict(
        self,
        question: str,
        answer1_no_external: str,
        answer2_with_external: str,
        external_document: str,
        **kwargs: Any,
    ) -> dict:
        """Generate a template-based conflict summary.

        Args:
            question: The original question.
            answer1_no_external: Answer without external document.
            answer2_with_external: Answer with external document.
            external_document: The external knowledge document.
            **kwargs: Additional arguments.

        Returns:
            Conflict summary dict.
        """
        summary = (
            f"The model produced two different answers for the question '{question}'. "
            f"Without external document, the answer was '{answer1_no_external}'. "
            f"With external document, the answer was '{answer2_with_external}'. "
            f"This indicates that the external document conflicts with the model's "
            f"visual analysis of the image."
        )
        key_disagreement = (
            f"The core disagreement is about the identity: visual evidence suggests "
            f"'{answer1_no_external}', while the external document claims '{answer2_with_external}'."
        )
        recommended_strategy = "trust_visual_evidence_or_verified_knowledge"

        return {
            "summary": summary,
            "key_disagreement": key_disagreement,
            "recommended_strategy": recommended_strategy,
        }

    def summarize_experience(
        self,
        entity_name: str,
        entity_type: str,
        knowledge_point: str,
        question: str,
        answer1_no_external: str,
        answer2_with_external: str,
        final_answer: str,
        external_document: str,
        conflict_detection_result: dict,
        **kwargs: Any,
    ) -> dict:
        """Generate a structured experience record.

        Args:
            entity_name: Entity name.
            entity_type: Entity type.
            knowledge_point: Knowledge point key.
            question: Original question.
            answer1_no_external: Answer without external document.
            answer2_with_external: Answer with external document.
            final_answer: The final resolved answer.
            external_document: External knowledge document.
            conflict_detection_result: Result from detect_conflict.
            **kwargs: Additional arguments.

        Returns:
            Experience dict ready for ExperienceBank.
        """
        safe_entity = entity_name.lower().replace(" ", "_").replace("/", "_")
        safe_kp = knowledge_point.lower().replace(" ", "_")

        conflict_detected = conflict_detection_result.get("conflict_detected", False)
        conflict_type = conflict_detection_result.get("conflict_type", "factual_contradiction")
        conflict_dimension = conflict_detection_result.get("conflict_dimension", "fact")

        experience_id = f"{safe_entity}_{safe_kp}_0001"

        # Determine if external claim was accepted or rejected
        a1_norm = normalize_answer(final_answer)
        a2_norm = normalize_answer(answer2_with_external)
        external_claim_status = "accepted" if a1_norm == a2_norm else "rejected"

        summary = (
            f"The model was exposed to conflicting external information about {entity_name}. "
            f"Without external document, the answer was '{answer1_no_external}'. "
            f"With external document, the answer changed to '{answer2_with_external}'. "
            f"After conflict resolution, the final answer is '{final_answer}'. "
            f"The external claim was {external_claim_status}."
        )

        experience = {
            "experience_id": experience_id,
            "question": question,
            "answer1_no_external": answer1_no_external,
            "answer2_with_external": answer2_with_external,
            "answer3_final": final_answer,
            "external_claim": external_document[:200] if len(external_document) > 200 else external_document,
            "conflict_detected": conflict_detected,
            "conflict_type": conflict_type,
            "conflict_dimension": conflict_dimension,
            "final_decision": final_answer,
            "decision_source": "visual_evidence_preferred",
            "external_claim_status": external_claim_status,
            "summary": summary,
            "created_at": str(date.today()),
        }

        return {
            "knowledge_point": knowledge_point,
            "question_patterns": [question],
            "experience": experience,
        }
