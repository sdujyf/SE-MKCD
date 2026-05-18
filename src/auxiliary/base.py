"""Base class and prompt templates for auxiliary models (conflict detection, summarization)."""

from abc import ABC, abstractmethod
from typing import Any


# ============================================================
# Prompt Templates
# ============================================================

PROMPT_DETECT_CONFLICT = """You are a conflict detection expert for multimodal systems. Determine whether two answers to the same question conflict with each other.

Question: {question}

External Document: {external_document}

Answer 1 (without external document): {answer1_no_external}

Answer 2 (with external document): {answer2_with_external}

Instructions:
- Compare the two answers carefully.
- Determine if they contradict each other (conflict) or are consistent (no conflict).
- Classify the type of conflict if detected:
  - "entity_recognition_conflict": The two answers identify different entities.
  - "attribute_value_conflict": The two answers disagree on an attribute value.
  - "factual_contradiction": The two answers make contradictory factual claims.
  - "no_conflict": The answers are consistent.
- Output a JSON object with the following keys:
  - "conflict_detected": true or false.
  - "conflict_type": One of the types above.
  - "conflict_dimension": "entity", "attribute", "fact", or "none".
  - "explanation": A brief explanation of why/why not there is a conflict.
  - "confidence": A float between 0.0 and 1.0.

Output JSON only, no additional text."""


PROMPT_SUMMARIZE_CONFLICT = """You are a conflict summarization expert. Summarize the conflict between two answers concisely.

Question: {question}

External Document Claim: {external_document}

Answer 1 (without external): {answer1_no_external}

Answer 2 (with external): {answer2_with_external}

Instructions:
- Summarize the key disagreement between the two answers.
- Identify what the external document claims vs. what visual analysis suggests.
- Recommend a resolution strategy: e.g., "trust_visual_evidence", "trust_verified_knowledge", "flag_for_human_review".
- Output a JSON object with the following keys:
  - "summary": A concise summary of the conflict.
  - "key_disagreement": The core point of disagreement.
  - "recommended_strategy": The recommended strategy for resolution.

Output JSON only, no additional text."""


PROMPT_SUMMARIZE_EXPERIENCE = """You are an experience distillation expert. Based on a conflict resolution process, generate a structured experience record.

Entity: {entity_name} (type: {entity_type})
Knowledge Point: {knowledge_point}
Question: {question}

Answer 1 (no external document): {answer1_no_external}
Answer 2 (with external document): {answer2_with_external}
Final Answer: {final_answer}

External Document: {external_document}

Conflict Detection Result: {conflict_detection_result}

Instructions:
- Generate a structured experience record in JSON format.
- The experience should capture what was learned from this conflict.
- Include a concise summary of the episode.
- Suggest question patterns that would retrieve this experience.
- Output a JSON object with the following keys:
  - "knowledge_point": The knowledge point key (e.g., "entity_recognition", "birth_year").
  - "question_patterns": A list of question strings that should match this experience.
  - "experience": An object containing:
    - "question": The original question.
    - "answer1_no_external": The first answer.
    - "answer2_with_external": The second answer.
    - "answer3_final": The final resolved answer.
    - "external_claim": The main claim from the external document.
    - "conflict_detected": true or false.
    - "conflict_type": The detected conflict type.
    - "conflict_dimension": The conflict dimension.
    - "final_decision": The final answer.
    - "decision_source": How the decision was made (e.g., "visual_evidence_preferred", "experience_guided").
    - "external_claim_status": "rejected" or "accepted".
    - "summary": A concise summary of this experience.
    - "created_at": Today's date.

Output JSON only, no additional text."""


# ============================================================
# Base Class
# ============================================================

class BaseAuxiliaryModel(ABC):
    """Abstract base class for auxiliary models (conflict detection, summarization)."""

    def __init__(self, model_name_or_path: str = "", **kwargs: Any):
        """Initialize the auxiliary model.

        Args:
            model_name_or_path: Model identifier or local path (for LLM-based judges).
            **kwargs: Additional arguments.
        """
        self.model_name_or_path = model_name_or_path

    @abstractmethod
    def detect_conflict(
        self,
        question: str,
        answer1_no_external: str,
        answer2_with_external: str,
        external_document: str = "",
        **kwargs: Any,
    ) -> dict:
        """Detect whether two answers conflict.

        Args:
            question: The original question.
            answer1_no_external: Answer without external document.
            answer2_with_external: Answer with external document.
            external_document: The external knowledge document.
            **kwargs: Additional arguments.

        Returns:
            Dict with keys: conflict_detected, conflict_type, conflict_dimension,
            explanation, confidence.
        """
        ...

    @abstractmethod
    def summarize_conflict(
        self,
        question: str,
        answer1_no_external: str,
        answer2_with_external: str,
        external_document: str,
        **kwargs: Any,
    ) -> dict:
        """Summarize a detected conflict.

        Args:
            question: The original question.
            answer1_no_external: Answer without external document.
            answer2_with_external: Answer with external document.
            external_document: The external knowledge document.
            **kwargs: Additional arguments.

        Returns:
            Dict with keys: summary, key_disagreement, recommended_strategy.
        """
        ...

    @abstractmethod
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
        """Summarize a conflict resolution episode into a structured experience.

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
            Dict with keys: knowledge_point, question_patterns, experience.
            The experience sub-dict has the format required by ExperienceBank.
        """
        ...
