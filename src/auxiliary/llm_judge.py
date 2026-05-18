"""LLM-based judge for conflict detection, summarization, and experience distillation.

This module provides a structured interface for using an LLM (via API or local)
as the auxiliary model. Currently, it contains placeholder implementations
that log warnings and delegate to rule-based fallback logic.

TODO:
- Implement OpenAI API integration.
- Implement Anthropic Claude API integration.
- Implement local LLM integration (vLLM, llama.cpp, etc.).
"""

import json
import logging
from datetime import date
from typing import Any, Optional

from .base import (
    BaseAuxiliaryModel,
    PROMPT_DETECT_CONFLICT,
    PROMPT_SUMMARIZE_CONFLICT,
    PROMPT_SUMMARIZE_EXPERIENCE,
)
from ..utils.text import normalize_answer

logger = logging.getLogger(__name__)


class LLMJudgeAuxiliaryModel(BaseAuxiliaryModel):
    """LLM-powered auxiliary model for conflict analysis.

    Can be configured to use OpenAI, Anthropic, or local LLM backends.
    Currently provides rule-based fallback when no LLM backend is configured.
    """

    def __init__(
        self,
        model_name_or_path: str = "",
        api_type: str = "openai",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the LLM judge.

        Args:
            model_name_or_path: Model identifier (e.g., "gpt-4", "claude-3-opus").
            api_type: Backend type: "openai", "anthropic", or "local".
            api_key: API key for the service.
            api_base: Optional custom API base URL.
            **kwargs: Additional arguments.
        """
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        self.api_type = api_type
        self.api_key = api_key
        self.api_base = api_base
        self._client = None
        logger.info(
            "Initialized LLMJudgeAuxiliaryModel (api_type=%s, model=%s)",
            api_type, model_name_or_path,
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM backend with a prompt.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Raw LLM response string.

        Raises:
            NotImplementedError: If the backend is not implemented.
        """
        if self.api_type == "openai":
            return self._call_openai(prompt)
        elif self.api_type == "anthropic":
            return self._call_anthropic(prompt)
        elif self.api_type == "local":
            return self._call_local(prompt)
        else:
            raise NotImplementedError(f"Unsupported api_type: {self.api_type}")

    def _call_openai(self, prompt: str) -> str:
        """Call DeepSeek API through OpenAI-compatible SDK.

        Although this method is named _call_openai, it can call any
        OpenAI-compatible API endpoint, including DeepSeek.

        Args:
            prompt: The formatted prompt string.

        Returns:
            The model response content as a string.

        Raises:
            RuntimeError: If API key is missing or API call fails.
        """
        import os

        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI SDK is not installed. Please install it with: pip install openai"
            ) from e

        api_key = "sk-2821b77bd742480695b1cd89304c88b0"

        if not api_key:
            raise RuntimeError(
                "DeepSeek API key is missing. Please set DEEPSEEK_API_KEY "
                "or pass api_key in the config."
            )

        base_url = self.api_base or "https://api.deepseek.com"
        model_name =  "deepseek-v4-flash"

        if self._client is None:
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

        try:
            response = self._client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise conflict detection and reasoning assistant. "
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                stream=False,
                reasoning_effort="high",
                extra_body={
                    "thinking": {
                        "type": "enabled"
                    }
                },
            )

            content = response.choices[0].message.content

            if content is None:
                raise RuntimeError("DeepSeek API returned empty content.")

            return content.strip()

        except Exception as e:
            raise RuntimeError(f"DeepSeek API call failed: {e}") from e

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API.

        TODO: Implement Anthropic API integration.
        """
        raise NotImplementedError(
            "Anthropic API integration is not yet implemented. "
            "Install anthropic package and implement _call_anthropic()."
        )

    def _call_local(self, prompt: str) -> str:
        """Call a local LLM.

        TODO: Implement local LLM integration (vLLM, llama.cpp, transformers).
        """
        raise NotImplementedError(
            "Local LLM integration is not yet implemented."
        )

    def _fallback_detect_conflict(
        self,
        answer1_no_external: str,
        answer2_with_external: str,
    ) -> dict:
        """Rule-based fallback for conflict detection."""
        a1 = normalize_answer(answer1_no_external)
        a2 = normalize_answer(answer2_with_external)
        conflict_detected = a1 != a2

        return {
            "conflict_detected": conflict_detected,
            "conflict_type": "entity_recognition_conflict" if conflict_detected else "no_conflict",
            "conflict_dimension": "entity" if conflict_detected else "none",
            "explanation": (
                f"Fallback detection: answers differ ('{a1}' vs '{a2}')." if conflict_detected
                else f"Fallback detection: answers match ('{a1}')."
            ),
            "confidence": 1.0 if conflict_detected else 0.95,
        }

    def detect_conflict(
        self,
        question: str,
        answer1_no_external: str,
        answer2_with_external: str,
        external_document: str = "",
        **kwargs: Any,
    ) -> dict:
        """Detect conflict using LLM or fallback.

        Args:
            question: The original question.
            answer1_no_external: Answer without external document.
            answer2_with_external: Answer with external document.
            external_document: The external knowledge document.
            **kwargs: Additional arguments.

        Returns:
            Conflict detection result dict.
        """
        print("------开始冲突检测-----------")
        
        prompt = PROMPT_DETECT_CONFLICT.format(
            question=question,
            external_document=external_document,
            answer1_no_external=answer1_no_external,
            answer2_with_external=answer2_with_external,
        )
        raw_output = self._call_llm(prompt)
        print("-----检测结果------")
        print(raw_output)
        parsed = self._parse_json(raw_output)
        if parsed:
            return parsed
        '''
        except NotImplementedError:
            logger.debug("LLM call not available, using fallback detection.")
        except Exception as e:
            logger.warning("LLM call failed: %s. Using fallback detection.", e)
        '''

        return self._fallback_detect_conflict(answer1_no_external, answer2_with_external)

    def summarize_conflict(
        self,
        question: str,
        answer1_no_external: str,
        answer2_with_external: str,
        external_document: str,
        **kwargs: Any,
    ) -> dict:
        """Summarize conflict using LLM or fallback.

        Args:
            question: The original question.
            answer1_no_external: Answer without external document.
            answer2_with_external: Answer with external document.
            external_document: The external knowledge document.
            **kwargs: Additional arguments.

        Returns:
            Conflict summary dict.
        """
        try:
            prompt = PROMPT_SUMMARIZE_CONFLICT.format(
                question=question,
                external_document=external_document,
                answer1_no_external=answer1_no_external,
                answer2_with_external=answer2_with_external,
            )
            raw_output = self._call_llm(prompt)
            parsed = self._parse_json(raw_output)
            if parsed:
                return parsed
        except NotImplementedError:
            logger.debug("LLM call not available, using fallback summary.")
        except Exception as e:
            logger.warning("LLM call failed: %s. Using fallback summary.", e)

        return {
            "summary": (
                f"Conflict: Answer1='{answer1_no_external}' vs Answer2='{answer2_with_external}'. "
                f"The external document contradicts the visual evidence."
            ),
            "key_disagreement": f"Entity identity: '{answer1_no_external}' vs '{answer2_with_external}'.",
            "recommended_strategy": "trust_visual_evidence_or_verified_knowledge",
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
        """Summarize experience using LLM or fallback.

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
        try:
            prompt = PROMPT_SUMMARIZE_EXPERIENCE.format(
                entity_name=entity_name,
                entity_type=entity_type,
                knowledge_point=knowledge_point,
                question=question,
                answer1_no_external=answer1_no_external,
                answer2_with_external=answer2_with_external,
                final_answer=final_answer,
                external_document=external_document,
                conflict_detection_result=json.dumps(conflict_detection_result, ensure_ascii=False),
            )
            raw_output = self._call_llm(prompt)
            parsed = self._parse_json(raw_output)
            if parsed:
                return parsed
        except NotImplementedError:
            logger.debug("LLM call not available, using fallback experience summary.")
        except Exception as e:
            logger.warning("LLM call failed: %s. Using fallback experience summary.", e)

        safe_entity = entity_name.lower().replace(" ", "_").replace("/", "_")
        safe_kp = knowledge_point.lower().replace(" ", "_")

        return {
            "knowledge_point": knowledge_point,
            "question_patterns": [question],
            "experience": {
                "experience_id": f"{safe_entity}_{safe_kp}_0001",
                "question": question,
                "answer1_no_external": answer1_no_external,
                "answer2_with_external": answer2_with_external,
                "answer3_final": final_answer,
                "external_claim": external_document[:200] if len(external_document) > 200 else external_document,
                "conflict_detected": conflict_detection_result.get("conflict_detected", False),
                "conflict_type": conflict_detection_result.get("conflict_type", "factual_contradiction"),
                "conflict_dimension": conflict_detection_result.get("conflict_dimension", "fact"),
                "final_decision": final_answer,
                "decision_source": "visual_evidence_preferred",
                "external_claim_status": "rejected",
                "summary": f"Resolved conflict for {entity_name}: chose '{final_answer}' over external claim.",
                "created_at": str(date.today()),
            },
        }

    @staticmethod
    def _parse_json(raw_output: str) -> dict:
        """Parse JSON from LLM output."""
        if not raw_output:
            return {}
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        import re
        match = re.search(r"```json\s*(.*?)\s*```", raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {}
