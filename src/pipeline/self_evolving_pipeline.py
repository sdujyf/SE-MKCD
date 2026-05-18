"""Self-Evolving Conflict Resolution Pipeline.

Orchestrates the full workflow: entity recognition, experience retrieval,
model inference, conflict detection, reconciliation, and experience bank updates.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from ..data.schema import SampleSchema, PipelineResultSchema
from ..experience.bank import ExperienceBank
from ..experience.retriever import ExperienceRetriever
from ..models.base import BaseMultimodalModel
from ..auxiliary.base import BaseAuxiliaryModel
from ..utils.text import normalize_answer

logger = logging.getLogger(__name__)


class SelfEvolvingConflictPipeline:
    """Main pipeline for self-evolving multimodal knowledge conflict detection and resolution.

    Workflow:
    1. Parse sample data (entity, question, images, external document).
    2. Retrieve relevant experiences from the experience bank.
    3. If experiences found: use experience-guided inference.
    4. If no experiences:
       a. Generate answer without external document (answer1).
       b. Generate answer with external document (answer2).
       c. Detect conflict between answer1 and answer2.
       d. If no conflict: use configured default answer.
       e. If conflict: summarize conflict, reconcile, update experience bank.
    5. Evaluate correctness against groundtruth.
    """

    def __init__(
        self,
        main_model: BaseMultimodalModel,
        auxiliary_model: BaseAuxiliaryModel,
        experience_bank: ExperienceBank,
        retriever: ExperienceRetriever,
        config: dict,
    ):
        """Initialize the pipeline.

        Args:
            main_model: The main multimodal model for inference.
            auxiliary_model: Auxiliary model for conflict detection and summarization.
            experience_bank: Experience bank for storing and retrieving experiences.
            retriever: Experience retriever for matching queries to experiences.
            config: Full configuration dictionary.
        """
        self.main_model = main_model
        self.auxiliary_model = auxiliary_model
        self.experience_bank = experience_bank
        self.retriever = retriever
        self.config = config

        pipeline_cfg = config.get("pipeline", {})
        self.final_answer_when_no_conflict = pipeline_cfg.get("final_answer_when_no_conflict", "with_external")
        self.save_no_conflict_experience = pipeline_cfg.get("save_no_conflict_experience", False)
        self.update_when_experience_used = pipeline_cfg.get("update_when_experience_used", False)

        logger.info("Pipeline initialized.")

    def run_sample(self, sample: SampleSchema) -> PipelineResultSchema:
        """Run the full pipeline on a single sample.

        Args:
            sample: The input sample.

        Returns:
            PipelineResultSchema with all results.
        """
        result = PipelineResultSchema(
            ID=sample.ID,
            instance=sample.instance,
            category=sample.category,
            question=sample.open_query,
            image_paths=sample.image_path,
            external_document=sample.conflict_knowledge,
            groundtruth=sample.open_groundtruth,
        )

        try:
            entity_name = sample.instance
            entity_type = sample.category or "unknown"
            question = sample.open_query
            image_paths = sample.image_path
            external_document = sample.conflict_knowledge

            # Step 1 & 2: Retrieve experiences
            retrieval_results = self.retriever.retrieve(
                entity_name=entity_name,
                question=question,
            )
            result.retrieved_results = retrieval_results
            result.retrieval_hit = len(retrieval_results) > 0

            if result.retrieval_hit:
                # Step 3: Experience-guided inference
                result.pipeline_mode = "experience_guided_inference"
                logger.info(
                    "[%d] Experience hit for entity='%s', kp='%s'",
                    sample.ID, entity_name,
                    retrieval_results[0].get("knowledge_point", ""),
                )
                self._run_experience_guided(result, entity_name, question, image_paths, external_document, retrieval_results)
            else:
                # Step 4: Full conflict detection and resolution
                self._run_conflict_resolution(result, entity_name, entity_type, question, image_paths, external_document)

            # Step 5: Evaluate correctness
            self._evaluate_correctness(result)

            result.status = "success"
        except Exception as e:
            logger.error("[%d] Pipeline failed: %s", sample.ID, e, exc_info=True)
            result.status = "failed"
            result.error = str(e)

        return result

    def _run_experience_guided(
        self,
        result: PipelineResultSchema,
        entity_name: str,
        question: str,
        image_paths: list[str],
        external_document: str,
        retrieval_results: list[dict],
    ) -> None:
        """Run experience-guided inference when experience retrieval hits."""
        # Collect all experiences from retrieval results
        all_experiences = []
        for rr in retrieval_results:
            all_experiences.extend(rr.get("experiences", []))

        output = self.main_model.generate_with_experience(
            image_paths=image_paths,
            question=question,
            external_document=external_document,
            experiences=all_experiences,
            entity_name=entity_name,
        )
        result.final_answer = output.get("answer", "")
        result.final_reason = output.get("reason", "")
        result.experience_updated = False

        # Optionally update experience bank
        if self.update_when_experience_used:
            # Record that existing experience was used (could add metadata)
            pass

    def _run_conflict_resolution(
        self,
        result: PipelineResultSchema,
        entity_name: str,
        entity_type: str,
        question: str,
        image_paths: list[str],
        external_document: str,
    ) -> None:
        """Run full conflict detection and resolution workflow."""
        # Generate answer1 (no external doc)
        output1 = self.main_model.generate_no_external(
            image_paths=image_paths,
            question=question,
            entity_name=entity_name,
        )
        result.answer1_no_external = output1.get("answer", "")

        # Generate answer2 (with external doc)
        output2 = self.main_model.generate_with_external(
            image_paths=image_paths,
            question=question,
            external_document=external_document,
            entity_name=entity_name,
        )
        result.answer2_with_external = output2.get("answer", "")

        # Detect conflict
        
        conflict_detection = self.auxiliary_model.detect_conflict(
            question=question,
            answer1_no_external=result.answer1_no_external,
            answer2_with_external=result.answer2_with_external,
            external_document=external_document,
        )
        result.conflict_detection = conflict_detection

        if not conflict_detection.get("conflict_detected", False):
            # No conflict
            result.pipeline_mode = "no_conflict"

            if self.final_answer_when_no_conflict == "with_external":
                result.final_answer = result.answer2_with_external
                result.final_reason = output2.get("reason", "")
            elif self.final_answer_when_no_conflict == "no_external":
                result.final_answer = result.answer1_no_external
                result.final_reason = output1.get("reason", "")
            else:
                result.final_answer = result.answer2_with_external
                result.final_reason = output2.get("reason", "")

            # Optionally save no-conflict experiences
            if self.save_no_conflict_experience:
                exp_output = self.auxiliary_model.summarize_experience(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    knowledge_point="no_conflict",
                    question=question,
                    answer1_no_external=result.answer1_no_external,
                    answer2_with_external=result.answer2_with_external,
                    final_answer=result.final_answer,
                    external_document=external_document,
                    conflict_detection_result=conflict_detection,
                )
                self.experience_bank.add_experience(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    knowledge_point=exp_output.get("knowledge_point", "no_conflict"),
                    question_patterns=exp_output.get("question_patterns", []),
                    experience=exp_output.get("experience", {}),
                )
                result.experience_updated = True
            else:
                result.experience_updated = False

        else:
            # Conflict detected
            result.pipeline_mode = "conflict_detected_and_resolved"
            logger.info("[%d] Conflict detected: %s", result.ID, conflict_detection.get("conflict_type", ""))

            # Summarize conflict
            conflict_summary = self.auxiliary_model.summarize_conflict(
                question=question,
                answer1_no_external=result.answer1_no_external,
                answer2_with_external=result.answer2_with_external,
                external_document=external_document,
            )
            result.conflict_summary = conflict_summary

            # Reconcile
            reconcile_output = self.main_model.reconcile(
                image_paths=image_paths,
                question=question,
                external_document=external_document,
                answer1_no_external=result.answer1_no_external,
                answer2_with_external=result.answer2_with_external,
                conflict_summary=conflict_summary.get("summary", ""),
            )
            result.final_answer = reconcile_output.get("answer", "")
            result.final_reason = reconcile_output.get("reason", "")

            # Determine knowledge_point key
            kp = conflict_detection.get("conflict_type", "entity_recognition_conflict")
            kp = kp.replace(" ", "_").lower()

            # Summarize experience and update bank
            exp_output = self.auxiliary_model.summarize_experience(
                entity_name=entity_name,
                entity_type=entity_type,
                knowledge_point=kp,
                question=question,
                answer1_no_external=result.answer1_no_external,
                answer2_with_external=result.answer2_with_external,
                final_answer=result.final_answer,
                external_document=external_document,
                conflict_detection_result=conflict_detection,
            )
            self.experience_bank.add_experience(
                entity_name=entity_name,
                entity_type=entity_type,
                knowledge_point=exp_output.get("knowledge_point", kp),
                question_patterns=exp_output.get("question_patterns", []),
                experience=exp_output.get("experience", {}),
            )
            result.experience_updated = True

    def _evaluate_correctness(self, result: PipelineResultSchema) -> None:
        """Evaluate whether the final answer matches the groundtruth."""
        groundtruth = result.groundtruth
        if not groundtruth:
            result.correct = None
            return

        final_answer = result.final_answer
        if isinstance(final_answer, dict):
            final_answer = final_answer.get("answer", "")

        gt_norm = normalize_answer(groundtruth)
        ans_norm = normalize_answer(final_answer)

        # Exact match or groundtruth is substring of answer
        if gt_norm == ans_norm:
            result.correct = True
        elif gt_norm in ans_norm:
            result.correct = True
        else:
            result.correct = False
