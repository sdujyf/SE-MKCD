"""Integration test for the full pipeline using dummy models."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.schema import SampleSchema
from src.experience.bank import ExperienceBank
from src.experience.retriever import ExperienceRetriever
from src.models.dummy_model import DummyMultimodalModel
from src.auxiliary.dummy_auxiliary import DummyAuxiliaryModel
from src.pipeline.self_evolving_pipeline import SelfEvolvingConflictPipeline


def make_sample(instance, category, question, image_paths, conflict_knowledge, groundtruth, sample_id=0):
    """Helper to create a SampleSchema for testing."""
    return SampleSchema(
        ID=sample_id,
        instance=instance,
        category=category,
        knowledge=f"This is an image of {instance}.",
        image_path=image_paths,
        conflict_knowledge=conflict_knowledge,
        open_query=question,
        open_groundtruth=groundtruth,
        conflict_openanswer="",
        mcq_query="",
        mcq_groundtruth="",
        conflict_mcqanswer="",
    )


class TestPipelineDummy:
    """Integration tests for the pipeline with dummy models."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline with dummy models and a temp experience bank."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = str(Path(tmpdir) / "test_bank.json")
            bank = ExperienceBank(bank_path=bank_path, backup=False)
            retriever = ExperienceRetriever(bank=bank, min_similarity=0.5)
            main_model = DummyMultimodalModel()
            main_model.load_model()
            aux_model = DummyAuxiliaryModel()

            config = {
                "pipeline": {
                    "final_answer_when_no_conflict": "with_external",
                    "save_no_conflict_experience": False,
                    "update_when_experience_used": False,
                    "backup_experience_bank": False,
                }
            }

            pipeline = SelfEvolvingConflictPipeline(
                main_model=main_model,
                auxiliary_model=aux_model,
                experience_bank=bank,
                retriever=retriever,
                config=config,
            )
            yield pipeline

    def test_sunflower_conflict_resolution(self, pipeline):
        """Test Sunflower sample: should detect conflict and resolve to Sunflower."""
        sample = make_sample(
            instance="Sunflower",
            category="ER",
            question="What is this plant in the image?",
            image_paths=["image/Sunflower/Sunflower_1.jpg"],
            conflict_knowledge="This is an image of Marigold. The common marigold is a flowering plant.",
            groundtruth="Sunflower",
            sample_id=0,
        )

        result = pipeline.run_sample(sample)

        # Check pipeline mode
        assert result.pipeline_mode == "conflict_detected_and_resolved"

        # Check answer1 = Sunflower (from dummy model using entity_name)
        assert "Sunflower" in result.answer1_no_external

        # Check answer2 = Marigold (extracted from conflict_knowledge)
        assert "Marigold" in result.answer2_with_external

        # Check conflict detected
        assert result.conflict_detection["conflict_detected"] is True
        assert result.conflict_detection["conflict_type"] == "entity_recognition_conflict"

        # Check final answer = Sunflower (reconcile trusts answer1)
        assert "Sunflower" in result.final_answer

        # Check experience updated
        assert result.experience_updated is True

        # Check correct
        assert result.correct is True

        # Check status
        assert result.status == "success"

    def test_no_conflict_scenario(self, pipeline):
        """Test scenario where no conflict exists (same answer)."""
        sample = make_sample(
            instance="Sunflower",
            category="ER",
            question="What is this plant in the image?",
            image_paths=["image/Sunflower/Sunflower_1.jpg"],
            # Conflict knowledge that also mentions Sunflower
            conflict_knowledge="This is an image of Sunflower. The common sunflower is widely cultivated.",
            groundtruth="Sunflower",
            sample_id=1,
        )

        result = pipeline.run_sample(sample)

        # Both answers should be "Sunflower"
        assert "Sunflower" in result.answer1_no_external
        assert "Sunflower" in result.answer2_with_external

        # No conflict should be detected
        assert result.conflict_detection["conflict_detected"] is False
        assert result.pipeline_mode == "no_conflict"

        # Final answer should be Sunflower
        assert "Sunflower" in result.final_answer

        # No experience update by default for no-conflict
        assert result.experience_updated is False

    def test_experience_retrieval_on_second_run(self, pipeline):
        """Test that after processing, the second run uses experience retrieval."""
        sample = make_sample(
            instance="Sunflower",
            category="ER",
            question="What is this plant in the image?",
            image_paths=["image/Sunflower/Sunflower_1.jpg"],
            conflict_knowledge="This is an image of Marigold. The common marigold is a flowering plant.",
            groundtruth="Sunflower",
            sample_id=0,
        )

        # First run: should detect conflict and create experience
        result1 = pipeline.run_sample(sample)
        assert result1.pipeline_mode == "conflict_detected_and_resolved"
        assert result1.experience_updated is True

        # Second run (same sample): should retrieve the experience
        result2 = pipeline.run_sample(sample)
        assert result2.pipeline_mode == "experience_guided_inference"
        assert result2.retrieval_hit is True
        assert len(result2.retrieved_results) > 0
        # Final answer should still be Sunflower
        assert "Sunflower" in result2.final_answer

    def test_pipeline_result_schema(self, pipeline):
        """Test that the result contains all expected fields."""
        sample = make_sample(
            instance="Eiffel Tower",
            category="landmark",
            question="In what year was the structure in the image completed?",
            image_paths=["image/EiffelTower/eiffel_1.jpg"],
            conflict_knowledge="The Eiffel Tower was completed in 1890.",
            groundtruth="1889",
            sample_id=2,
        )

        result = pipeline.run_sample(sample)

        result_dict = result.to_dict()
        assert "ID" in result_dict
        assert "instance" in result_dict
        assert "question" in result_dict
        assert "answer1_no_external" in result_dict
        assert "answer2_with_external" in result_dict
        assert "conflict_detection" in result_dict
        assert "final_answer" in result_dict
        assert "pipeline_mode" in result_dict
        assert "experience_updated" in result_dict
        assert "correct" in result_dict
        assert "status" in result_dict

    def test_elons_musk_birth_year(self, pipeline):
        """Test Elon Musk birth year conflict resolution."""
        sample = make_sample(
            instance="Elon Musk",
            category="person",
            question="What year was the person in the image born?",
            image_paths=["image/ElonMusk/musk_1.jpg"],
            conflict_knowledge="Elon Musk was born in 1972. Some early biographies misreported this.",
            groundtruth="1971",
            sample_id=3,
        )

        result = pipeline.run_sample(sample)

        # Should detect conflict between 1971 and 1972
        assert result.conflict_detection["conflict_detected"] is True
        # Final answer should be 1971 (reconcile trusts answer1)
        assert "Elon Musk" in result.final_answer or "1971" in result.final_answer.split()[-1]

    def test_empty_retrieval_when_bank_empty(self):
        """Test that pipeline works when experience bank is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = str(Path(tmpdir) / "empty_bank.json")
            bank = ExperienceBank(bank_path=bank_path, backup=False)
            retriever = ExperienceRetriever(bank=bank, min_similarity=0.5)

            main_model = DummyMultimodalModel()
            main_model.load_model()
            aux_model = DummyAuxiliaryModel()

            config = {
                "pipeline": {
                    "final_answer_when_no_conflict": "with_external",
                    "save_no_conflict_experience": False,
                    "update_when_experience_used": False,
                    "backup_experience_bank": False,
                }
            }

            pipeline = SelfEvolvingConflictPipeline(
                main_model=main_model,
                auxiliary_model=aux_model,
                experience_bank=bank,
                retriever=retriever,
                config=config,
            )

            sample = make_sample(
                instance="Sunflower",
                category="ER",
                question="What is this plant in the image?",
                image_paths=["image/Sunflower/Sunflower_1.jpg"],
                conflict_knowledge="This is an image of Marigold.",
                groundtruth="Sunflower",
                sample_id=0,
            )

            result = pipeline.run_sample(sample)
            assert result.status == "success"
            assert result.pipeline_mode == "conflict_detected_and_resolved"
            assert result.experience_updated is True

            # Bank should now have one entity
            assert bank.get_statistics()["total_entities"] == 1
