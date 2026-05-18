"""Tests for the ExperienceBank class."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.experience.bank import ExperienceBank


class TestExperienceBank:
    """Test suite for ExperienceBank."""

    @pytest.fixture
    def temp_bank_path(self):
        """Create a temporary bank file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield str(Path(tmpdir) / "test_bank.json")

    def test_empty_bank_creation(self, temp_bank_path):
        """Test creating an empty experience bank."""
        bank = ExperienceBank(bank_path=temp_bank_path, backup=False)
        assert bank.get_all_entities() == []
        assert bank.get_statistics() == {
            "total_entities": 0,
            "total_knowledge_points": 0,
            "total_experiences": 0,
        }
        # Check file was created
        assert os.path.exists(temp_bank_path)

    def test_add_experience_new_entity(self, temp_bank_path):
        """Test adding an experience for a new entity."""
        bank = ExperienceBank(bank_path=temp_bank_path, backup=False)

        bank.add_experience(
            entity_name="Sunflower",
            entity_type="plant",
            knowledge_point="entity_recognition",
            question_patterns=["What is this plant in the image?"],
            experience={
                "question": "What is this plant in the image?",
                "answer1_no_external": "Sunflower",
                "answer2_with_external": "Marigold",
                "answer3_final": "Sunflower",
                "external_claim": "This is an image of Marigold.",
                "conflict_detected": True,
                "conflict_type": "entity_recognition_conflict",
                "final_decision": "Sunflower",
                "decision_source": "visual_evidence_preferred",
                "external_claim_status": "rejected",
                "summary": "Resolved conflict.",
            },
        )

        # Check entity exists
        assert bank.entity_exists("Sunflower")

        # Check stats
        stats = bank.get_statistics()
        assert stats["total_entities"] == 1
        assert stats["total_knowledge_points"] == 1
        assert stats["total_experiences"] == 1

        # Check entity data
        entity = bank.get_entity("Sunflower")
        assert entity is not None
        assert entity["entity_type"] == "plant"
        assert "entity_recognition" in entity["knowledge_points"]

        kp = entity["knowledge_points"]["entity_recognition"]
        assert len(kp["experiences"]) == 1
        assert kp["experiences"][0]["answer3_final"] == "Sunflower"

    def test_add_experience_existing_entity(self, temp_bank_path):
        """Test adding a second experience to an existing entity."""
        bank = ExperienceBank(bank_path=temp_bank_path, backup=False)

        # First experience
        bank.add_experience(
            entity_name="Elon Musk",
            entity_type="person",
            knowledge_point="birth_year",
            question_patterns=["What year was Elon Musk born?"],
            experience={
                "question": "What year was Elon Musk born?",
                "answer1_no_external": "1971",
                "answer2_with_external": "1972",
                "answer3_final": "1971",
                "external_claim": "Born in 1972.",
                "conflict_detected": True,
                "conflict_type": "attribute_value_conflict",
                "final_decision": "1971",
                "decision_source": "verified_entity_knowledge",
                "external_claim_status": "rejected",
                "summary": "Conflict resolved.",
            },
        )

        # Second experience - same entity, new knowledge_point
        bank.add_experience(
            entity_name="Elon Musk",
            entity_type="person",
            knowledge_point="nationality",
            question_patterns=["What is Elon Musk's nationality?"],
            experience={
                "question": "What is Elon Musk's nationality?",
                "answer1_no_external": "South African",
                "answer2_with_external": "American",
                "answer3_final": "South African",
                "external_claim": "Elon Musk is American.",
                "conflict_detected": True,
                "conflict_type": "attribute_value_conflict",
                "final_decision": "South African",
                "decision_source": "verified_entity_knowledge",
                "external_claim_status": "rejected",
                "summary": "Conflict resolved.",
            },
        )

        # Check stats
        stats = bank.get_statistics()
        assert stats["total_entities"] == 1
        assert stats["total_knowledge_points"] == 2
        assert stats["total_experiences"] == 2

        # Check both knowledge points
        entity = bank.get_entity("Elon Musk")
        assert "birth_year" in entity["knowledge_points"]
        assert "nationality" in entity["knowledge_points"]

    def test_save_and_reload(self, temp_bank_path):
        """Test that data persists after save and reload."""
        bank = ExperienceBank(bank_path=temp_bank_path, backup=False)
        bank.add_experience(
            entity_name="Eiffel Tower",
            entity_type="landmark",
            knowledge_point="completion_year",
            question_patterns=["When was the Eiffel Tower completed?"],
            experience={
                "question": "When was the Eiffel Tower completed?",
                "answer1_no_external": "1889",
                "answer2_with_external": "1890",
                "answer3_final": "1889",
                "external_claim": "Completed in 1890.",
                "conflict_detected": True,
                "conflict_type": "attribute_value_conflict",
                "final_decision": "1889",
                "decision_source": "verified_entity_knowledge",
                "external_claim_status": "rejected",
                "summary": "Conflict resolved.",
            },
        )

        # Reload from disk
        bank2 = ExperienceBank(bank_path=temp_bank_path, backup=False)
        assert bank2.entity_exists("Eiffel Tower")

        entity = bank2.get_entity("Eiffel Tower")
        assert entity["entity_type"] == "landmark"
        assert len(entity["knowledge_points"]["completion_year"]["experiences"]) == 1

    def test_add_experience_append_question_patterns(self, temp_bank_path):
        """Test that new question patterns are appended without duplicates."""
        bank = ExperienceBank(bank_path=temp_bank_path, backup=False)

        bank.add_experience(
            entity_name="Sunflower",
            entity_type="plant",
            knowledge_point="identification",
            question_patterns=["What is this plant?"],
            experience={
                "question": "What is this plant?",
                "answer1_no_external": "Sunflower",
                "answer2_with_external": "Marigold",
                "answer3_final": "Sunflower",
                "external_claim": "It's a Marigold.",
                "conflict_detected": True,
                "conflict_type": "entity_recognition_conflict",
                "final_decision": "Sunflower",
                "decision_source": "visual_evidence_preferred",
                "external_claim_status": "rejected",
                "summary": "Resolved.",
            },
        )

        # Add second experience with same question_patterns
        bank.add_experience(
            entity_name="Sunflower",
            entity_type="plant",
            knowledge_point="identification",
            question_patterns=["What is this plant?", "Which plant is shown?"],
            experience={
                "question": "Which plant is shown?",
                "answer1_no_external": "Sunflower",
                "answer2_with_external": "Dandelion",
                "answer3_final": "Sunflower",
                "external_claim": "It's a Dandelion.",
                "conflict_detected": True,
                "conflict_type": "entity_recognition_conflict",
                "final_decision": "Sunflower",
                "decision_source": "visual_evidence_preferred",
                "external_claim_status": "rejected",
                "summary": "Resolved.",
            },
        )

        kp = bank.get_entity("Sunflower")["knowledge_points"]["identification"]
        patterns = kp["question_patterns"]
        assert "What is this plant?" in patterns
        assert "Which plant is shown?" in patterns
        assert len(patterns) == 2  # No duplicate
        assert len(kp["experiences"]) == 2

    def test_experience_id_auto_generation(self, temp_bank_path):
        """Test that experience_id is auto-generated if not provided."""
        bank = ExperienceBank(bank_path=temp_bank_path, backup=False)

        bank.add_experience(
            entity_name="Test Entity",
            entity_type="test",
            knowledge_point="test_kp",
            question_patterns=["test question?"],
            experience={
                "question": "test question?",
                "answer1_no_external": "A",
                "answer2_with_external": "B",
                "answer3_final": "A",
                "external_claim": "B is correct.",
                "conflict_detected": True,
                "conflict_type": "factual_contradiction",
                "final_decision": "A",
                "decision_source": "verified_entity_knowledge",
                "external_claim_status": "rejected",
                "summary": "Resolved.",
            },
        )

        kp = bank.get_entity("Test Entity")["knowledge_points"]["test_kp"]
        exp = kp["experiences"][0]
        assert "experience_id" in exp
        assert "test_entity_test_kp_0001" == exp["experience_id"]
        assert "created_at" in exp
