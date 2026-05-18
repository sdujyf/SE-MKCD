"""Tests for the ExperienceRetriever class."""

import tempfile
from pathlib import Path

import pytest

from src.experience.bank import ExperienceBank
from src.experience.retriever import ExperienceRetriever


class TestExperienceRetriever:
    """Test suite for ExperienceRetriever."""

    @pytest.fixture
    def bank_with_data(self):
        """Create an experience bank with pre-loaded test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = str(Path(tmpdir) / "test_bank.json")
            bank = ExperienceBank(bank_path=bank_path, backup=False)

            # Add Elon Musk birth_year experience
            bank.add_experience(
                entity_name="Elon Musk",
                entity_type="person",
                aliases=["Musk", "Elon Reeve Musk"],
                knowledge_point="birth_year",
                question_patterns=[
                    "What year was the person in the image born?",
                    "When was Elon Musk born?",
                    "What is Elon Musk's birth year?",
                ],
                experience={
                    "experience_id": "elon_musk_birth_year_0001",
                    "question": "What year was Elon Musk born?",
                    "answer1_no_external": "1971",
                    "answer2_with_external": "1972",
                    "answer3_final": "1971",
                    "external_claim": "Elon Musk was born in 1972.",
                    "conflict_detected": True,
                    "conflict_type": "attribute_value_conflict",
                    "final_decision": "1971",
                    "decision_source": "verified_entity_knowledge",
                    "external_claim_status": "rejected",
                    "summary": "Conflict resolved: retained 1971.",
                    "created_at": "2026-05-16",
                },
            )

            # Add another knowledge point
            bank.add_experience(
                entity_name="Elon Musk",
                entity_type="person",
                knowledge_point="nationality",
                question_patterns=["What nationality is Elon Musk?"],
                experience={
                    "experience_id": "elon_musk_nationality_0001",
                    "question": "What nationality is Elon Musk?",
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
                    "created_at": "2026-05-16",
                },
            )

            # Add Sunflower entity
            bank.add_experience(
                entity_name="Sunflower",
                entity_type="plant",
                aliases=["Helianthus", "common sunflower"],
                knowledge_point="entity_recognition",
                question_patterns=["What is this plant in the image?"],
                experience={
                    "experience_id": "sunflower_entity_recognition_0001",
                    "question": "What is this plant in the image?",
                    "answer1_no_external": "Sunflower",
                    "answer2_with_external": "Marigold",
                    "answer3_final": "Sunflower",
                    "external_claim": "This is Marigold.",
                    "conflict_detected": True,
                    "conflict_type": "entity_recognition_conflict",
                    "final_decision": "Sunflower",
                    "decision_source": "visual_evidence_preferred",
                    "external_claim_status": "rejected",
                    "summary": "Conflict resolved.",
                    "created_at": "2026-05-16",
                },
            )

            yield bank

    def test_retrieve_by_exact_entity_and_question(self, bank_with_data):
        """Test retrieval with exact entity name and exact question match."""
        retriever = ExperienceRetriever(bank=bank_with_data, min_similarity=0.5)

        results = retriever.retrieve(
            entity_name="Elon Musk",
            question="When was Elon Musk born?",
        )

        assert len(results) > 0
        assert results[0]["knowledge_point"] == "birth_year"
        assert results[0]["score"] >= 0.9
        assert len(results[0]["experiences"]) == 1
        assert results[0]["experiences"][0]["answer3_final"] == "1971"

    def test_retrieve_by_alias(self, bank_with_data):
        """Test retrieval using an entity alias."""
        retriever = ExperienceRetriever(bank=bank_with_data, min_similarity=0.5)

        results = retriever.retrieve(
            entity_name="Musk",
            question="When was Elon Musk born?",
        )

        assert len(results) > 0
        assert results[0]["matched_entity_key"] == "Elon Musk"
        assert results[0]["knowledge_point"] == "birth_year"

    def test_retrieve_substring_match(self, bank_with_data):
        """Test retrieval with a substring-matched question."""
        retriever = ExperienceRetriever(bank=bank_with_data, min_similarity=0.5)

        results = retriever.retrieve(
            entity_name="Elon Musk",
            question="What year was Elon Musk born?",
        )

        assert len(results) > 0
        assert results[0]["knowledge_point"] == "birth_year"

    def test_retrieve_token_overlap(self, bank_with_data):
        """Test retrieval based on token overlap score."""
        retriever = ExperienceRetriever(bank=bank_with_data, min_similarity=0.3)

        results = retriever.retrieve(
            entity_name="Elon Musk",
            question="When was this person born?",
        )

        # Should match birth_year via token overlap ("born", "was")
        assert len(results) > 0
        assert results[0]["knowledge_point"] == "birth_year"

    def test_retrieve_entity_not_found(self, bank_with_data):
        """Test that retrieval returns empty for unknown entity."""
        retriever = ExperienceRetriever(bank=bank_with_data, min_similarity=0.5)

        results = retriever.retrieve(
            entity_name="Bill Gates",
            question="When was Bill Gates born?",
        )

        assert len(results) == 0

    def test_retrieve_no_matching_question(self, bank_with_data):
        """Test that retrieval returns empty when question doesn't match."""
        retriever = ExperienceRetriever(bank=bank_with_data, min_similarity=0.9)

        results = retriever.retrieve(
            entity_name="Elon Musk",
            question="What is Elon Musk's favorite food?",
        )

        assert len(results) == 0

    def test_retrieve_sunflower(self, bank_with_data):
        """Test retrieval for Sunflower entity."""
        retriever = ExperienceRetriever(bank=bank_with_data, min_similarity=0.5)

        results = retriever.retrieve(
            entity_name="Sunflower",
            question="What is this plant in the image?",
        )

        assert len(results) > 0
        assert results[0]["knowledge_point"] == "entity_recognition"
        assert results[0]["score"] == 1.0
        assert len(results[0]["experiences"]) == 1

    def test_retrieve_with_alias_helianthus(self, bank_with_data):
        """Test retrieval using the Helianthus alias."""
        retriever = ExperienceRetriever(bank=bank_with_data, min_similarity=0.5)

        results = retriever.retrieve(
            entity_name="Helianthus",
            question="What is this plant in the image?",
        )

        assert len(results) > 0
        assert results[0]["matched_entity_key"] == "Sunflower"
