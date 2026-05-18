"""Experience Retriever: matches queries to stored experiences."""

import logging
from typing import Optional

from .bank import ExperienceBank
from .utils import normalize_text, compute_token_overlap, match_entity_name

logger = logging.getLogger(__name__)


class ExperienceRetriever:
    """Retrieves relevant experiences from the ExperienceBank for a given entity and question.

    Supports multiple matching strategies: exact, substring, and token overlap.
    Can be extended with sentence-transformer based semantic matching.
    """

    def __init__(
        self,
        bank: ExperienceBank,
        min_similarity: float = 0.5,
        use_sentence_transformers: bool = False,
        sentence_transformer_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize the retriever.

        Args:
            bank: The ExperienceBank instance to search.
            min_similarity: Minimum similarity threshold for a match (0.0-1.0).
            use_sentence_transformers: If True, use embedding-based retrieval.
            sentence_transformer_model: SentenceTransformer model name.
        """
        self.bank = bank
        self.min_similarity = min_similarity
        self.use_sentence_transformers = use_sentence_transformers
        self._embedder = None

        if use_sentence_transformers:
            self._init_embedder(sentence_transformer_model)

    def _init_embedder(self, model_name: str) -> None:
        """Initialize sentence-transformer embedder.

        Args:
            model_name: Name of the sentence-transformer model.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(model_name)
            logger.info("Initialized SentenceTransformer: %s", model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed. Falling back to lexical matching.")
            self.use_sentence_transformers = False
        except Exception as e:
            logger.warning("Failed to load SentenceTransformer: %s. Falling back to lexical matching.", e)
            self.use_sentence_transformers = False

    def retrieve(self, entity_name: str, question: str) -> list[dict]:
        """Retrieve matching experiences for an entity and question.

        Args:
            entity_name: The entity name to search for.
            question: The question text to match against knowledge points.

        Returns:
            List of match result dicts, each containing:
                - entity_name
                - matched_entity_key
                - knowledge_point
                - score
                - question_patterns
                - experiences
        """
        results = []

        # Step 1: Find matching entity
        matched_key = self._find_entity(entity_name)
        if matched_key is None:
            logger.debug("No entity match for '%s'", entity_name)
            return results

        entity_data = self.bank.get_entity(matched_key)
        if entity_data is None:
            return results

        kps = entity_data.get("knowledge_points", {})

        # Step 2: For each knowledge point, match question against question_patterns
        for kp_name, kp_data in kps.items():
            question_patterns = kp_data.get("question_patterns", [])
            experiences = kp_data.get("experiences", [])

            if not experiences:
                continue

            score = self._match_question(question, question_patterns)
            if score >= self.min_similarity:
                results.append({
                    "entity_name": entity_name,
                    "matched_entity_key": matched_key,
                    "knowledge_point": kp_name,
                    "score": score,
                    "question_patterns": question_patterns,
                    "experiences": experiences,
                })

        # Sort by score descending
        results.sort(key=lambda r: r["score"], reverse=True)
        logger.debug("Retrieved %d matches for entity='%s', question='%s'", len(results), entity_name, question)
        return results

    def _find_entity(self, entity_name: str) -> Optional[str]:
        """Find the matching entity key in the bank.

        Args:
            entity_name: The entity name to search for.

        Returns:
            The matching entity key, or None if no match.
        """
        query_norm = normalize_text(entity_name)

        for entity_key, entity_data in self.bank._data.items():
            aliases = entity_data.get("aliases", [])
            if match_entity_name(query_norm, entity_key, aliases):
                return entity_key

        return None

    def _match_question(self, question: str, patterns: list[str]) -> float:
        """Match a question against a list of question patterns.

        Tries exact match, then substring, then token overlap.

        Args:
            question: The query question.
            patterns: List of known question patterns.

        Returns:
            Best match score (0.0-1.0).
        """
        if not patterns:
            return 0.0

        q_norm = normalize_text(question)
        best_score = 0.0

        for pattern in patterns:
            p_norm = normalize_text(pattern)

            # Exact match
            if q_norm == p_norm:
                return 1.0

            # Substring match
            if q_norm in p_norm or p_norm in q_norm:
                best_score = max(best_score, 0.9)

            # Token overlap
            overlap = compute_token_overlap(q_norm, p_norm, normalize=False)
            best_score = max(best_score, overlap)

        return best_score
