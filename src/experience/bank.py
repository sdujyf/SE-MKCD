"""Experience Bank: persistent storage of conflict resolution experiences.

Structure:
{
    "<entity_name>": {
        "aliases": [...],
        "entity_type": "...",
        "knowledge_points": {
            "<knowledge_point>": {
                "question_patterns": [...],
                "experiences": [...]
            }
        }
    }
}
"""

import copy
import json
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ExperienceBank:
    """Persistent, self-evolving experience bank backed by a JSON file."""

    def __init__(self, bank_path: str, backup: bool = True):
        """Initialize the experience bank.

        Args:
            bank_path: Path to the experience bank JSON file.
            backup: Whether to create a .bak backup before each save.
        """
        self.bank_path = Path(bank_path)
        self.backup = backup
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load the experience bank from disk, or initialize empty."""
        if self.bank_path.exists():
            with open(self.bank_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            logger.info("Loaded experience bank from %s (%d entities)", self.bank_path, len(self._data))
        else:
            self._data = {}
            logger.info("Initialized empty experience bank at %s", self.bank_path)
            self._save()

    def _save(self) -> None:
        """Persist the experience bank to disk."""
        self.bank_path.parent.mkdir(parents=True, exist_ok=True)

        if self.backup and self.bank_path.exists():
            backup_path = self.bank_path.with_suffix(self.bank_path.suffix + ".bak")
            shutil.copy2(self.bank_path, backup_path)

        with open(self.bank_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get_entity(self, entity_name: str) -> Optional[dict]:
        """Get the data for a specific entity.

        Args:
            entity_name: The entity name (key in the bank).

        Returns:
            Entity data dict, or None if not found.
        """
        return self._data.get(entity_name)

    def entity_exists(self, entity_name: str) -> bool:
        """Check if an entity exists in the bank.

        Args:
            entity_name: The entity name.

        Returns:
            True if the entity exists.
        """
        return entity_name in self._data

    def get_knowledge_points(self, entity_name: str) -> dict[str, Any]:
        """Get all knowledge points for an entity.

        Args:
            entity_name: The entity name.

        Returns:
            Dictionary of knowledge_point -> data, or empty dict if entity not found.
        """
        entity = self._data.get(entity_name, {})
        return entity.get("knowledge_points", {})

    def add_experience(
        self,
        entity_name: str,
        entity_type: str,
        knowledge_point: str,
        question_patterns: list[str],
        experience: dict,
        aliases: Optional[list[str]] = None,
    ) -> None:
        """Add a new experience to the bank, creating entity/knowledge_point as needed.

        Args:
            entity_name: The entity name (key).
            entity_type: Type of entity (e.g., "person", "plant", "ER").
            knowledge_point: The knowledge point key (e.g., "birth_year").
            question_patterns: List of question patterns for matching.
            experience: The experience dict to append.
            aliases: Optional list of entity aliases.
        """
        # Ensure entity exists
        if entity_name not in self._data:
            self._data[entity_name] = {
                "aliases": aliases or [],
                "entity_type": entity_type,
                "knowledge_points": {},
            }
            logger.info("Created new entity: %s", entity_name)
        else:
            # Update entity_type and aliases if provided
            if entity_type and entity_type != "unknown":
                self._data[entity_name]["entity_type"] = entity_type
            if aliases:
                existing = set(self._data[entity_name].get("aliases", []))
                for a in aliases:
                    existing.add(a)
                self._data[entity_name]["aliases"] = sorted(existing)

        entity = self._data[entity_name]
        kps = entity["knowledge_points"]

        # Ensure knowledge point exists
        if knowledge_point not in kps:
            kps[knowledge_point] = {
                "question_patterns": [],
                "experiences": [],
            }
            logger.info("Created new knowledge_point '%s' for entity '%s'", knowledge_point, entity_name)

        kp = kps[knowledge_point]

        # Merge question patterns (deduplicate)
        existing_patterns = set(kp.get("question_patterns", []))
        for p in question_patterns:
            if p not in existing_patterns:
                kp["question_patterns"].append(p)
                existing_patterns.add(p)

        # Ensure experience has required fields
        experience.setdefault("created_at", str(date.today()))

        # Generate experience_id if not present
        if "experience_id" not in experience:
            safe_entity = entity_name.lower().replace(" ", "_").replace("/", "_")
            safe_kp = knowledge_point.lower().replace(" ", "_")
            count = len(kp["experiences"]) + 1
            experience["experience_id"] = f"{safe_entity}_{safe_kp}_{count:04d}"

        kp["experiences"].append(experience)
        logger.info(
            "Added experience '%s' to entity '%s' / knowledge_point '%s'",
            experience.get("experience_id"), entity_name, knowledge_point,
        )
        self._save()

    def get_all_entities(self) -> list[str]:
        """Get a list of all entity names in the bank.

        Returns:
            List of entity name strings.
        """
        return list(self._data.keys())

    def get_statistics(self) -> dict:
        """Get summary statistics about the experience bank.

        Returns:
            Dictionary with counts.
        """
        total_entities = len(self._data)
        total_kps = 0
        total_experiences = 0
        for entity in self._data.values():
            kps = entity.get("knowledge_points", {})
            total_kps += len(kps)
            for kp in kps.values():
                total_experiences += len(kp.get("experiences", []))
        return {
            "total_entities": total_entities,
            "total_knowledge_points": total_kps,
            "total_experiences": total_experiences,
        }

    def to_dict(self) -> dict:
        """Return a deep copy of the bank data."""
        return copy.deepcopy(self._data)
