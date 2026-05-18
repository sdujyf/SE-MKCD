"""Main entry point for the Self-Evolving Multimodal Knowledge Conflict Detection and Resolution Framework.

Usage:
    python main.py \\
      --config configs/default.yaml \\
      --data_path data/examples.json \\
      --image_root ./ \\
      --output_path outputs/results.jsonl \\
      --max_samples 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

from src.utils.io import (
    load_yaml_config,
    read_json,
    write_jsonl_line,
    read_completed_ids,
)
from src.utils.logging_utils import setup_logging
from src.utils.metrics import compute_summary
from src.data.dataset import load_dataset
from src.data.schema import SampleSchema, PipelineResultSchema
from src.experience.bank import ExperienceBank
from src.experience.retriever import ExperienceRetriever
from src.models.dummy_model import DummyMultimodalModel
from src.models.huggingface_vlm import HuggingFaceVLMModel
from src.auxiliary.dummy_auxiliary import DummyAuxiliaryModel
from src.auxiliary.llm_judge import LLMJudgeAuxiliaryModel
from src.pipeline.self_evolving_pipeline import SelfEvolvingConflictPipeline

logger = logging.getLogger(__name__)


def build_models(config: dict):
    """Build main and auxiliary models from configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        Tuple of (main_model, auxiliary_model).
    """
    model_cfg = config.get("models", {})

    # Main model
    main_cfg = model_cfg.get("main_model", {})
    main_type = main_cfg.get("type", "dummy")

    if main_type == "dummy":
        main_model = DummyMultimodalModel(
            model_name_or_path=main_cfg.get("model_name_or_path", ""),
            device=main_cfg.get("device", "cpu"),
        )
        main_model.load_model()
        logger.info("Using DummyMultimodalModel as main model.")
    elif main_type == "huggingface_vlm":
        main_model = HuggingFaceVLMModel(
            model_name_or_path=main_cfg.get("model_name_or_path", ""),
            device=main_cfg.get("device", "cuda"),
            torch_dtype=main_cfg.get("torch_dtype", "auto"),
        )
        main_model.load_model()
        logger.info("Using HuggingFaceVLMModel as main model.")
    else:
        raise ValueError(f"Unknown main model type: {main_type}")

    # Auxiliary model
    aux_cfg = model_cfg.get("auxiliary_model", {})
    aux_type = aux_cfg.get("type", "dummy")

    if aux_type == "dummy":
        aux_model = DummyAuxiliaryModel(
            model_name_or_path=aux_cfg.get("model_name_or_path", ""),
        )
        logger.info("Using DummyAuxiliaryModel as auxiliary model.")
    elif aux_type == "llm_judge":
        aux_model = LLMJudgeAuxiliaryModel(
            model_name_or_path=aux_cfg.get("model_name_or_path", ""),
        )
        logger.info("Using LLMJudgeAuxiliaryModel as auxiliary model.")
    else:
        raise ValueError(f"Unknown auxiliary model type: {aux_type}")
    return main_model, aux_model


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Load config
    config = load_yaml_config(args.config)
    # Set seed
    seed = config.get("project", {}).get("seed", 42)
    import random
    random.seed(seed)

    # Override from CLI
    if args.data_path:
        config["_data_path"] = args.data_path
    else:
        config["_data_path"] = config.get("data", {}).get("data_path", "data/examples.json")

    if args.image_root:
        config["data"]["image_root"] = args.image_root

    if args.max_samples is not None:
        config["_max_samples"] = args.max_samples
    else:
        config["_max_samples"] = config.get("data", {}).get("max_samples", None)

    output_path = args.output_path or config.get("paths", {}).get("output_path", "outputs/results.jsonl")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "{timestamp}" in output_path:
        output_path = output_path.format(timestamp=timestamp)
    output_path = str(Path(output_path))

    # Experience bank
    bank_path = config.get("paths", {}).get("experience_bank_path", "experience_bank/experience_bank.json")
    backup = config.get("pipeline", {}).get("backup_experience_bank", True)
    experience_bank = ExperienceBank(bank_path=bank_path, backup=backup)

    stats = experience_bank.get_statistics()
    logger.info("Experience bank stats: %s", stats)

    # Retriever
    retrieval_cfg = config.get("retrieval", {})
    retriever = ExperienceRetriever(
        bank=experience_bank,
        min_similarity=retrieval_cfg.get("min_similarity", 0.5),
        use_sentence_transformers=retrieval_cfg.get("use_sentence_transformers", False),
        sentence_transformer_model=retrieval_cfg.get("sentence_transformer_model", "all-MiniLM-L6-v2"),
    )

    # Models
    main_model, aux_model = build_models(config)
    # Pipeline
    pipeline = SelfEvolvingConflictPipeline(
        main_model=main_model,
        auxiliary_model=aux_model,
        experience_bank=experience_bank,
        retriever=retriever,
        config=config,
    )

    # Load data
    data_path = config["_data_path"]
    max_samples = config.get("_max_samples")
    image_root = config.get("data", {}).get("image_root", "./")
    max_images = config.get("data", {}).get("max_images", 2)

    samples = load_dataset(
        data_path=data_path,
        image_root=image_root,
        max_images=max_images,
        max_samples=max_samples,
    )
    logger.info("Loaded %d samples.", len(samples))

    # Resumability
    completed_ids = read_completed_ids(output_path)
    pending_samples = [s for s in samples if s.ID not in completed_ids]
    logger.info(
        "Total: %d, Completed: %d, Pending: %d",
        len(samples), len(completed_ids), len(pending_samples),
    )

    if not pending_samples:
        logger.info("All samples already processed. Nothing to do.")
        return

    # Run pipeline with tqdm
    try:
        from tqdm import tqdm
        progress = tqdm(pending_samples, desc="Processing samples", unit="sample")
    except ImportError:
        progress = pending_samples
        logger.warning("tqdm not installed. Install with: pip install tqdm")

    for sample in progress:
        result: PipelineResultSchema = pipeline.run_sample(sample)
        result_dict = result.to_dict()
        write_jsonl_line(result_dict, output_path)

        if hasattr(progress, "set_postfix"):
            progress.set_postfix({
                "ID": result.ID,
                "mode": result.pipeline_mode[:10] if result.pipeline_mode else "N/A",
                "correct": result.correct,
            })

    # Final summary
    logger.info("=" * 60)
    logger.info("Pipeline completed. Output saved to: %s", output_path)

    all_results = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    summary = compute_summary(all_results)
    logger.info("Summary:")
    logger.info("  Total samples:          %d", summary["total"])
    logger.info("  Successful:             %d", summary["success"])
    logger.info("  Failed:                 %d", summary["failed"])
    logger.info("  Retrieval hits:         %d", summary["retrieval_hit_count"])
    logger.info("  Conflicts detected:     %d", summary["conflict_detected_count"])
    logger.info("  Experiences updated:    %d", summary["experience_updated_count"])
    if "accuracy" in summary:
        logger.info("  Accuracy:               %.2f%% (%d/%d)",
                    summary["accuracy"] * 100,
                    summary["correct_count"],
                    summary["total_with_groundtruth"])

    final_stats = experience_bank.get_statistics()
    logger.info("Final experience bank stats: %s", final_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Self-Evolving Multimodal Knowledge Conflict Detection and Resolution Framework"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to input JSON dataset.",
    )
    parser.add_argument(
        "--image_root", type=str, default=None,
        help="Root directory for resolving relative image paths.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Path for output JSONL results file.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to process.",
    )

    args = parser.parse_args()

    # Setup logging from config
    try:
        config = load_yaml_config(args.config)
        log_level = config.get("logging", {}).get("level", "INFO")
    except Exception:
        log_level = "INFO"

    setup_logging(level=log_level)

    logger.info("=" * 60)
    logger.info("Self-Evolving Multimodal Knowledge Conflict Detection and Resolution")
    logger.info("=" * 60)

    run_pipeline(args)


if __name__ == "__main__":
    main()
