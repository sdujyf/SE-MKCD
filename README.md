# Self-Evolving Multimodal Knowledge Conflict Detection and Resolution Framework

A structured, extensible research framework for detecting and resolving knowledge conflicts in multimodal (image + text) settings, with a self-evolving experience bank that learns from past conflict resolutions.

## Pipeline Overview

```
Input Sample (image + question + external document)
       │
       ▼
┌──────────────────────┐
│ 1. Entity & Question │  Extract entity name and question from sample
│    Identification    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 2. Experience        │  Search experience bank for matching entity + knowledge point
│    Retrieval         │
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     │  Hit?     │
     └─────┬─────┘
      Yes  │  No
           │
  ┌────────┴────────┐
  │                 │
  ▼                 ▼
┌──────────┐   ┌──────────────────────┐
│ 3a.      │   │ 3b. Generate answers │
│ Exp-Guided│   │  answer1 (no external)│
│ Inference│   │  answer2 (with ext.)  │
└──────────┘   └──────────┬───────────┘
                          │
                          ▼
                 ┌──────────────────────┐
                 │ 4. Conflict Detection│
                 └──────────┬───────────┘
                            │
                      ┌─────┴─────┐
                      │ Conflict? │
                      └─────┬─────┘
                       Yes  │  No
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
       ┌──────────────┐          ┌──────────────┐
       │ 5a. Summarize│          │ 5b. Use      │
       │  conflict    │          │  configured   │
       │  + Reconcile │          │  answer       │
       └──────┬───────┘          └──────────────┘
              │
              ▼
       ┌──────────────┐
       │ 6. Update    │
       │  Experience  │
       │  Bank        │
       └──────────────┘
              │
              ▼
       ┌──────────────┐
       │ 7. Evaluate  │
       │  + Output    │
       └──────────────┘
```

When experiences are retrieved (left branch), the model uses historical resolution patterns to guide inference directly - skipping the full conflict detection and reconciliation process.

## Data Format

Input JSON file (list of samples):

```json
{
    "instance": "Sunflower",
    "category": "ER",
    "knowledge": "This is an image of Sunflower...",
    "image_path": ["image/Sunflower/Sunflower_1.jpg"],
    "conflict_knowledge": "This is an image of Marigold...",
    "open_query": "What is this plant in the image?",
    "open_groundtruth": "Sunflower",
    "conflict_openanswer": "Marigold",
    "mcq_query": "What is this plant? A) Sunflower B) Dandelion C) Black-eyed Susan D) Marigold",
    "mcq_groundtruth": "A",
    "conflict_mcqanswer": "D",
    "ID": 0
}
```

- `image_path`: string or list of strings
- `conflict_knowledge`: external document that may conflict with visual evidence

## Experience Bank Format

```json
{
    "Elon Musk": {
        "aliases": ["Musk", "Elon Reeve Musk"],
        "entity_type": "person",
        "knowledge_points": {
            "birth_year": {
                "question_patterns": [
                    "When was Elon Musk born?",
                    "What year was the person born?"
                ],
                "experiences": [
                    {
                        "experience_id": "elon_musk_birth_year_0001",
                        "question": "...",
                        "answer1_no_external": "1971",
                        "answer2_with_external": "1972",
                        "answer3_final": "1971",
                        "external_claim": "...",
                        "conflict_detected": true,
                        "conflict_type": "attribute_value_conflict",
                        "final_decision": "1971",
                        "decision_source": "verified_entity_knowledge",
                        "external_claim_status": "rejected",
                        "summary": "...",
                        "created_at": "2026-05-16"
                    }
                ]
            }
        }
    }
}
```

Three-level structure: **Entity → Knowledge Point → Experiences**

## Installation

```bash
# Basic dependencies (for dummy mode)
pip install -r requirements.txt

# Optional: for real VLM inference
pip install torch transformers pillow

# Optional: for semantic retrieval
pip install sentence-transformers
```

## Quick Start (Dummy Mode)

Run the full pipeline using dummy models (no GPU or model downloads needed):

```bash
python main.py \
  --config configs/default.yaml \
  --data_path data/examples.json \
  --image_root ./ \
  --output_path outputs/results.jsonl \
  --max_samples 100
```

This will:
1. Load the example dataset
2. Process each sample with dummy models (simulated conflict resolution)
3. Write results to `outputs/results.jsonl`
4. Update `experience_bank/experience_bank.json` with new experiences
5. Print a summary with accuracy, conflict stats, etc.

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Or individual test files:
```bash
python -m pytest tests/test_experience_bank.py -v
python -m pytest tests/test_retriever.py -v
python -m pytest tests/test_pipeline_dummy.py -v
```

## Using Real HuggingFace VLM

Edit `configs/default.yaml`:

```yaml
models:
  main_model:
    type: "huggingface_vlm"
    model_name_or_path: "Qwen/Qwen2.5-VL-7B-Instruct"
    device: "cuda"
    torch_dtype: "bfloat16"
  auxiliary_model:
    type: "llm_judge"
    model_name_or_path: "gpt-4"
```

Or via CLI override:
```bash
python main.py --config configs/hf_vlm.yaml --data_path data/examples.json
```

## Output Format

Results are written as JSONL (one JSON object per line). Each result contains:

- `ID`, `instance`, `category`, `question`
- `answer1_no_external`, `answer2_with_external`
- `conflict_detection` (conflict_detected, conflict_type, explanation)
- `conflict_summary` (summary, key_disagreement, recommended_strategy)
- `final_answer`, `final_reason`
- `pipeline_mode`: "experience_guided_inference" / "no_conflict" / "conflict_detected_and_resolved"
- `experience_updated`, `retrieval_hit`, `correct`, `status`

The pipeline supports **resume**: if `output_path` already exists, completed IDs are skipped.

## Extending the Framework

### Stronger Retriever
Enable semantic retrieval in `configs/default.yaml`:
```yaml
retrieval:
  use_sentence_transformers: true
  sentence_transformer_model: "all-MiniLM-L6-v2"
```

Or replace `ExperienceRetriever` with a custom class implementing the same interface.

### Stronger Conflict Detector
Implement the `BaseAuxiliaryModel` interface with an LLM-based judge. The `LLMJudgeAuxiliaryModel` class provides a scaffold — implement `_call_openai()`, `_call_anthropic()`, or `_call_local()` based on your setup.

### Stronger Experience Summarizer
Customize the `summarize_experience()` method in your auxiliary model for more sophisticated experience distillation.

### MCQ Support
Extend the pipeline to handle multiple-choice questions via the `mcq_query` and `mcq_groundtruth` fields already present in the data schema.

### Multimodal Evidence Attribution
Add an evidence attribution module that traces which parts of the image or external document contributed to the final decision.

## Project Structure

```
Self-Evolving-Multimodal-Knowledge-Conflict/
├── main.py                     # Entry point
├── requirements.txt
├── configs/default.yaml        # Configuration
├── data/examples.json          # Example dataset
├── experience_bank/            # Persisted experiences
├── outputs/                    # Results output
├── src/
│   ├── data/                   # Data loading & schemas
│   ├── experience/             # Experience bank & retriever
│   ├── models/                 # Main multimodal models
│   ├── auxiliary/              # Auxiliary models (conflict detection)
│   ├── pipeline/               # Main pipeline orchestration
│   └── utils/                  # IO, logging, text, metrics
└── tests/                      # Unit & integration tests
```
