"""HuggingFace VLM model wrapper for multimodal inference.

Supports Qwen2.5-VL, Qwen-VL, and similar multimodal models from HuggingFace.

NOTE: This module requires `torch` and `transformers` to be installed.
If dependencies are missing, the class will raise a clear error at initialization.
"""

import json
import logging
from typing import Any, Optional
import os
from .base import (
    BaseMultimodalModel,
    PROMPT_NO_EXTERNAL,
    PROMPT_WITH_EXTERNAL,
    PROMPT_WITH_EXPERIENCE,
    PROMPT_RECONCILE,
)
import torch
logger = logging.getLogger(__name__)
def build_image_path(image_root, image_path):
    """
    构造图片完整路径。
    如果 image_path 已经是绝对路径，则直接返回；
    如果是相对路径，则拼接 image_root。
    """
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(image_root, image_path)

class HuggingFaceVLMModel(BaseMultimodalModel):
    """Multimodal model backed by a HuggingFace VLM (e.g., Qwen2.5-VL).

    Usage:
        model = HuggingFaceVLMModel(
            model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            device="cuda",
            torch_dtype="bfloat16",
        )
        model.load_model()
        result = model.generate_no_external(["image.jpg"], "What is this?")
    """

    def __init__(
        self,
        model_name_or_path: str = "",
        device: str = "cuda",
        torch_dtype: str = "auto",
        **kwargs: Any,
    ):
        """Initialize the HF VLM model.

        Args:
            model_name_or_path: HF model ID or local path.
            device: Device string ("cuda", "cpu", "mps").
            torch_dtype: Torch dtype string ("auto", "float16", "bfloat16").
            **kwargs: Additional arguments.
        """
        super().__init__(model_name_or_path=model_name_or_path, device=device, **kwargs)
        self.torch_dtype_str = torch_dtype
        self._loaded = False

    def load_model(self) -> None:
        """Load the model and processor from HuggingFace.

        Raises:
            ImportError: If torch or transformers is not installed.
            RuntimeError: If model loading fails.
        """
        if self._loaded:
            return

        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for HuggingFaceVLMModel. "
                "Install it with: pip install torch"
            )

        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFaceVLMModel. "
                "Install it with: pip install transformers"
            )

        if not self.model_name_or_path:
            raise ValueError(
                "model_name_or_path must be set for HuggingFaceVLMModel. "
                "Example: 'Qwen/Qwen2.5-VL-7B-Instruct'"
            )

        logger.info("Loading model from %s ...", self.model_name_or_path)

        try:
            dtype_map = {
                "auto": "auto",
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.torch_dtype_str, "auto")

            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
            )
            self._processor = AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
            )
            self._loaded = True
            logger.info("Model loaded successfully on %s", self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_name_or_path}: {e}")

    def _generate_response(self, prompt: str, image_paths: list[str],mood:str) -> str:
        """Generate a raw text response from the model.

        Args:
            prompt: The formatted prompt text.
            image_paths: List of image file paths.

        Returns:
            Raw model output string.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError(
                "Model is not loaded. Call load_model() first, or use DummyMultimodalModel for testing."
            )

        from PIL import Image
        base_image_path =''
        
        # Load images
        images = []
        for img_path in image_paths:
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except FileNotFoundError:
                logger.warning("Image not found: %s, skipping.", img_path)
            except Exception as e:
                logger.warning("Failed to load image %s: %s", img_path, e)

        if not images:
            raise RuntimeError("No valid images could be loaded.")

        # Build messages in Qwen2.5-VL format
        content = []
        if mood =="1":
            content.append(
            {
            "type": "text",
            "text ": (
            "Answer the following question based on the image content.\n"
            "You must strictly follow this output format:\n"
            "Answer: <your answer>\n"
            "Reason: <brief explanation>\n\n")
            })
            img1_path = build_image_path(base_image_path, image_paths[0])
            content.append({
            "type": "image",
            "image": img1_path
            })
            content.append(
                {"type":"text",
                "text":prompt
                }
            )

        if mood =="2":
            content.append(
            {
            "type": "text",
            "text ": (
            "Answer the following question based on the image content and external document.\n"
            "You must strictly follow this output format:\n"
            "Answer: <your answer>\n"
            "Reason: <brief explanation>\n\n")
            })
            img1_path = build_image_path(base_image_path, image_paths[0])
            content.append({
            "type": "image",
            "image": img1_path
            })
            img2_path = build_image_path(base_image_path, image_paths[1])
            content.append({
            "type": "image",
            "image": img2_path
            })
            content.append(
                {"type":"text",
                "text":prompt
                }
            )
        if mood =="3":
            img1_path = build_image_path(base_image_path, image_paths[0])
            content.append({
            "type": "image",
            "image": img1_path
            })
            img2_path = build_image_path(base_image_path, image_paths[1])
            content.append({
            "type": "image",
            "image": img2_path
            })
            content.append(
                {"type":"text",
                "text":prompt
                }
            )
        if mood == "4":
            img1_path = build_image_path(base_image_path, image_paths[0])
            content.append({
            "type": "image",
            "image": img1_path
            })
            img2_path = build_image_path(base_image_path, image_paths[1])
            content.append({
            "type": "image",
            "image": img2_path
            })
            content.append(
                {"type":"text",
                "text":prompt
                }
            )
        messages = [
        {
            "role": "user",
            "content": content
        }
        ]
        inputs = self._processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
        inputs = inputs.to(self._model.device)
    
        with torch.no_grad():
            generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=512,
        )
            
        generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        response = output_text.strip()

        return response

    def generate_no_external(
        self,
        image_paths: list[str],
        question: str,
        **kwargs: Any,
    ) -> dict:
        """Generate an answer without external document."""
        mood = "1"
        prompt = PROMPT_NO_EXTERNAL.format(question=question)
        raw_output = self._generate_response(prompt, image_paths,mood)
        parsed = self._parse_answer_reason(raw_output)
        return {
            "answer": parsed.get("answer", "")
        }

    def generate_with_external(
        self,
        image_paths: list[str],
        question: str,
        external_document: str,
        **kwargs: Any,
    ) -> dict:
        """Generate an answer with external document."""
        mood = "2"
        prompt = PROMPT_WITH_EXTERNAL.format(
            question=question,
            external_document=external_document,
        )
        raw_output = self._generate_response(prompt, image_paths,mood)
        parsed = self._parse_answer_reason(raw_output)
        return {
             "answer": parsed.get("answer", "")
        }

    def generate_with_experience(
        self,
        image_paths: list[str],
        question: str,
        external_document: str,
        experiences: list[dict],
        **kwargs: Any,
    ) -> dict:
        """Generate an answer using historical experiences."""
        mood = "3"
        experiences_text = json.dumps(experiences, ensure_ascii=False, indent=2)
        prompt = PROMPT_WITH_EXPERIENCE.format(
            question=question,
            external_document=external_document,
            experiences_text=experiences_text,
        )
        raw_output = self._generate_response(prompt, image_paths,mood)
        parsed = self._parse_output(raw_output)
        return {
            "recognized_entity": parsed.get("recognized_entity", ""),
            "answer": parsed.get("answer", ""),
            "reason": parsed.get("reason", ""),
            "confidence": float(parsed.get("confidence", 0.0)),
            "raw_output": raw_output,
        }

    def reconcile(
        self,
        image_paths: list[str],
        question: str,
        external_document: str,
        answer1_no_external: str,
        answer2_with_external: str,
        conflict_summary: str,
        **kwargs: Any,
    ) -> dict:
        """Reconcile conflicting answers."""
        mood ="4"
        prompt = PROMPT_RECONCILE.format(
            question=question,
            external_document=external_document,
            answer1_no_external=answer1_no_external,
            answer2_with_external=answer2_with_external,
            conflict_summary=conflict_summary,
        )
        raw_output = self._generate_response(prompt, image_paths,mood)
        parsed = self._parse_output(raw_output)
        return {
            "answer": parsed.get("answer", ""),
            "reason": parsed.get("reason", ""),
            "confidence": float(parsed.get("confidence", 0.0)),
            "raw_output": raw_output,
        }

    @staticmethod
    def _parse_output(raw_output: str) -> dict:
        """Parse JSON from model output."""
        if not raw_output:
            return {}

        # Try ```json fence
        import re
        match = re.search(r"```json\s*(.*?)\s*```", raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try first { ... } block
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {}
    
    @staticmethod
    def _parse_answer_reason(raw_output: str) -> dict:
       

        return {
            "answer": raw_output,
        }