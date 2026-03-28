# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import PIL
import torch


class UltralyticsSam3ImageAdapter:
    """Adapter exposing a Sam3Processor-like interface over Ultralytics SAM3."""

    def __init__(self, predictor, confidence_threshold: float = 0.5):
        self.predictor = predictor
        self.confidence_threshold = confidence_threshold

    @property
    def device(self):
        return getattr(self.predictor, "device", None)

    def _to_numpy_image(self, image):
        if isinstance(image, PIL.Image.Image):
            return np.asarray(image.convert("RGB"))

        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError("NumPy image must be HWC")
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return image

        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            if tensor.ndim != 3:
                raise ValueError("Torch image must be CHW or HWC")
            if tensor.dtype.is_floating_point:
                tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)
            else:
                tensor = tensor.to(torch.uint8)
            return tensor.numpy()

        raise ValueError("Image must be a PIL image, NumPy array, or torch tensor")

    def _xywh_norm_to_xyxy_abs(self, box, width: int, height: int):
        cx, cy, w, h = box
        x0 = (cx - w / 2.0) * width
        y0 = (cy - h / 2.0) * height
        x1 = (cx + w / 2.0) * width
        y1 = (cy + h / 2.0) * height
        return [x0, y0, x1, y1]

    def _extract_outputs(self, results, state: Dict):
        result = results[0] if isinstance(results, list) else results

        if result.boxes is None or result.boxes.data.numel() == 0:
            boxes = torch.zeros((0, 4))
            scores = torch.zeros((0,))
            masks = torch.zeros(
                (0, state["original_height"], state["original_width"]), dtype=torch.bool
            )
            masks_logits = torch.zeros(
                (0, state["original_height"], state["original_width"]),
                dtype=torch.float32,
            )
        else:
            boxes = result.boxes.xyxy.detach().cpu()
            scores = result.boxes.conf.detach().cpu()
            if result.masks is None:
                masks = torch.zeros(
                    (boxes.shape[0], state["original_height"], state["original_width"]),
                    dtype=torch.bool,
                )
                masks_logits = masks.float()
            else:
                masks = result.masks.data.detach().cpu().bool()
                masks_logits = masks.float()

        state["boxes"] = boxes
        state["scores"] = scores
        state["masks"] = masks
        state["masks_logits"] = masks_logits
        return state

    @torch.inference_mode()
    def set_image(self, image, state: Optional[Dict] = None):
        if state is None:
            state = {}

        image_np = self._to_numpy_image(image)
        state["original_height"], state["original_width"] = image_np.shape[:2]
        state["_ultralytics_image"] = image_np
        state["_ultralytics_boxes"] = []
        state["_ultralytics_labels"] = []
        state["_ultralytics_text"] = None

        self.predictor.set_image(image_np)
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        if "_ultralytics_image" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        state["_ultralytics_text"] = prompt
        kwargs = {"text": [prompt]}
        if state["_ultralytics_boxes"]:
            kwargs["bboxes"] = state["_ultralytics_boxes"]
            kwargs["labels"] = state["_ultralytics_labels"]
        results = self.predictor(**kwargs)
        return self._extract_outputs(results, state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        if "_ultralytics_image" not in state:
            raise ValueError("You must call set_image before add_geometric_prompt")

        xyxy_box = self._xywh_norm_to_xyxy_abs(
            box,
            width=state["original_width"],
            height=state["original_height"],
        )
        state["_ultralytics_boxes"].append(xyxy_box)
        state["_ultralytics_labels"].append(int(label))

        kwargs = {
            "bboxes": state["_ultralytics_boxes"],
            "labels": state["_ultralytics_labels"],
        }
        if state.get("_ultralytics_text"):
            kwargs["text"] = [state["_ultralytics_text"]]

        results = self.predictor(**kwargs)
        return self._extract_outputs(results, state)

    def reset_all_prompts(self, state: Dict):
        for key in (
            "_ultralytics_boxes",
            "_ultralytics_labels",
            "_ultralytics_text",
            "boxes",
            "masks",
            "masks_logits",
            "scores",
        ):
            if key in state:
                del state[key]
        state["_ultralytics_boxes"] = []
        state["_ultralytics_labels"] = []
        state["_ultralytics_text"] = None

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        self.confidence_threshold = threshold
        if hasattr(self.predictor, "args"):
            self.predictor.args.conf = threshold
        return state
