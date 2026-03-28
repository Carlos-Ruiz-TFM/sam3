# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from __future__ import annotations

from .ultralytics_sam3_adapter import UltralyticsSam3ImageAdapter


def build_ultralytics_sam3_image_model(
    checkpoint_path: str = "sam3.pt",
    device: str | None = None,
    confidence_threshold: float = 0.5,
    compile: bool = False,
    half: bool = True,
    iou: float = 0.7,
):
    """Build an Ultralytics SAM3 predictor wrapped in a Sam3Processor-like adapter."""
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError as exc:
        raise ImportError(
            "Ultralytics SAM3 backend requires the `ultralytics` package. "
            "Install it with `pip install -U ultralytics`."
        ) from exc

    overrides = {
        "model": checkpoint_path,
        "task": "segment",
        "mode": "predict",
        "conf": confidence_threshold,
        "iou": iou,
        "compile": compile,
        "half": half,
        "verbose": False,
    }
    if device is not None:
        overrides["device"] = device

    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.setup_model()
    return UltralyticsSam3ImageAdapter(
        predictor=predictor,
        confidence_threshold=confidence_threshold,
    )
