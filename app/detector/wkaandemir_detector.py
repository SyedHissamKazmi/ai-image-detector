from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import torch

from app.core.logging_config import get_logger
from app.detector.base import BaseDetector

logger = get_logger("detector.wkaandemir")


class WkaandemirDetector(BaseDetector):
    MODEL_NAME = "wkaandemir/ai-image-detector"

    def __init__(self) -> None:
        self._model: Any | None = None
        self._loaded_path: Path | None = None
        self._load_attempted = False
        self._available = False
        self._lock = Lock()
        self.device = torch.device("cpu")

    @property
    def name(self) -> str:
        return "wkaandemir"

    @property
    def available(self) -> bool:
        return self._available

    def _load(self) -> bool:
        if self._load_attempted:
            return self._available

        with self._lock:
            if self._load_attempted:
                return self._available

            self._load_attempted = True

            try:
                import timm
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file

                logger.info(
                    "ML_MODEL_LOADING | model=%s | architecture=CLIP-ViT-B/16 | device=CPU",
                    self.MODEL_NAME,
                )

                weights_path = hf_hub_download(
                    repo_id=self.MODEL_NAME,
                    filename="model.safetensors",
                )

                self._loaded_path = Path(weights_path)

                model = timm.create_model(
                    "vit_base_patch16_clip_224.openai",
                    pretrained=False,
                    num_classes=1,
                    img_size=256,          # ← critical fix for pos_embed
                )

                state_dict = load_file(weights_path)

                missing_keys, unexpected_keys = model.load_state_dict(
                    state_dict,
                    strict=False,
                )

                if missing_keys:
                    logger.warning(
                        "ML_MODEL_LOAD_PARTIAL | model=%s | missing_keys=%d",
                        self.MODEL_NAME,
                        len(missing_keys),
                    )

                if unexpected_keys:
                    logger.warning(
                        "ML_MODEL_LOAD_PARTIAL | model=%s | unexpected_keys=%d",
                        self.MODEL_NAME,
                        len(unexpected_keys),
                    )

                model.to(self.device)
                model.eval()

                self._model = model
                self._available = True

                logger.info(
                    "ML_MODEL_READY | model=%s | architecture=CLIP-ViT-B/16 | device=CPU",
                    self.MODEL_NAME,
                )

            except Exception as exc:
                self._model = None
                self._available = False

                logger.warning(
                    "ML_MODEL_UNAVAILABLE | model=%s | error=%s",
                    self.MODEL_NAME,
                    exc,
                )

            return self._available

    def predict(self, image_path: str | Path) -> float | None:
        path = Path(image_path)

        if not path.is_file():
            logger.warning(
                "ML_PREDICTION_FAILED | model=%s | file=%s | reason=not_found",
                self.MODEL_NAME,
                path.name,
            )
            return None

        if not self._load():
            return None

        try:
            from PIL import Image
            from torchvision import transforms

            image = Image.open(path).convert("RGB")

            transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.481, 0.458, 0.408],
                        std=[0.269, 0.261, 0.276],
                    ),
                ]
            )

            tensor = transform(image).unsqueeze(0).to(self.device)
            image.close()

            with torch.inference_mode():
                output = self._model(tensor)

            # The model outputs one logit representing p(real).
            logit = output.reshape(-1)[0]
            p_real = torch.sigmoid(logit).item()
            p_ai = 1.0 - p_real

            p_ai = max(0.0, min(1.0, p_ai))

            logger.info(
                "ML_MODEL_PREDICTION | model=%s | file=%s | ai_probability=%.4f",
                self.MODEL_NAME,
                path.name,
                p_ai,
            )

            return p_ai

        except Exception as exc:
            logger.warning(
                "ML_PREDICTION_FAILED | model=%s | file=%s | error=%s",
                self.MODEL_NAME,
                path.name,
                exc,
            )
            return None