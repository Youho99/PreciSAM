import contextlib

import torch
from PIL import Image

from .utils import xyxy_to_cxcywh_norm, mask_to_xywh


class SAM3Predictor:
    def __init__(self, device: str = "auto", model_id: str | None = None):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._image_state: dict | None = None
        self._image_size: tuple[int, int] | None = None
        self._infer_ctx = (
            contextlib.nullcontext()
            if device == "cpu"
            else torch.autocast("cuda", dtype=torch.bfloat16)
        )

    def load(self) -> None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        kwargs = dict(
            device=self.device,
            eval_mode=True,
            load_from_HF=True,
            enable_segmentation=True,
        )
        if self.model_id is not None:
            kwargs["model_id"] = self.model_id

        self._model = build_sam3_image_model(**kwargs)
        self._processor = Sam3Processor(self._model, resolution=1008, device=self.device)

    def set_image(self, image: Image.Image) -> None:
        """Encode image once. Reuse across multiple refine_bbox() calls."""
        state: dict = {}
        with torch.inference_mode(), self._infer_ctx:
            self._processor.set_image(image=image, state=state)
        self._image_state = state
        self._image_size = image.size

    def refine_bbox(self, bbox_xyxy: list) -> list | None:
        """
        Refine one bbox using the cached image encoding.
        Call set_image() first. Returns [x, y, w, h] or None on failure.
        """
        if self._image_state is None or self._image_size is None:
            raise RuntimeError("Call set_image() before refine_bbox()")

        img_w, img_h = self._image_size
        box_norm = xyxy_to_cxcywh_norm(bbox_xyxy, img_w, img_h)

        # Shallow copy: image features are read-only tensors, prompt keys are new
        state = dict(self._image_state)
        with torch.inference_mode(), self._infer_ctx:
            self._processor.add_geometric_prompt(box=box_norm, label=True, state=state)
            self._processor._forward_grounding(state=state)

        masks = state.get("masks")
        if masks is None or len(masks) == 0:
            return None

        mask = masks[0].cpu().numpy().astype(bool)
        return mask_to_xywh(mask)
