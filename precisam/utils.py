import numpy as np


def xywh_to_xyxy(bbox: list) -> list:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_xywh(bbox: list) -> list:
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def xyxy_to_cxcywh_norm(bbox: list, img_w: int, img_h: int) -> list:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [cx, cy, w, h]


def mask_to_xywh(mask: np.ndarray) -> list | None:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
