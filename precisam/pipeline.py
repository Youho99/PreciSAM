import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from PIL import Image

from .predictor import SAM3Predictor
from .utils import xywh_to_xyxy


def process_coco(
    annotations_path: str | Path,
    images_dir: str | Path,
    predictor: SAM3Predictor,
    skip_existing: bool = True,
    progress: bool = True,
) -> dict:
    """
    Refine all bboxes in a COCO annotation file using SAM3.

    Groups annotations by image so each image is encoded only once.
    Returns a new COCO dict (original is not mutated).
    """
    annotations_path = Path(annotations_path)
    images_dir = Path(images_dir)

    with open(annotations_path) as f:
        coco = json.load(f)

    images_map = {img["id"]: img["file_name"] for img in coco["images"]}

    by_image: dict[int, list[int]] = defaultdict(list)
    for idx, ann in enumerate(coco["annotations"]):
        by_image[ann["image_id"]].append(idx)

    result = deepcopy(coco)

    image_ids = list(by_image.keys())

    if progress:
        from tqdm import tqdm
        image_iter = tqdm(image_ids, desc="Images", unit="img")
    else:
        image_iter = image_ids

    refined_count = 0
    skipped_count = 0
    failed_count = 0

    for image_id in image_iter:
        filename = images_map.get(image_id)
        if not filename:
            continue

        img_path = images_dir / filename
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        predictor.set_image(image)

        for idx in by_image[image_id]:
            ann = result["annotations"][idx]

            if skip_existing and ann.get("sam_bbox") == 1:
                skipped_count += 1
                continue

            bbox_xyxy = xywh_to_xyxy(ann["bbox"])
            refined = predictor.refine_bbox(bbox_xyxy)

            if refined is not None:
                ann["bbox"] = refined
                ann["area"] = float(refined[2] * refined[3])
                ann["sam_bbox"] = 1
                refined_count += 1
            else:
                failed_count += 1

    if progress:
        print(
            f"Done — refined: {refined_count}, skipped: {skipped_count}, failed: {failed_count}"
        )

    return result
