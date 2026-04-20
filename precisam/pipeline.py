import json
import queue
import threading
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .predictor import SAM3Predictor
from .utils import xywh_to_xyxy

_SENTINEL = object()


def _prefetch_images(image_ids, images_map, images_dir, buf: queue.Queue) -> None:
    for image_id in image_ids:
        filename = images_map.get(image_id)
        if not filename:
            buf.put((image_id, None))
            continue
        img_path = images_dir / filename
        if not img_path.exists():
            buf.put((image_id, None))
            continue
        try:
            buf.put((image_id, Image.open(img_path).convert("RGB")))
        except Exception:
            buf.put((image_id, None))
    buf.put(_SENTINEL)


def process_coco(
    annotations_path: str | Path,
    images_dir: str | Path,
    predictor: SAM3Predictor,
    skip_existing: bool = True,
    progress: bool = True,
    checkpoint_every: int = 50,
) -> dict:
    """
    Refine all bboxes in a COCO annotation file using SAM3.

    - Groups annotations by image so each image is encoded only once.
    - Prefetches images on a background thread while GPU processes the previous one.
    - Writes a checkpoint JSON every `checkpoint_every` images (same path, .ckpt.json suffix).
    - Returns a new COCO dict (original is not mutated).
    """
    annotations_path = Path(annotations_path)
    images_dir = Path(images_dir)
    checkpoint_path = annotations_path.with_suffix(".ckpt.json")

    with open(annotations_path) as f:
        coco = json.load(f)

    images_map = {img["id"]: img["file_name"] for img in coco["images"]}

    by_image: dict[int, list[int]] = defaultdict(list)
    for idx, ann in enumerate(coco["annotations"]):
        by_image[ann["image_id"]].append(idx)

    # Partial copy: only annotations is mutated, everything else is shared
    result = {**coco, "annotations": [dict(a) for a in coco["annotations"]]}

    image_ids = list(by_image.keys())
    total_images = len(image_ids)

    refined_count = 0
    skipped_count = 0
    failed_count = 0

    # Prefetch queue (buffer 3 images ahead)
    buf: queue.Queue = queue.Queue(maxsize=3)
    threading.Thread(
        target=_prefetch_images,
        args=(image_ids, images_map, images_dir, buf),
        daemon=True,
    ).start()

    if progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(result["annotations"]), desc="Annotations", unit="ann")

    for i in range(total_images):
        item = buf.get()
        if item is _SENTINEL:
            break
        image_id, image = item

        if image is None:
            if progress:
                pbar.update(len(by_image[image_id]))
            continue

        predictor.set_image(image)

        for idx in by_image[image_id]:
            ann = result["annotations"][idx]

            if skip_existing and ann.get("sam_bbox") == 1:
                skipped_count += 1
                if progress:
                    pbar.update(1)
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
                pbar.update(1)

        if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(result, f)

    if progress:
        pbar.close()
        print(f"Done — refined: {refined_count}, skipped: {skipped_count}, failed: {failed_count}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return result
