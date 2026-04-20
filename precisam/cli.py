import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

from .pipeline import process_coco
from .predictor import SAM3Predictor


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="precisam",
        description="Refine COCO bounding boxes using SAM3",
    )
    parser.add_argument("input", help="Input ZIP (COCO) or directory")
    parser.add_argument("-o", "--output", required=True, help="Output ZIP or directory")
    parser.add_argument("--model", default=None, metavar="MODEL_ID", help="HuggingFace model ID")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device (default: auto)"
    )
    parser.add_argument(
        "--no-skip", action="store_true", help="Reprocess annotations already marked sam_bbox=1"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bar")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    predictor = SAM3Predictor(device=args.device, model_id=args.model)
    print(f"Loading SAM3 model on {args.device}...", flush=True)
    predictor.load()
    print("Model loaded.", flush=True)

    skip_existing = not args.no_skip
    show_progress = not args.quiet

    if input_path.suffix.lower() == ".zip":
        _process_zip(input_path, output_path, predictor, skip_existing, show_progress)
    else:
        _process_dir(input_path, output_path, predictor, skip_existing, show_progress)


def _find_coco_json(directory: Path) -> Path:
    for p in sorted(directory.rglob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            if "annotations" in data and "images" in data:
                return p
        except Exception:
            continue
    raise FileNotFoundError(f"No COCO JSON found in {directory}")


def _find_images_dir(directory: Path) -> Path:
    candidate = directory / "images"
    if candidate.exists():
        contents = [p for p in candidate.iterdir()]
        if len(contents) == 1 and contents[0].is_dir():
            return contents[0]
        return candidate
    return directory


def _process_zip(
    input_zip: Path,
    output_zip: Path,
    predictor: SAM3Predictor,
    skip_existing: bool,
    progress: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        with zipfile.ZipFile(input_zip, "r") as z:
            z.extractall(tmp)

        ann_path = _find_coco_json(tmp)
        images_dir = _find_images_dir(tmp)

        result = process_coco(ann_path, images_dir, predictor, skip_existing, progress)

        with open(ann_path, "w") as f:
            json.dump(result, f)

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for file in sorted(tmp.rglob("*")):
                if file.is_file():
                    z.write(file, file.relative_to(tmp))

    print(f"Saved: {output_zip}")


def _process_dir(
    input_dir: Path,
    output_dir: Path,
    predictor: SAM3Predictor,
    skip_existing: bool,
    progress: bool,
) -> None:
    ann_path = _find_coco_json(input_dir)
    images_dir = _find_images_dir(input_dir)

    result = process_coco(ann_path, images_dir, predictor, skip_existing, progress)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_ann = output_dir / ann_path.name
    with open(out_ann, "w") as f:
        json.dump(result, f)

    print(f"Saved: {out_ann}")


if __name__ == "__main__":
    main()
