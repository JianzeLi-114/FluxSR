"""Evaluate image quality with optional ground-truth images.

Example:
    python scripts/evaluate.py --pred-dir outputs/images \
        --gt-dir data/gt --output-dir outputs/metrics \
        --metrics psnr ssim lpips dists musiq maniqa topiq_nr

Images match by case-sensitive relative paths without extensions by default.
Optional suffixes are removed only from the end of filename stems. Without
ground truth, evaluation supports MUSIQ, MANIQA, TOPIQ NR and Q-Align.
PSNR and SSIM use YCbCr Y by default, with optional RGB evaluation and border
cropping. SSIM downsampling is disabled. Other metrics receive uncropped RGB
images and use PyIQA default model variants.
"""

import argparse
import csv
import gc
import json
import math
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


METRICS = ("psnr", "ssim", "lpips", "dists", "musiq", "maniqa", "topiq_nr", "qalign")
NR_METRICS = ("musiq", "maniqa", "topiq_nr", "qalign")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LOCAL_WEIGHTS = {
    "maniqa": "ckpt_koniq10k.pt",
    "topiq_nr": "cfanet_nr_koniq_res50-9a73138b.pth",
}


def image_id(path, directory, match_mode, suffix=""):
    """Keep filename case and strip only the extension and a trailing suffix."""
    stem = path.stem
    if suffix and stem.endswith(suffix):
        stem = stem[:-len(suffix)]
    if not stem:
        raise ValueError(f"Empty image ID after removing suffix {suffix!r}: {path}")
    if match_mode == "stem":
        return stem
    return (path.relative_to(directory).parent / stem).as_posix()


def image_files(directory, pattern):
    return sorted(p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def make_pairs(args):
    """Index predictions and optional references, checking IDs, coverage and sizes."""
    groups = []
    sources = [("prediction", args.pred_dir, args.pred_glob, args.pred_suffix)]
    if args.gt_dir is not None:
        sources.append(("ground truth", args.gt_dir, args.gt_glob, args.gt_suffix))
    for label, directory, pattern, suffix in sources:
        paths = image_files(directory, pattern)
        if not paths:
            raise ValueError(f"No {label} images in {directory} matching {pattern!r}")
        indexed = {}
        for path in paths:
            key = image_id(path, directory, args.match_mode, suffix)
            if key in indexed:
                raise ValueError(f"Duplicate {label} ID {key!r}: {indexed[key]} and {path}")
            indexed[key] = path
        groups.append(indexed)
    predictions = groups[0]
    references = groups[1] if args.gt_dir is not None else None
    if references is not None:
        unexpected = sorted(predictions.keys() - references.keys())
        missing = sorted(references.keys() - predictions.keys())
        if unexpected:
            raise ValueError(f"{len(unexpected)} predictions have no GT match; first IDs: {unexpected[:5]}")
        if missing and not args.allow_subset:
            raise ValueError(f"{len(missing)} GT images have no prediction; first IDs: {missing[:5]}. "
                             "Use --allow-subset only for an explicitly partial evaluation.")
    pairs = []
    minimum = 11 if "ssim" in args.metrics else 1
    crop_border = args.crop_border if any(name in {"psnr", "ssim"} for name in args.metrics) else 0
    for key, prediction in predictions.items():
        reference = references[key] if references is not None else None
        with Image.open(prediction) as pred_image:
            width, height = pred_image.size
        if reference is not None:
            with Image.open(reference) as gt_image:
                if (width, height) != gt_image.size:
                    raise ValueError(f"Image size mismatch for {key}: prediction {(width, height)}, "
                                     f"GT {gt_image.size}; evaluation does not resize images")
        if min(width, height) - 2 * crop_border < minimum:
            raise ValueError(f"{key}: image too small for the requested metrics/crop border")
        pairs.append({"id": key, "prediction": str(prediction),
                      "reference": str(reference) if reference is not None else None,
                      "width": width, "height": height})
    return pairs, len(references) if references is not None else None


def read_tensor(path, device):
    with Image.open(path) as image:
        pixels = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    # Convert RGB uint8 pixels to float32 metric inputs in [0, 1].
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device, torch.float32) / 255.0


def create_metric(pyiqa, name, device, options, weights_dir=None):
    if weights_dir is None or name not in LOCAL_WEIGHTS:
        return pyiqa.create_metric(name, device=device, **options).eval()

    checkpoint = weights_dir / LOCAL_WEIGHTS[name]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing local {name} weights: {checkpoint}")
    # Load both the backbone and scoring head from the local IQA checkpoint.
    if name == "maniqa":
        import timm

        original_create_model = timm.create_model

        def create_without_vit_download(model_name, *args, **kwargs):
            if model_name == "vit_base_patch8_224":
                kwargs["pretrained"] = False
            return original_create_model(model_name, *args, **kwargs)

        with patch.object(timm, "create_model", create_without_vit_download):
            metric = pyiqa.create_metric(name, device=device, pretrained=False, **options)
    else:
        metric = pyiqa.create_metric(name, device=device, pretrained=False,
                                    backbone_pretrain=False, **options)

    from pyiqa.archs.arch_util import clean_state_dict

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "params" in state:
        state = state["params"]
    # Restore the complete metric network.
    metric.net.load_state_dict(clean_state_dict(state), strict=True)
    print(f"Loaded complete {name} checkpoint from {checkpoint}", flush=True)
    return metric.eval()


def save_results(args, rows, results, gt_count, pyiqa_version):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prediction", "reference", "width", "height", *args.metrics])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "prediction_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir) if args.gt_dir is not None else None,
        "image_count": len(rows), "gt_count": gt_count,
        "complete_dataset": len(rows) == gt_count if gt_count is not None else None,
        "pyiqa_version": pyiqa_version,
        "weights_dir": str(args.weights_dir) if args.weights_dir is not None else None,
        "protocol": {
            "matching": {"mode": args.match_mode, "case_sensitive": True,
                         "pred_suffix": args.pred_suffix, "gt_suffix": args.gt_suffix},
            "full_size": True, "alignment_or_resize": False,
            "psnr_ssim_color_space": "YCbCr Y" if args.color_space == "y" else "RGB",
            "psnr_ssim_crop_border": args.crop_border,
            "other_metrics_crop_border": 0, "ssim_downsample": False,
            "other_metric_variants": "PyIQA defaults; LPIPS alex/v0.1, TOPIQ NR",
            "seed": args.seed,
        },
        "means": {name: result["mean"] for name, result in results.items() if result["status"] == "complete"},
        "metrics": results,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)


def evaluate(args):
    import pyiqa

    rows, gt_count = make_pairs(args)
    if gt_count is None:
        print(f"Validated {len(rows)} prediction images for no-reference evaluation.", flush=True)
    else:
        print(f"Validated {len(rows)} prediction/GT pairs ({gt_count} GT images).", flush=True)
    results = {}
    version = getattr(pyiqa, "__version__", "unknown")
    save_results(args, rows, results, gt_count, version)
    for name in args.metrics:
        metric = None
        values = []
        try:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            options = {}
            if name in {"psnr", "ssim"}:
                options.update(test_y_channel=args.color_space == "y", color_space="ycbcr",
                               crop_border=args.crop_border)
            if name == "ssim":
                options["downsample"] = False
            print(f"Loading {name} on {args.device} ...", flush=True)
            metric = create_metric(pyiqa, name, args.device, options, args.weights_dir)
            for index, row in enumerate(rows):
                with torch.inference_mode():
                    prediction = read_tensor(row["prediction"], args.device)
                    if metric.metric_mode == "FR":
                        reference = read_tensor(row["reference"], args.device)
                        output = metric(prediction, reference)
                    else:
                        output = metric(prediction)
                    if output.numel() != 1:
                        raise ValueError(f"{name} returned {output.numel()} scores for {row['id']}")
                    value = float(output.item())
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {name} score for {row['id']}: {value}")
                row[name] = value
                values.append(value)
                if (index + 1) % 10 == 0 or index + 1 == len(rows):
                    print(f"{name}: {index + 1}/{len(rows)}", flush=True)
            mean = sum(values) / len(values)
            results[name] = {"status": "complete", "count": len(values), "mean": mean}
            print(f"{name}: mean={mean:.6f}", flush=True)
        except Exception as exc:
            results[name] = {"status": "failed", "count": len(values), "mean": None,
                             "error": f"{type(exc).__name__}: {exc}"}
            print(f"{name} unavailable/failed: {results[name]['error']}", flush=True)
            if args.fail_on_metric_error:
                save_results(args, rows, results, gt_count, version)
                raise
        finally:
            del metric
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        save_results(args, rows, results, gt_count, version)
    print(f"Results: {args.output_dir / 'summary.json'}", flush=True)
    if not any(result["status"] == "complete" for result in results.values()):
        raise RuntimeError("None of the requested metrics completed; see summary.json")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, help="Ground-truth directory; required for PSNR, SSIM, LPIPS and DISTS")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pred-glob", default="**/*", help="Path.glob pattern relative to prediction directory")
    parser.add_argument("--gt-glob", default="**/*", help="Path.glob pattern relative to ground-truth directory")
    parser.add_argument("--match-mode", choices=("relative", "stem"), default="relative",
                        help="Match case-sensitive paths without extensions: relative preserves subdirectories; stem ignores them")
    parser.add_argument("--pred-suffix", default="",
                        help="Remove this exact suffix only from the end of prediction filename stems")
    parser.add_argument("--gt-suffix", default="",
                        help="Remove this exact suffix only from the end of ground-truth filename stems")
    parser.add_argument("--metrics", nargs="+", choices=METRICS,
                        help="Defaults to seven metrics with GT, or musiq maniqa topiq_nr without GT; qalign is optional")
    parser.add_argument("--weights-dir", type=Path, help="Local MANIQA/TOPIQ checkpoint directory; avoids backbone downloads")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crop-border", type=int, default=0, help="Pixels cropped from each edge for PSNR/SSIM only; default: 0")
    parser.add_argument("--color-space", choices=("y", "rgb"), default="y",
                        help="PSNR/SSIM color space: YCbCr Y (default) or RGB")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--allow-subset", action="store_true", help="Allow predictions to cover only part of the GT dataset")
    parser.add_argument("--fail-on-metric-error", action="store_true", help="Stop if any metric fails instead of reporting and continuing")
    args = parser.parse_args(input_args)
    if args.crop_border < 0:
        parser.error("--crop-border must be nonnegative")
    if args.metrics is None:
        args.metrics = list(METRICS[:-1] if args.gt_dir is not None else NR_METRICS[:-1])
    else:
        args.metrics = list(dict.fromkeys(args.metrics))
    if args.gt_dir is None:
        full_reference = [name for name in args.metrics if name not in NR_METRICS]
        if full_reference:
            parser.error(f"--gt-dir is required for: {', '.join(full_reference)}")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
