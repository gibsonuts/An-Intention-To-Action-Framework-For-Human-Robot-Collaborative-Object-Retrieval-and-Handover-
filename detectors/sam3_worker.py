#!/usr/bin/env python3
"""Internal worker for one-shot SAM3 subprocess inference."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from detectors.sam3_object_detection import Sam3Detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("segment", "detect_bbox"), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    color = np.load(args.image, allow_pickle=False)
    params = json.loads(args.params.read_text(encoding="utf-8"))
    detector = Sam3Detector(device=args.device)

    if args.operation == "segment":
        result = detector.segment(color_bgr=color, **params)
    else:
        result = detector.detect_bbox(color=color, **params)

    with args.output.open("wb") as output_file:
        pickle.dump(result, output_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
