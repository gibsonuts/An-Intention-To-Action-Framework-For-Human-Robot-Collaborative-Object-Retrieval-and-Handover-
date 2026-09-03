#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}


def iter_image_paths(folder: Path, recursive: bool):
    if recursive:
        yield from (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
        return
    yield from (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def convert_image_in_place(path: Path) -> None:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")

    # The file contents are assumed to be channel-swapped already.
    # Writing the RGB array with cv2.imwrite corrects the stored channels in place.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ok = cv2.imwrite(str(path), image_rgb)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swap BGR/RGB channels for PNG/JPG images in a folder and overwrite them in place."
    )
    parser.add_argument("folder", help="Folder containing images to convert")
    parser.add_argument("--recursive", action="store_true", help="Process images in subfolders too")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    image_paths = list(iter_image_paths(folder, recursive=args.recursive))
    if not image_paths:
        print("No PNG/JPG images found.")
        return

    converted = 0
    for path in image_paths:
        convert_image_in_place(path)
        converted += 1
        print(f"converted: {path}")

    print(f"done: converted {converted} image(s)")


if __name__ == "__main__":
    main()
