#!/usr/bin/env python3
"""
detect_with_chatgpt.py

Send an image + query to OpenAI (Responses API) and get back *points*.
Each detected object is represented as a 1x1 "box" at the target point:
[x, y, w, h] where w = 1 and h = 1, in pixel coordinates.

Usage:
  python detect_with_chatgpt.py --image path/to/img.jpg --query "Find the red mug" --model gpt-4o
  # (Optional) If you have your own drawing utility, a 1x1 boTarget Marks the point.
"""

import os
import sys
import json
import base64
import argparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import yaml
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# MODEL = "gpt-5-mini" #"gpt-5"  # or gpt-4.1, gpt-4o, etc.
# REASONING_EFFORT = "medium"  # "none", "low", "medium", "high"

MODEL = "gpt-5"  
REASONING_EFFORT = "medium"  # "none", "low", "medium", "high"
# MODEL = "gpt-5-mini"  
# REASONING_EFFORT = "medium"  # "none", "low", "medium", "high"

FUNCTION_JSON = "function_pts.json"  # in ./config/
SYSTEM_INSTRUCTIONS = "system_instructions.txt"  # in ./config/

def load_image_as_data_url(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}", img.size  # (data_url, (W,H))


def load_api_key_from_yaml(yaml_path: str) -> str:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return os.path.expandvars(cfg["openai"]["api_key"])


def extract_boxes_from_response(resp, bbox_size=24, image_size=None):
    """
    Returns a normalized list of dicts like:
      {"label": str, "confidence": float|None, "xywh": [x, y, w, h]}
    If only points are present (xy), we convert them to centreed square boxes
    of `bbox_size`. If `image_size=(W,H)` is provided, we clip to image bounds.
    """
    import json

    def _clip_box(x, y, w, h, W, H):
        if W is None or H is None:
            # round to ints even if we don't clip
            return int(round(x)), int(round(y)), int(round(w)), int(round(h))
        x = max(0, min(int(round(x)), W - 1))
        y = max(0, min(int(round(y)), H - 1))
        w = max(1, min(int(round(w)), W - x))
        h = max(1, min(int(round(h)), H - y))
        return x, y, w, h

    W = H = None
    if image_size:
        W, H = image_size

    def _normalize(points):
        out = []
        for i, p in enumerate(points or []):
            # handle dict items; ignore unknown shapes
            if not isinstance(p, dict):
                continue

            label = p.get("label") or f"obj_{i}"
            conf = p.get("confidence")
            # already a box?
            if "xywh" in p and isinstance(p["xywh"], (list, tuple)) and len(p["xywh"]) >= 4:
                x, y, w, h = p["xywh"][:4]
            else:
                # turn a point into a centreed square box
                if "xy" in p and isinstance(p["xy"], (list, tuple)) and len(p["xy"]) >= 2:
                    px, py = p["xy"][:2]
                elif {"x", "y"}.issubset(p.keys()):
                    px, py = p["x"], p["y"]
                else:
                    # nothing we can parse
                    continue
                s = int(round(bbox_size))
                x = int(round(px - s / 2))
                y = int(round(py - s / 2))
                w = s
                h = s

            x, y, w, h = _clip_box(x, y, w, h, W, H)
            out.append({"label": label, "confidence": conf, "xywh": [x, y, w, h]})
        return out

    # 1) Structured output_text path (if model returned JSON there)
    try:
        out_txt = getattr(resp, "output_text", None)
        if out_txt:
            data = json.loads(out_txt)
            if isinstance(data, dict):
                if "points" in data:
                    return _normalize(data["points"])
                if isinstance(data.get("output"), dict) and "points" in data["output"]:
                    return _normalize(data["output"]["points"])
    except Exception:
        pass

    # 2) Tool/function-call path (your example)
    for item in (getattr(resp, "output", None) or []):
        t = getattr(item, "type", None)
        if t == "function_call":
            try:
                args = json.loads(getattr(item, "arguments", "{}"))
                pts = args.get("points")
                # prefer image_size from tool if not already provided
                if not image_size and isinstance(args.get("image_size"), dict):
                    iw = args["image_size"].get("width")
                    ih = args["image_size"].get("height")
                    if isinstance(iw, int) and isinstance(ih, int):
                        W, H = iw, ih
                if pts is not None:
                    return _normalize(pts)
            except Exception:
                continue
        elif t == "tool_result":
            # some SDKs return tool results this way
            try:
                content = getattr(item, "content", None)
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content
                if isinstance(data, dict) and "points" in data:
                    return _normalize(data["points"])
            except Exception:
                continue

    return []


def print_locations_and_confidence(boxes):
    if not boxes:
        print("No points found.")
        return
    for i, b in enumerate(boxes):
        x, y, w, h = b["xywh"]
        conf = b.get("confidence")
        label = b.get("label", f"obj_{i}")
        # Round to ints for readability
        xi, yi = int(round(x)), int(round(y))
        if conf is not None:
            print(f"{label}: point (x,y)=({xi}, {yi}), confidence={conf:.3f}")
        else:
            print(f"{label}: point (x,y)=({xi}, {yi}), confidence=N/A")


def detect_boxes(image, query):
    """
    Detect point(s) for objects matching `query`.
    
    NOTE: For compatibility, this returns a list of dicts with "xywh" fields,
    but width and height are always set to 1, i.e., [x, y, 1, 1].
    """
    boxes = None
    # 1) Prepare image as a data URL (works well for Responses API image inputs).
    data_url, (W, H) = load_image_as_data_url(image)

    #read system instructions from file
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    if not os.path.exists(config_dir) or not os.path.exists(os.path.join(config_dir, SYSTEM_INSTRUCTIONS)):
        print(f"[!] Please create {os.path.join(config_dir, 'system_instructions.txt')} with your system instructions.")
        sys.exit(1)
    with open(os.path.join(config_dir, SYSTEM_INSTRUCTIONS), "r") as f:
        system_instructions = f.read()

 
    # print("System instructions:", system_instructions)

    # 3) Call the Responses API with image + text. (See OpenAI Quickstart & Responses reference.)
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    if not os.path.exists(config_dir) or not os.path.exists(os.path.join(config_dir, "config.yaml")):
        print(f"[!] Please create {os.path.join(config_dir, 'config.yaml')} with your OpenAI API key.")
        sys.exit(1)

    json_dir = os.path.join(os.path.dirname(__file__), "config")
    if not os.path.exists(json_dir) or not os.path.exists(os.path.join(json_dir, FUNCTION_JSON)):
        print(f"[!] Please create {os.path.join(json_dir, 'function.json')} with your function schema.")
        sys.exit(1)
    with open(os.path.join(json_dir, FUNCTION_JSON), "r") as f:
        json_fuction = json.load(f)

    api_key = load_api_key_from_yaml(os.path.join(config_dir, "config.yaml"))  # path to your yml
    client = OpenAI(api_key=api_key)

    # The Responses API accepts multimodal "input" blocks; we’ll pass an input_text and an input_image.
    print("sending request to chatGPT...", MODEL,'prompt',query)
    start_time = time.time()
    resp = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_instructions,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Image size is {W}x{H} px. Query: {query}"},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text={
            "format": {"type": "text"},
            "verbosity": "medium",
        },
        reasoning={"effort": REASONING_EFFORT},
        tools=[json_fuction],
        store=True,
        include=["reasoning.encrypted_content", "web_search_call.action.sources"],
    )
    elapsed = time.time() - start_time
    print(f"chatgpt response  took {elapsed:.3f} seconds")

    # print("response received.",resp)
    reasoning = getattr(resp, "reasoning", None)
    print("=== Reasoning ===", reasoning)

    # 5) Print points & confidence for quick visibility.
    # boxes = extract_boxes_from_response(resp)
    # if not boxes:
    #     print("No boxs found in LLM found.")
    #     return None

    # print_locations_and_confidence(boxes)

    # 6) Extract and coerce to 1x1 boxes (points), clipped to image bounds.
    # boxes = extract_boxes_from_response(resp)
    auto_box = max(30, int(round(min(W, H) * 0.1)))
    boxes = extract_boxes_from_response(resp, bbox_size=auto_box, image_size=(W, H))
    if not boxes:
        print("No boxs found in LLM found")
        return None

    # Create a bounding box of size bbox_size around each detected point
    fixed = []
    bbox_size = auto_box  # use the same size as used in extract_boxes_from_response
    half_box = bbox_size // 2
    for b in boxes:
        x, y, _, _ = b["xywh"]  # ignore any model-provided w/h
        x = int(round(x))
        y = int(round(y))
        # Calculate top-left corner of bbox
        x1 = max(0, min(x - half_box, W - 1))
        y1 = max(0, min(y - half_box, H - 1))
        # Calculate width and height, ensuring bbox stays within image bounds
        w = min(bbox_size, W - x1)
        h = min(bbox_size, H - y1)
        nb = dict(b)
        nb["xywh"] = [x1, y1, w, h]
        fixed.append(nb)

    return fixed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image file (jpg/png/webp).")
    ap.add_argument("--query", required=True, help='e.g., "Find the red mug" or "Locate all traffic cones".')
    args = ap.parse_args()

    detect_boxes(args.image, args.query)
