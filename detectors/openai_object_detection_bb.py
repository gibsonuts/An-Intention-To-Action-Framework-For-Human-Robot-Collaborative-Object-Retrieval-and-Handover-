#!/usr/bin/env python3
"""
detect_with_chatgpt.py

Send an image + query to OpenAI (Responses API) and get back bounding boxes.
Boxes are [x, y, w, h] in *pixel* coordinates relative to the input image.

Usage:
  python detect_with_chatgpt.py --image path/to/img.jpg --query "Find the red mug" --model gpt-4o
  # Optional: draw results
  python detect_with_chatgpt.py --image img.png --query "Locate all traffic cones" --draw out.png
"""

import os, sys, json, base64, argparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import yaml
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.patches as patches


MODEL = "gpt-5"  # or gpt-4.1, etc.
REASONING_EFFORT = "low"  # "none", "low", "medium", "high"
FUNCTION_JSON = FUNCTION_JSON  # in ./config/
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


def extract_boxes_from_response(resp):
    """Works for either structured JSON or function-call tool output."""
    # 1) Structured outputs path
    out = getattr(resp, "output_text", None)
    if out:
        try:
            data = json.loads(out)
            if isinstance(data, dict) and "boxes" in data:
                return data["boxes"]
        except Exception:
            pass

    # 2) Tool/function-call path (your case)
    for item in getattr(resp, "output", []):
        if getattr(item, "type", None) == "function_call":
            try:
                args = json.loads(getattr(item, "arguments", "{}"))
                return args.get("boxes", [])
            except Exception:
                pass
    return []

def print_locations_and_confidence(resp):
    boxes = extract_boxes_from_response(resp)
    if not boxes:
        print("No boxes found.")
        return
    for i, b in enumerate(boxes):
        x, y, w, h = b["xywh"]
        conf = b.get("confidence")
        label = b.get("label", f"obj_{i}")
        # Round to ints for readability
        xi, yi, wi, hi = [int(round(v)) for v in (x, y, w, h)]
        if conf is not None:
            print(f"{label}: location [x,y,w,h]=[{xi}, {yi}, {wi}, {hi}], confidence={conf:.3f}")
        else:
            print(f"{label}: location [x,y,w,h]=[{xi}, {yi}, {wi}, {hi}], confidence=N/A")

def detect_boxes(image,query):

    # 1) Prepare image as a data URL (works well for Responses API image inputs).
    data_url, (W, H) = load_image_as_data_url(image)

    # 2) Build a strict JSON schema so the model MUST return valid JSON.
    # See: Structured Outputs (JSON schema) in OpenAI docs.
    #read from json file

    system_instructions = (
        "You are a precise visual grounding assistant. "
        "Given an image and a user query, return tight bounding boxes around every matching object. "
        "Rules:\n"
        "- Coordinates are integers in image pixel space, [x, y, width, height], with (0,0) top-left.\n"
        "- Only include boxes that clearly match the query; if none, return an empty list.\n"
        "- Be conservative; avoid duplicate boxes for the same object.\n"
        "- Include a confidence in [0,1]."
    )
    print("System instructions:", system_instructions)
    # 3) Call the Responses API with image + text. (See OpenAI Quickstart & Responses reference.)
    # --- inside main() or before using OpenAI() ---
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    if not os.path.exists(config_dir) or not os.path.exists(os.path.join(config_dir, "config.yaml")):
        print(f"[!] Please create {os.path.join(config_dir, 'config.yaml')} with your OpenAI API key.")
        sys.exit(1)


    json_dir = os.path.join(os.path.dirname(__file__), 'config')   
    if not os.path.exists(json_dir) or not os.path.exists(os.path.join(json_dir, FUNCTION_JSON)):
        print(f"[!] Please create {os.path.join(json_dir, 'function_bbs.json')} with your function schema.")
        sys.exit(1)
    with open(os.path.join(json_dir, FUNCTION_JSON), 'r') as f:
        json_fuction = json.load(f)

    api_key = load_api_key_from_yaml(os.path.join(config_dir, "config.yaml"))  # path to your yml
    client = OpenAI(api_key=api_key)

    # The Responses API accepts multimodal "input" blocks; we’ll pass an input_text and an input_image.
    # (See docs for image inputs & Responses API.)
    print('sending request to chatGPT...',MODEL)
    resp = client.responses.create(
        model=MODEL,
        input=[
             {
                "role": "system",
                "content": [
                    {
                    "type": "input_text",
                    "text": system_instructions
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Image size is {W}x{H} px. Query: {query}"},
                    {"type": "input_image", "image_url": data_url}
                ]
            }],
        text={
            "format": {
            "type": "text"
            },
            "verbosity": "medium"
        },
        reasoning={"effort": REASONING_EFFORT},
        tools=[json_fuction],
        store=True,
        include=['reasoning.encrypted_content',
                 "web_search_call.action.sources"
        ]
    )
    print('response received.')
    # 4) Parse the JSON payload from the first output content block.
    # Depending on SDK version, you may also access resp.output_text (already JSON due to schema).
   
    out = getattr(resp, "output_text", None)
    if not out and getattr(resp, "output", None):
        blocks = resp.output[0].content
        if blocks and hasattr(blocks[0], "text"):
            out = blocks[0].text
    # result = json.loads(out or "{}")

   
    # 5) Print JSON and optionally draw.
    # print(json.dumps(result, indent=2))
    print_locations_and_confidence(resp)

    boxes = extract_boxes_from_response(resp)
    fixed = []
    for b in boxes:
        x, y, w, h = b["xywh"]
        x = int(round(x)); y = int(round(y)); w = int(round(w)); h = int(round(h))
        # clip to image bounds
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        if w < 0: w = 0
        if h < 0: h = 0
        if x + w > W: w = max(0, W - x)
        if y + h > H: h = max(0, H - y)
        nb = dict(b); nb["xywh"] = [x, y, w, h]
        fixed.append(nb)
    return fixed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image file (jpg/png/webp).")
    ap.add_argument("--query", required=True, help='e.g., "Find the red mug" or "Locate all traffic cones".')
    args = ap.parse_args()

    detect_boxes(args.image,args.query)
