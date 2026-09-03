#!/usr/bin/env python3
"""
Web search ON by default with the OpenAI Responses API.
- No realtime
- No custom handlers
- Location bias defaults to London, GB (change as needed)
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from openai import OpenAI
from commons.grasp_utils import check_path_exists
import yaml, re

# Set OPENAI_API_KEY in your environment or pass api_key to WebSearcher(...)
#   export OPENAI_API_KEY=sk-...
CFG_PATH = "config/config.yaml"

@dataclass
class UserLocation:
    type: str = "approximate"
    country: str = "Au"
    city: str = "Sydney"
    region: str = "Sydney"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class WebSearcher:
    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        location: Optional[UserLocation] = None,
        max_results: int = 6,
    ):
        def expand_env(x):
            if isinstance(x, str):
                return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), ""), x)
            if isinstance(x, list):
                return [expand_env(i) for i in x]
            if isinstance(x, dict):
                return {k: expand_env(v) for k, v in x.items()}
            return x
        
        cfg = {}
        cfg_file = check_path_exists(CFG_PATH,__file__)
        if cfg_file:
            with cfg_file.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            print('ERROR no cfg file',cfg_file)
            exit()
        cfg = expand_env(cfg)
        params = cfg.get("realtime_client") or cfg.get("client", {})
        self.client = OpenAI(api_key=params["api_key"])
        self.model = model
        self.location = location or UserLocation()
        self.max_results = max_results

        # Prepare the web_search tool once (always on by default)
        self._tools = [{
            "type": "web_search",
            "user_location": self.location.to_dict()
        }]

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Runs the query with web_search enabled by default and returns:
          - output_text: the model’s synthesized answer
          - sources: best-effort list of citations (if available)
          - raw: the raw Responses object (SDK wrapper) for advanced use
        """
        print("asking",query)
        resp = self.client.responses.create(
            model=self.model,
            tools=self._tools,
            input=query,
        )
        print(resp)

        # Synthesis
        answer = getattr(resp, "output_text", "") or ""

        # Collect citations if the SDK exposes them (varies by version)
        sources: List[Dict[str, str]] = []
        try:
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", None) == "citation" and getattr(item, "document", None):
                    doc = item.document
                    sources.append({
                        "title": getattr(doc, "title", "") or "",
                        "url": getattr(doc, "url", "") or "",
                        "snippet": getattr(doc, "snippet", "") or "",
                    })
        except Exception:
            pass

        # Trim to max_results if needed
        if self.max_results and len(sources) > self.max_results:
            sources = sources[: self.max_results]

        return {
            "output_text": answer.strip(),
            "sources": sources,
            "raw": resp,
        }

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="What are the best restaurants around Granary Square?")
    args = ap.parse_args()
    # Example usage
    searcher = WebSearcher(
        # model="o4-mini",  # keep or change
        # location=UserLocation(country="GB", city="London", region="London"),
        # max_results=8,
    )

    query =  args.prompt
    result = searcher.ask(query)

    print("\n==== ANSWER ====")
    print(result["output_text"])

    if result["sources"]:
        print("\n==== SOURCES ====")
        for i, s in enumerate(result["sources"], 1):
            title = s.get("title") or "(untitled)"
            url = s.get("url") or ""
            snippet = s.get("snippet") or ""
            print(f"{i}. {title}\n   {url}\n   {snippet}\n")
