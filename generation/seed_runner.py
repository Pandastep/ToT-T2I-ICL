# generation/seed_runner.py

from typing import Optional, List, Dict, Any
import torch
import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from load_model_tot import load_model_tot

class SeedTextRunner:
    def __init__(self, device: str = "cuda", seed: int = 123, shot: int = 2, finetuned: bool = False):
        bundle = load_model_tot(
            model="seed",
            device=device,
            gen_mode="text",
            finetuned=finetuned,
            shot=shot,
            prompt_type="default",
        )
        self.call_model = bundle["call_model"]
        self.model = bundle["model"]
        self.tokenizer = bundle["tokenizer"]
        self.transform = bundle["transform"]
        self.device = device
        self.seed = seed

    def llm_generate(self, prompt: str, *_, max_new_tokens: int = 128, seed: int = None) -> str:
        out = self.call_model({
        "text_inputs": [prompt],
        "image_inputs": [],
        "gen_mode": "text",
        "device": self.device,
        "seed": self.seed if seed is None else seed,
        "call_mode": "text",
        "instruction": ("", ""),
        "save_history": False,
        })
        text = out.get("description", "")
        if isinstance(text, (list, tuple)):
            text = "\n".join(map(str, text))
        return text.strip()
