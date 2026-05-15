#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARGenSeg Demo: Multi-turn Chat with Segmentation

Showcases segmentation + visual QA + multi-turn dialogue capabilities.

Usage:
    python demos/seg_demo_chat.py
"""

import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from internvl.model.internvl_chat import InternVLGenSeg
from internvl.train.dataset import build_transform, vis_mask


def load_image(image_path: str, input_size: int = 448) -> torch.Tensor:
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size, pad2square=False)
    return transform(image).unsqueeze(0)


def main():
    # Config
    MODEL_PATH = 'pretrained/InternVL2_5-ARGenSeg-8B'
    DEFAULT_IMAGE = './assets/image1.jpg'
    OUTPUT_DIR = './results'
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Load model
    print(f"🚀 Loading model...")
    model = InternVLGenSeg.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=False
    )
    print(f"✅ Model ready!\n")
    
    generation_config = dict(max_new_tokens=4096, do_sample=False)
    
    # === Demo 1: Segmentation ===
    print("="*50)
    print("📍 Demo 1: Referring Segmentation")
    print("="*50)
    pixel_values = load_image(DEFAULT_IMAGE).to(torch.bfloat16).cuda()
    image = Image.open(DEFAULT_IMAGE)
    print(f"🖼️  Image: {DEFAULT_IMAGE}")
    
    question = "<image>\nSegment the animal in the image."
    response = model.chat(
        tokenizer, pixel_values, question, 
        generation_config, return_multi_results=True
    )
    
    # Save masks
    for idx, mask_img in enumerate(model.language_model.gen_cache):
        mask_img = vis_mask(np.array(image), np.array(mask_img))
        save_path = f"{OUTPUT_DIR}/mask_{idx}.png"
        mask_img.resize(image.size).save(save_path)
    model.language_model.gen_cache = []
    print(f"Q: {question}")
    print(f"A: {response}")
    print(f"💾 Masks saved to {OUTPUT_DIR}/\n")
    
    # === Demo 2: Visual QA ===
    print("="*50)
    print("💬 Demo 2: Visual Question Answering")
    print("="*50)
    pixel_values = load_image(DEFAULT_IMAGE).to(torch.bfloat16).cuda()
    image = Image.open(DEFAULT_IMAGE)
    print(f"🖼️  Image: {DEFAULT_IMAGE}")
    
    question = "<image>\nWhat is in this image?"
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(f"Q: {question}")
    print(f"A: {response}\n")
    
    # === Demo 3: Multi-turn Dialogue ===
    print("="*50)
    print("🔄 Demo 3: Multi-turn Dialogue")
    print("="*50)
    history = []
    
    q1 = "<image>\nIs there an animal in the image?"
    r1, history = model.chat(tokenizer, pixel_values, q1, generation_config, 
                             history=None, return_history=True)
    print(f"[1] Q: {q1}")
    print(f"    A: {r1}\n")
    
    q2 = "What color is it?"
    r2, history = model.chat(tokenizer, pixel_values, q2, generation_config,
                             history=history, return_history=True)
    print(f"[2] Q: {q2}")
    print(f"    A: {r2}\n")
    
    q3 = "Describe the background."
    r3, history = model.chat(tokenizer, pixel_values, q3, generation_config,
                             history=history, return_history=True)
    print(f"[3] Q: {q3}")
    print(f"    A: {r3}\n")
    
    # === Demo 4: Pure Text Chat ===
    print("="*50)
    print("💭 Demo 4: Pure Text Chat (No Image)")
    print("="*50)
    q = "Hello! Who are you?"
    r, history = model.chat(tokenizer, None, q, generation_config, 
                            history=None, return_history=True)
    print(f"Q: {q}")
    print(f"A: {r}\n")
    
    print("="*50)
    print(f"✨ Demo completed! Check {OUTPUT_DIR}/ for masks.")
    print("="*50)


if __name__ == "__main__":
    main()