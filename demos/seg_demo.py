#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARGenSeg Demo: Referring Expression Segmentation

This script demonstrates image segmentation with natural language instructions.

Usage:
    python demos/seg_demo_simple.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import time
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from internvl.model.internvl_chat import InternVLGenSeg
from internvl.train.dataset import build_transform


def load_image(image_path: str, input_size: int = 448) -> torch.Tensor:
    """Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        input_size: Input size for the vision transformer
        
    Returns:
        Preprocessed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size, pad2square=False)
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values


def main():
    """Main demo function."""
    # ============ Configuration ============
    MODEL_PATH = 'pretrained/InternVL2_5-ARGenSeg-8B'
    DEFAULT_IMAGE = './assets/teddy.jpg'
    DEFAULT_QUERY = "Given the following instructions: bear the child is hugging; please perform referring segmentation on this image."
    OUTPUT_DIR = './results'
    time_stamp = time.strftime("%m-%d-%H-%M")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ============ Load Model ============
    print(f"🚀 Loading model from {MODEL_PATH}...")
    model = InternVLGenSeg.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        use_fast=False
    )
    print(f"✅ Model loaded successfully!")
    
    # ============ Load Image ============
    image_path = DEFAULT_IMAGE
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    pixel_values = load_image(image_path).to(torch.bfloat16).cuda()
    image = Image.open(image_path)
    print(f"🖼️  Loaded image: {image_path} ({image.width}x{image.height})")
    
    # ============ Build Query ============
    question = f"<image>\n{DEFAULT_QUERY}"
    generation_config = dict(max_new_tokens=4096, do_sample=False)
    
    # ============ Model Inference ============
    print(f"🔮 Running inference...")
    response = model.chat(
        tokenizer, 
        pixel_values, 
        question, 
        generation_config, 
        return_multi_results=True
    )
    
    # ============ Save Results ============
    print(f"💾 Saving results to {OUTPUT_DIR}/...")
    
    # Save all generated masks
    for idx, mask_img in enumerate(model.language_model.gen_cache):
        mask_resized = mask_img.resize(image.size)
        save_path = f"{OUTPUT_DIR}/mask_{Path(image_path).stem}_{idx}.png"
        mask_resized.save(save_path)
        print(f"   ✓ {save_path}")
    
    # Save visualization overlay
    if hasattr(model.language_model, 'multi_mask'):
        from internvl.train.dataset import vis_multi_mask
        vis_multi_mask(image_path, model.language_model.multi_mask, time_stamp)
    
    # Clear cache
    model.language_model.gen_cache = []
    
    # ============ Print Results ============
    print(f"\n{'='*60}")
    print(f"Query: {DEFAULT_QUERY}")
    print(f"Response: {response}")
    print(f"{'='*60}")
    
    print(f"\n✨ Demo completed! Check {OUTPUT_DIR}/ for results.")


if __name__ == "__main__":
    main()