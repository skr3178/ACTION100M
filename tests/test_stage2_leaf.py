"""Stage 2 test: Leaf node captioning (single frame per leaf) with Qwen3-VL-4B-AWQ."""
import sys
sys.path.insert(0, "/media/skr/storage/3DGS/RhodusAI/Action100M/action100m/src")

import json
import os
import time
import torch
import numpy as np
import logging
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import decord
decord.bridge.set_bridge("native")

# ── Config ───────────────────────────────────────────────────────────────────
VIDEO_PATH     = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/videos/zpcK1IzH6b8.mp4"
STAGE1_RESULT  = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/test_stage1_short_result.json"
OUTPUT_PATH    = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/test_stage2_leaf_result.json"
MODEL_PATH     = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/models/qwen3-vl-4b-awq"
SAMPLE_RATE    = 4
RESOLUTION     = 320
MAX_TOKENS     = 256   # shorter for speed; increase to 1024 for production
MAX_LEAVES     = 20    # limit for testing; set to None for all leaves

# ── 1. Load Stage 1 results ─────────────────────────────────────────────────
print("Loading Stage 1 results...")
with open(STAGE1_RESULT) as f:
    stage1 = json.load(f)

tree = stage1["tree"]
leaf_nodes = [v for v in tree.values() if v["is_leaf"]]
leaf_nodes.sort(key=lambda x: x["start_time"])

if MAX_LEAVES:
    # Sample evenly across the video for variety
    indices = np.linspace(0, len(leaf_nodes) - 1, MAX_LEAVES, dtype=int)
    leaf_nodes = [leaf_nodes[i] for i in indices]

print(f"Will caption {len(leaf_nodes)} leaf nodes")

# ── 2. Open video ───────────────────────────────────────────────────────────
vr = decord.VideoReader(VIDEO_PATH, ctx=decord.cpu(0))
fps = vr.get_avg_fps()
print(f"Video opened: {len(vr)} frames, {fps:.1f} fps")

# ── 3. Load Qwen3-VL ────────────────────────────────────────────────────────
print("Loading Qwen3-VL-4B-AWQ...")
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded on CUDA")

# ── 4. Caption each leaf (middle frame) ──────────────────────────────────────
prompt = "Describe this image in one sentence. Focus on the main action or activity."
results = []

t0 = time.time()
for node in tqdm(leaf_nodes, desc="Captioning leaves"):
    # Get middle frame in original-frame space
    mid_frame = (node["start_frame"] + node["end_frame"]) // 2
    mid_frame = min(mid_frame, len(vr) - 1)

    frame = vr[mid_frame].asnumpy()
    image = Image.fromarray(frame).resize((RESOLUTION, RESOLUTION))

    # Qwen3-VL chat format
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    caption = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    results.append({
        "node_id": node["node_id"],
        "start_time": node["start_time"],
        "end_time": node["end_time"],
        "mid_frame": mid_frame,
        "caption": caption,
    })
    # Print first few for sanity check
    if len(results) <= 3:
        print(f"  [{node['start_time']:.1f}s] {caption[:100]}")

elapsed = time.time() - t0
print(f"\nDone: {len(results)} captions in {elapsed:.0f}s ({elapsed/len(results):.1f}s/leaf)")

# ── 5. Save ──────────────────────────────────────────────────────────────────
output = {
    "video_path": VIDEO_PATH,
    "model": MODEL_PATH,
    "num_leaves_captioned": len(results),
    "elapsed_seconds": round(elapsed, 1),
    "captions": results,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Saved to {OUTPUT_PATH}")
