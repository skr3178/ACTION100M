"""Stage 2 test: Non-leaf segment captioning (multi-frame) with Qwen3-VL-4B-AWQ."""
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
VIDEO_PATH      = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/videos/zpcK1IzH6b8_90_120s.mp4"
STAGE1_RESULT   = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/test_stage1_short_pruned.json"
OUTPUT_PATH     = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/test_stage2_nonleaf_result.json"
MODEL_PATH      = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/models/qwen3-vl-4b-awq"
SAMPLE_RATE     = 4
RESOLUTION      = 320
FRAMES_PER_SEG  = 32    # paper: 32 evenly spaced frames per segment
MAX_TOKENS      = 1024  # paper: generation limit of 1024 tokens
MAX_SEGMENTS    = 10    # limit for testing; set to None for all non-leaves

# ── 1. Load Stage 1 results ─────────────────────────────────────────────────
print("Loading Stage 1 results...")
with open(STAGE1_RESULT) as f:
    stage1 = json.load(f)

tree = stage1["tree"]
non_leaf_nodes = [v for v in tree.values() if not v["is_leaf"]]
non_leaf_nodes.sort(key=lambda x: -x["duration"])  # longest first

if MAX_SEGMENTS:
    non_leaf_nodes = non_leaf_nodes[:MAX_SEGMENTS]

print(f"Will caption {len(non_leaf_nodes)} non-leaf segments")
for n in non_leaf_nodes[:5]:
    print(f"  [L{n['level']}] {n['start_time']:.1f}s–{n['end_time']:.1f}s ({n['duration']:.1f}s)")

# ── 2. Open video ───────────────────────────────────────────────────────────
vr = decord.VideoReader(VIDEO_PATH, ctx=decord.cpu(0))
fps = vr.get_avg_fps()
total_frames = len(vr)
print(f"Video opened: {total_frames} frames, {fps:.1f} fps")

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

# ── 4. Caption each non-leaf segment ─────────────────────────────────────────
prompt = "Describe this video in detail."
results = []

t0 = time.time()
for node in tqdm(non_leaf_nodes, desc="Captioning segments"):
    # Sample evenly-spaced frames from this segment's time range
    start_frame = node["start_frame"]
    end_frame = min(node["end_frame"], total_frames - 1)

    if end_frame <= start_frame:
        continue

    n_sample = min(FRAMES_PER_SEG, end_frame - start_frame)
    frame_indices = np.linspace(start_frame, end_frame - 1, n_sample, dtype=int).tolist()

    # Load frames via decord
    raw_frames = vr.get_batch(frame_indices).asnumpy()
    images = [Image.fromarray(f).resize((RESOLUTION, RESOLUTION)) for f in raw_frames]

    # Qwen3-VL chat format — multiple images
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    caption = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    results.append({
        "node_id": node["node_id"],
        "level": node["level"],
        "start_time": node["start_time"],
        "end_time": node["end_time"],
        "duration": node["duration"],
        "num_frames_used": len(images),
        "caption": caption,
    })

    # Print first few for sanity check
    if len(results) <= 3:
        print(f"  [L{node['level']}] {node['start_time']:.1f}s–{node['end_time']:.1f}s: {caption[:120]}")

elapsed = time.time() - t0
print(f"\nDone: {len(results)} captions in {elapsed:.0f}s ({elapsed/len(results):.1f}s/segment)")

# ── 5. Save ──────────────────────────────────────────────────────────────────
output = {
    "video_path": VIDEO_PATH,
    "model": MODEL_PATH,
    "frames_per_segment": FRAMES_PER_SEG,
    "num_segments_captioned": len(results),
    "elapsed_seconds": round(elapsed, 1),
    "captions": results,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Saved to {OUTPUT_PATH}")
