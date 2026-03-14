"""Stage 1 short test: 2 minutes of video — validate full pipeline before long runs."""
import sys
sys.path.insert(0, "/media/skr/storage/3DGS/RhodusAI/Action100M/action100m/src")

import json
import os
import time
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL_PATH = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/models/vjepa2"
VIDEO_PATH = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/videos/zpcK1IzH6b8_90_120s.mp4"
OUTPUT_DIR = "/media/skr/storage/3DGS/RhodusAI/Action100M/data"

CLIP_SECONDS = 9999  # process full video

config = {
    "model_name": MODEL_PATH,
    "model_device": "cuda",
    "frame_sample_rate": 4,
    "window_size": 64,
    "window_stride": 8,
    "min_duration_seconds": 0.5,
    "clustering": {"linkage": "ward", "n_neighbors": 5},
}

# ── 1. Load model ───────────────────────────────────────────────────────────
from stage1_segmentation import TemporalSegmentationStage
stage1 = TemporalSegmentationStage(config)
stage1.load_model()
print(f"Model loaded — hidden_size: {stage1.encoder.model.config.hidden_size}")

# ── 2. Open video and compute 2-min clip size ───────────────────────────────
vr, fps, duration, num_sampled = stage1.video_loader.open_video(VIDEO_PATH)
clip_frames = int(CLIP_SECONDS * fps / config["frame_sample_rate"])
clip_frames = min(clip_frames, num_sampled)
print(f"Video: {duration:.1f}s total | Using first {CLIP_SECONDS}s → {clip_frames} sampled frames")

# ── 3. Encode (with cache) ───────────────────────────────────────────────────
EMBED_CACHE = os.path.join(OUTPUT_DIR, "test_embeddings_short.npy")

if os.path.exists(EMBED_CACHE):
    print(f"Loading cached embeddings from {EMBED_CACHE}")
    embeddings = np.load(EMBED_CACHE)[:clip_frames]
    elapsed = 0
    print(f"Loaded embeddings: {embeddings.shape}")
else:
    n_windows = max(1, (clip_frames - 64) // 8 + 1)
    print(f"\n=== Encoding: {n_windows} windows × ~6.5s ≈ {n_windows * 6.5 / 60:.1f} min ===")

    t0 = time.time()
    vr2, _, _, _ = stage1.video_loader.open_video(VIDEO_PATH)
    embeddings = stage1.encoder.encode_windows(
        video_loader=stage1.video_loader,
        vr=vr2,
        num_sampled_frames=clip_frames,
        window_size=64,
        stride=8,
    )
    elapsed = time.time() - t0
    np.save(EMBED_CACHE, embeddings)
    print(f"Encoding done in {elapsed:.1f}s ({elapsed/60:.1f} min) — saved to {EMBED_CACHE}")
    print(f"Embeddings shape: {embeddings.shape}")

# ── 4. Cluster ───────────────────────────────────────────────────────────────
print("\n=== Clustering ===")
segments = stage1.segmenter.segment(embeddings, fps, config["frame_sample_rate"])
tree = stage1.segmenter.build_tree()

leaves = [s for s in segments if s.is_leaf]
non_leaves = [s for s in segments if not s.is_leaf]
print(f"Total: {len(segments)} | Leaves: {len(leaves)} | Non-leaves: {len(non_leaves)}")
print(f"Tree nodes: {len(tree)}")

# ── 5. Show top segments ─────────────────────────────────────────────────────
print("\nLongest non-leaf segments:")
for s in sorted(non_leaves, key=lambda x: -x.duration)[:5]:
    print(f"  [L{s.level}] {s.start_time:.1f}s–{s.end_time:.1f}s ({s.duration:.1f}s)")

print("\nSample leaf segments (first 10):")
for s in sorted(leaves, key=lambda x: x.start_time)[:10]:
    print(f"  {s.start_time:.2f}s–{s.end_time:.2f}s ({s.duration:.2f}s)")

# ── 6. Save results ─────────────────────────────────────────────────────────
output_path = os.path.join(OUTPUT_DIR, "test_stage1_short_result.json")
result = {
    "video_path": VIDEO_PATH,
    "clip_seconds": CLIP_SECONDS,
    "num_sampled_frames": clip_frames,
    "encoding_time_seconds": round(elapsed, 1),
    "num_segments": len(segments),
    "num_leaves": len(leaves),
    "num_non_leaves": len(non_leaves),
    "tree": {str(k): v for k, v in tree.items()},
}
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nFull tree saved to {output_path}")

# ── 7. Prune tree (collapse leaves < 0.5s) ──────────────────────────────────
print("\n=== Pruning ===")
pruned_segments = stage1.segmenter.prune(min_duration=config["min_duration_seconds"])
pruned_tree = stage1.segmenter.build_tree()

pruned_leaves = [s for s in pruned_segments if s.is_leaf]
pruned_non_leaves = [s for s in pruned_segments if not s.is_leaf]
print(f"Pruned: {len(pruned_segments)} nodes | {len(pruned_leaves)} leaves | {len(pruned_non_leaves)} non-leaves")
print(f"Leaf duration: min={min(s.duration for s in pruned_leaves):.2f}s, "
      f"max={max(s.duration for s in pruned_leaves):.2f}s, "
      f"avg={sum(s.duration for s in pruned_leaves)/len(pruned_leaves):.2f}s")

print("\nPruned leaf segments:")
for s in sorted(pruned_leaves, key=lambda x: x.start_time):
    print(f"  {s.start_time:.1f}s–{s.end_time:.1f}s ({s.duration:.1f}s) [L{s.level}]")

pruned_path = os.path.join(OUTPUT_DIR, "test_stage1_short_pruned.json")
pruned_result = {
    "video_path": VIDEO_PATH,
    "clip_seconds": CLIP_SECONDS,
    "num_sampled_frames": clip_frames,
    "encoding_time_seconds": round(elapsed, 1),
    "min_duration_filter": config["min_duration_seconds"],
    "num_segments": len(pruned_segments),
    "num_leaves": len(pruned_leaves),
    "num_non_leaves": len(pruned_non_leaves),
    "tree": {str(k): v for k, v in pruned_tree.items()},
}
with open(pruned_path, "w") as f:
    json.dump(pruned_result, f, indent=2)
print(f"Pruned tree saved to {pruned_path}")
print("DONE ✓")
