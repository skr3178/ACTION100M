"""Re-cluster from saved embeddings and prune the tree (no V-JEPA 2 needed)."""
import sys
sys.path.insert(0, "/media/skr/storage/3DGS/RhodusAI/Action100M/action100m/src")

import json
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from stage1_segmentation import HierarchicalSegmenter

EMBED_PATH  = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/test_embeddings.npy"
OUTPUT_PATH = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/test_stage1_short_pruned.json"
VIDEO_PATH  = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/videos/zpcK1IzH6b8.mp4"

FPS = 30.0
SAMPLE_RATE = 4
MIN_DURATION = 0.5
CLIP_SECONDS = 120

# ── 1. Load embeddings ──────────────────────────────────────────────────────
embeddings = np.load(EMBED_PATH)
clip_frames = int(CLIP_SECONDS * FPS / SAMPLE_RATE)
clip_frames = min(clip_frames, len(embeddings))
embeddings = embeddings[:clip_frames]
print(f"Loaded embeddings: {embeddings.shape}")

# ── 2. Cluster (full dendrogram) ────────────────────────────────────────────
segmenter = HierarchicalSegmenter(linkage="ward", n_neighbors=5, min_duration=MIN_DURATION)
segments = segmenter.segment(embeddings, FPS, SAMPLE_RATE)
print(f"Full dendrogram: {len(segments)} nodes")

# ── 3. Prune ────────────────────────────────────────────────────────────────
pruned = segmenter.prune(min_duration=MIN_DURATION)
tree = segmenter.build_tree()

leaves = [s for s in pruned if s.is_leaf]
non_leaves = [s for s in pruned if not s.is_leaf]

print(f"\nPruned tree: {len(pruned)} nodes | {len(leaves)} leaves | {len(non_leaves)} non-leaves")
print(f"Leaf duration: min={min(s.duration for s in leaves):.2f}s, "
      f"max={max(s.duration for s in leaves):.2f}s, "
      f"avg={sum(s.duration for s in leaves)/len(leaves):.2f}s")

print("\nLeaf segments:")
for s in sorted(leaves, key=lambda x: x.start_time):
    print(f"  {s.start_time:.1f}s–{s.end_time:.1f}s ({s.duration:.1f}s) [L{s.level}]")

print("\nTop non-leaf segments:")
for s in sorted(non_leaves, key=lambda x: -x.duration)[:5]:
    print(f"  [L{s.level}] {s.start_time:.1f}s–{s.end_time:.1f}s ({s.duration:.1f}s)")

# ── 4. Save ──────────────────────────────────────────────────────────────────
output = {
    "video_path": VIDEO_PATH,
    "clip_seconds": CLIP_SECONDS,
    "num_sampled_frames": clip_frames,
    "min_duration_filter": MIN_DURATION,
    "num_segments": len(pruned),
    "num_leaves": len(leaves),
    "num_non_leaves": len(non_leaves),
    "tree": {str(k): v for k, v in tree.items()},
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUTPUT_PATH}")
