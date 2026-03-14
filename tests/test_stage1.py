"""Stage 1 test: V-JEPA 2 encoding + hierarchical segmentation."""
import sys
sys.path.insert(0, "/media/skr/storage/3DGS/RhodusAI/Action100M/action100m/src")

import os
import time
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL_PATH  = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/models/vjepa2"
VIDEO_PATH  = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/videos/zpcK1IzH6b8.mp4"
EMBED_CACHE = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/test_embeddings.npy"

config = {
    "model_name": MODEL_PATH,
    "model_device": "cuda",
    "frame_sample_rate": 4,
    "window_size": 64,
    "window_stride": 8,
    "min_duration_seconds": 0.5,
    "clustering": {"linkage": "ward", "n_neighbors": 5},
}

# ── 1. Init stage (single model load) ────────────────────────────────────────
from stage1_segmentation import TemporalSegmentationStage
stage1 = TemporalSegmentationStage(config)
stage1.load_model()
print(f"Model loaded — hidden_size: {stage1.encoder.model.config.hidden_size}")

# ── 2. Open video ─────────────────────────────────────────────────────────────
vr, fps, duration, num_sampled = stage1.video_loader.open_video(VIDEO_PATH)
print(f"Video: {duration:.1f}s | {num_sampled} sampled frames (1-in-4)")

# ── 3. Single-window sanity check ─────────────────────────────────────────────
print("\n=== Single window test ===")
window_frames = stage1.video_loader.get_window_frames(vr, 0, 64)
emb = stage1.encoder.encode_frames(window_frames)
print(f"Window embeddings shape: {emb.shape}")  # expect (64, 1408)

# ── 4. Encode half the video (streaming) ──────────────────────────────────────
clip_sampled = num_sampled // 2
print(f"\n=== Encoding {clip_sampled} sampled frames ({duration/2:.0f}s clip) ===")

if os.path.exists(EMBED_CACHE):
    print(f"Loading cached embeddings from {EMBED_CACHE}")
    embeddings = np.load(EMBED_CACHE)
else:
    t0 = time.time()
    n_windows = (clip_sampled - 64) // 8 + 1
    print(f"~{n_windows} windows × 6-7s ≈ {n_windows * 6.5 / 60:.0f} min estimated")

    vr2, _, _, _ = stage1.video_loader.open_video(VIDEO_PATH)
    embeddings = stage1.encoder.encode_windows(
        video_loader=stage1.video_loader,
        vr=vr2,
        num_sampled_frames=clip_sampled,
        window_size=64,
        stride=8,
    )
    np.save(EMBED_CACHE, embeddings)
    print(f"Done in {(time.time()-t0)/60:.1f} min — saved to {EMBED_CACHE}")
    print(f"Embeddings shape: {embeddings.shape}")

# ── 5. Cluster ────────────────────────────────────────────────────────────────
print("\n=== Clustering ===")
segments = stage1.segmenter.segment(embeddings, fps, config["frame_sample_rate"])
tree = stage1.segmenter.build_tree()

leaves     = [s for s in segments if s.is_leaf]
non_leaves = [s for s in segments if not s.is_leaf]
print(f"Total: {len(segments)} | Leaves: {len(leaves)} | Non-leaves: {len(non_leaves)}")
print(f"Tree nodes: {len(tree)}")

print("\nLongest non-leaf segments:")
for s in sorted(non_leaves, key=lambda x: -x.duration)[:5]:
    print(f"  [L{s.level}] {s.start_time:.1f}s-{s.end_time:.1f}s ({s.duration:.1f}s)")
