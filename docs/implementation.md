# Stage 1: Temporal Segmentation — Medium difficulty

- V-JEPA 2 encoder (facebook/vjepa2-vitg-fpc64-384) — available on HuggingFace. You extract frame embeddings with spatial average pooling.
- Sample 1-in-4 raw frames, create overlapping windows of 64 frames with stride 8, encode each window, average overlapping frame embeddings.
- Hierarchical agglomerative clustering with Ward linkage + temporal connectivity constraint — this is literally sklearn.cluster.AgglomerativeClustering with connectivity
set to a nearest-neighbor chain. Well-documented, straightforward.
- Prune nodes with duration < 0.5s.

What makes it tractable: All components are off-the-shelf. The tricky part is efficiently batching V-JEPA 2 inference across 1M+ videos (they spent ~1.3M V100 GPU-hours on
this + Stage 2).

---
# Stage 2: Caption Generation — Medium difficulty

Two models, both on HuggingFace, both fit on a single V100 32GB:

┌────────────┬──────────────────────────────────┬──────────────────────────────────┐
│            │            Leaf nodes            │          Non-leaf nodes          │
├────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ Model      │ LLaMA-3.2-Vision-11B             │ PerceptionLM-3B                  │
├────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ Input      │ Middle frame at 320²             │ 32 evenly-spaced frames at 320²  │
├────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ Prompt     │ "Describe this image in detail." │ "Describe this video in detail." │
├────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ Max tokens │ 1024                             │ 1024                             │
└────────────┴──────────────────────────────────┴──────────────────────────────────┘

Then organize captions into the Tree-of-Captions hierarchy (parent-child structure from Stage 1).

What makes it tractable: Simple inference calls, no fine-tuning. The prompt is trivial. The main challenge is throughput at scale.

# Stage 3 — LLM aggregation

- Use an API (Claude, GPT-4o, etc.) — no local GPU needed
- Or run a local model like Llama-3.1-70B-Q4 (~40GB) if you have 2× 24GB cards
- For a few test videos, API cost is pennies

By GPU

┌────────────────────┬─────────┬─────────────────────────────────────────────────────────────────────────────┐
│        GPU         │  VRAM   │                                 Feasibility                                 │
├────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────────┤
│ RTX 4090 / 3090    │ 24 GB   │ All stages run locally. LLaMA-11B needs 4-bit quant. Comfortable.           │
├────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────────┤
│ RTX 4070 Ti Super  │ 16 GB   │ V-JEPA 2 fits (tight). LLaMA-11B needs 4-bit. PLM-3B fine.                  │
├────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────────┤
│ RTX 4060 Ti / 3080 │ 8-12 GB │ V-JEPA 2 at fp16 may need smaller window or offloading. Rest OK with quant. │
├────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────────┤
│ RTX 4060 / 3060    │ 8 GB    │ Possible with aggressive quantization + CPU offloading, but slow.           │
└────────────────────┴─────────┴─────────────────────────────────────────────────────────────────────────────┘

Practical Estimate for "a couple of videos"

For say 5 videos of ~5 min each:

┌───────────────────────────────────────────────┬─────────────────────────────────┬─────────────────┐
│                     Stage                     │      Time (RTX 3090/4090)       │ Time (RTX 4070) │
├───────────────────────────────────────────────┼─────────────────────────────────┼─────────────────┤
│ Stage 1: V-JEPA 2 embedding + clustering      │ ~2-5 min per video              │ ~5-10 min       │
├───────────────────────────────────────────────┼─────────────────────────────────┼─────────────────┤
│ Stage 2: Captioning (~50-100 nodes/video)     │ ~10-20 min per video            │ ~20-40 min      │
├───────────────────────────────────────────────┼─────────────────────────────────┼─────────────────┤
│ Stage 3: LLM API calls (3 rounds × ~80 nodes) │ ~5-10 min per video (API-bound) │ Same            │
└───────────────────────────────────────────────┴─────────────────────────────────┴─────────────────┘

Total: ~1-2 hours for 5 videos on a 24GB card. Very doable.


Key Practical Tips

1. V-JEPA 2 is the only tricky part — it's a video transformer with 64-frame windows. Use torch.inference_mode(), fp16, batch size 1. If it
OOMs, you can reduce to 32-frame windows and interpolate.
2. Skip GPT-OSS-120B — use Claude or GPT-4o for Stage 3. The prompt is already provided. Results will be comparable or better.
3. You don't need all tree levels — for testing, you can cap the hierarchy at 3-4 levels and skip very short segments.
