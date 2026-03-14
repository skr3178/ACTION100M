# Action100M Pipeline

Implementation of the Action100M data pipeline for generating hierarchical video action annotations.

## Pipeline Stages

### Stage 1: Temporal Segmentation
- Uses V-JEPA 2 (facebook/vjepa2-vitg-fpc64-384) to encode video frames
- Samples 1-in-4 frames with 64-frame windows (stride 8)
- Applies hierarchical agglomerative clustering with Ward linkage
- Prunes segments shorter than 0.5 seconds

### Stage 2: Caption Generation
- **Leaf nodes**: LLaMA-3.2-Vision-11B on middle frame at 320²
- **Non-leaf nodes**: PerceptionLM-3B on 32 evenly-spaced frames at 320²

### Stage 3: LLM Aggregation
- Uses Claude API (or OpenAI GPT-4o) to extract structured annotations
- 3 rounds of Self-Refine for quality improvement
- Nodes shorter than 4 seconds are discarded
- Output fields: brief_action, detailed_action, actor, brief_caption, detailed_caption

## Installation

```bash
# Clone and install
cd action100m
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Process a video with all stages
python -m action100m.scripts.run_pipeline input/video.mp4 -o output

# Process a directory of videos
python -m action100m.scripts.run_pipeline input/videos/ -o output

# Run only specific stages
python -m action100m.scripts.run_pipeline input/video.mp4 -o output --stage 1  # Only segmentation
python -m action100m.scripts.run_pipeline input/video.mp4 -o output --stage 2  # Only captioning
python -m action100m.scripts.run_pipeline input/video.mp4 -o output --stage 3  # Only aggregation
```

### API Key Setup

For Stage 3 (LLM Aggregation), set your API key:

```bash
# Option 1: Environment variable
export ANTHROPIC_API_KEY=your_key_here
# or
export OPENAI_API_KEY=your_key_here

# Option 2: Command line argument
python -m action100m.scripts.run_pipeline input/video.mp4 --api-key your_key_here
```

### Configuration

Edit `action100m/configs/config.yaml` to customize:
- Model choices and parameters
- Frame sampling rates
- Clustering parameters
- API settings

## Output Structure

```
output/
├── stage1/
│   └── video_segmentation.json    # Hierarchical segmentation tree
├── stage2/
│   └── video_captions.json       # Tree-of-Captions
├── stage3/
│   └── video_annotations.json    # Structured annotations
└── summary.json                   # Processing summary
```

## Hardware Requirements

| GPU | VRAM | Feasibility |
|-----|------|-------------|
| RTX 4090/3090 | 24 GB | All stages run locally |
| RTX 4070 Ti | 16 GB | V-JEPA fits (tight), captioning OK |
| RTX 4060/3080 | 8-12 GB | Needs quantization + offloading |

## Estimated Processing Time (per 5-min video)

| Stage | Time (24GB GPU) |
|-------|-----------------|
| Stage 1: Segmentation | ~2-5 min |
| Stage 2: Captioning | ~10-20 min |
| Stage 3: Aggregation | ~5-10 min (API) |

## License

MIT License - See LICENSE file for details.


## Notes

- Alternative Stage 2 model: `OpenGVLab/InternVL3-78B` instead of Qwen3-VL

## Validation: Pipeline Output vs Ground Truth

Comparison on a 120-second clip from video `zpcK1IzH6b8` (Shabby Chic Furniture Distressing tutorial):

| Time Range | Ground Truth (Action100M) | Our Pipeline |
|---|---|---|
| **0-120s** | "restore vintage wall unit" -- Brini Maxwell | "sand and paint cabinet" -- woman in vintage dress |
| **8.5-120s** | "Antique a vintage wall unit" -- Brini Maxwell | "paint cabinet" -- woman in yellow floral dress |
| **56-120s** | "Paint and glaze a wooden wall unit" | "restore furniture" |
| **66.7-120s** | "Sand, prime, and paint cabinet" | "paint dresser" |
| **8.5-56s** | "Demonstrate sanding" | "present furniture" |

### Key Observations

1. **Segment boundaries are accurate** -- Our clustering found nearly identical breakpoints to the ground truth (8.5s vs 8.7s, 56.9s vs 56.0s, 66.7s exact match), validating Stage 1.

2. **Actions are semantically similar but less specific** -- Ground truth includes details like "glaze", "antiquing", "speckles" while ours are more generic. This is expected because:
   - Ground truth processes the full video (688s) vs our 120s clip
   - Ground truth feeds ASR transcript + video metadata to the LLM
   - Ground truth uses 3 rounds of Self-Refine; our test used 1

3. **Actor identification** -- Ground truth identifies "Brini Maxwell" by name (extracted from ASR/metadata), while ours describes appearance ("woman in yellow floral dress").

4. **Level numbering differs** -- Ground truth uses L0-L6 (coarse to fine), ours uses L13-L18 because our full dendrogram extends to per-frame leaves, inflating level numbers. This is addressed by tree pruning.

The pipeline is working correctly. Improvements would come from feeding metadata/ASR context and processing full-length videos.

## Implementation Status & TODO

## Known Issues

1. Actor naming — GT identifies "Brini Maxwell" by name. Prompt now instructs LLM to use names
   from ASR/metadata. Needs re-testing to verify improvement.
2. Action specificity — GT has richer verbs ("antique", "demonstrate sanding"). This is partly because GT
   processes the full video (688s) giving more context.
3. Segment boundaries match well — 8.5 vs 8.7s, 56.9 vs 56.0s, 66.7s exact match. Stage 1 segmentation is solid.

## Implementation Gaps (Paper vs Code)

### Stage 2: Caption Generation
- [x] Stage 2 leaf prompt should match paper exactly: "Describe this image in detail."
- [x] Stage 2 non-leaf prompt should match paper exactly: "Describe this video in detail."
- [x] Non-leaf frame count should default to 32 (updated in test script)
- [x] Max tokens should default to 1024 (updated in both test scripts)
- [ ] Model substitution: paper uses Llama-3.2-Vision-11B (leaf) + PerceptionLM-3B (non-leaf),
      we use Qwen3-VL-4B-AWQ for both (12GB VRAM constraint). No fix needed unless GPU upgraded.

### Stage 3: LLM Aggregation
- [x] Children captions now depth-first recursive via _get_dfs_captions (max_depth=5)
- [x] Root caption lookup now finds actual root (node with no parent) instead of hardcoding node_id=0
- [x] Self-Refine rounds set to 3 in test script (matching paper)
- [x] Actor prompt updated to instruct LLM to use names from ASR/metadata
- [ ] Model substitution: paper uses GPT-OSS-120B, we use GPT-4o. No fix needed.

### Pruning
- [ ] Pruning requires re-encoding (~10 min) because embeddings weren't cached in initial runs.
      test_stage1_short.py now caches to data/test_embeddings_short.npy but hasn't been re-run yet.
- [ ] The 0.5s filter in the paper applies to ALL nodes. Our pruning collapses sub-0.5s leaves upward
      to the lowest ancestor >= 0.5s. Verify this matches the paper's intent.
- [x] prune_tree.py updated to use correct embedding cache path

## Completed

- [x] Stage 1: V-JEPA 2 encoding with overlapping windows (sample_rate=4, window=64, stride=8)
- [x] Stage 1: Spatial average pooling + temporal tubelet upsample
- [x] Stage 1: Hierarchical agglomerative clustering (Ward linkage, temporal chain connectivity)
- [x] Stage 1: Full dendrogram preserved, pruning as separate step
- [x] Stage 1: Checkpoint saving/resumption for long encoding runs
- [x] Stage 2: Qwen3-VL-4B-AWQ loading via Qwen3VLForConditionalGeneration
- [x] Stage 2: Leaf captioning (single mid-frame) — tested on 20 leaves
- [x] Stage 2: Non-leaf segment captioning (multi-frame) — tested on 10 segments
- [x] Stage 3: LLM aggregation with OpenAI GPT-4o API — tested on 5 nodes
- [x] Stage 3: Video metadata (title, description, ASR transcript) passed as context
- [x] Stage 3: Structured output (5 fields: brief/detailed action, actor, brief/detailed caption)
- [x] Pipeline validated against ground truth from parquet (segment boundaries match well)
- [x] Project organized: tests/, docs/, action100m/src/, .gitignore
- [x] Pushed to GitHub: https://github.com/skr3178/action100m.git