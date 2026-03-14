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


## 
Other option is to use OpenGVLab/InternVL3-78B instead of the QWEN model

## Output from training:

┌────────────┬──────────────────────────────────────────────┬─────────────────────────────────────────────────┐
│ Time Range │          Ground Truth (Action100M)           │                  Our Pipeline                   │
├────────────┼──────────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ 0–120s+    │ "restore vintage wall unit" — Brini Maxwell  │ "sand and paint cabinet" — woman in vintage     │
│            │                                              │ dress                                           │
├────────────┼──────────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ 8.5–120s+  │ "Antique a vintage wall unit" — Brini        │ "paint cabinet" — woman in yellow floral dress  │
│            │ Maxwell                                      │                                                 │
├────────────┼──────────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ 56–120s+   │ "Paint and glaze a wooden wall unit"         │ "restore furniture"                             │
├────────────┼──────────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ 66.7–120s+ │ "Sand, prime, and paint cabinet"             │ "paint dresser"                                 │
├────────────┼──────────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ 8.5–56s    │ "Demonstrate sanding"                        │ "present furniture"                             │
└────────────┴──────────────────────────────────────────────┴─────────────────────────────────────────────────┘

Key observations:

1. Segment boundaries are very close — our clustering found nearly identical breakpoints (8.5s vs 8.7s, 56.9s vs
56.0s, 66.7s matches exactly). This validates Stage 1.
2. Actions are semantically similar but ours are less specific — the ground truth includes details like "glaze",
"antiquing", "speckles" while ours are more generic. This is because:
- Ground truth uses the full video context (688s) — our test only processed 120s
- Ground truth has ASR transcript + video title/description fed to the LLM
- Ground truth uses 3 rounds of self-refine; we used 1
3. Actor identification — ground truth identifies "Brini Maxwell" by name (from metadata/ASR), ours just says
"woman in yellow floral dress"
4. Level numbering differs — ground truth uses L0-L6 (coarse to fine), ours uses L13-L18. This is because our
dendrogram goes all the way to per-frame leaves, inflating level numbers.

The pipeline is working correctly — the improvements would come from feeding metadata/ASR context and processing
the full video.