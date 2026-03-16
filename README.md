# Action100M Pipeline

Implementation of the Action100M data pipeline for generating hierarchical video action annotations.

## Data Source

The video files used in this repository were sourced from the [HowTo100M](https://www.cs.cmu.edu/~howto100m/) dataset. HowTo100M is a large-scale dataset of instructional videos from YouTube, containing over 1 million videos spanning 23,000 categories. The YouTube video IDs and other metadata are provided in the dataset.

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

### Running the Test Scripts (nohup)

Run each stage sequentially — wait for the previous stage to finish before starting the next.

```bash
# Stage 1: Temporal Segmentation (with embedding cache)
nohup python tests/test_stage1_short.py > tests/logs/stage1.log 2>&1 &
tail -f tests/logs/stage1.log

# Stage 2a: Leaf captioning (single mid-frame per leaf segment)
nohup python tests/test_stage2_leaf.py > tests/logs/stage2_leaf.log 2>&1 &
tail -f tests/logs/stage2_leaf.log

# Stage 2b: Non-leaf captioning (multi-frame per segment)
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True python tests/test_stage2_nonleaf.py > tests/logs/stage2_nonleaf.log 2>&1 &
tail -f tests/logs/stage2_nonleaf.log

# Stage 3: LLM aggregation (requires OPENAI_API_KEY)
export OPENAI_API_KEY=your_key_here
nohup python tests/test_stage3.py > tests/logs/stage3.log 2>&1 &
tail -f tests/logs/stage3.log
```
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