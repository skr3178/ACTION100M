"""Stage 3 test: LLM aggregation/refinement of Stage 2 captions."""
import sys
sys.path.insert(0, "/media/skr/storage/3DGS/RhodusAI/Action100M/action100m/src")

import json
import os
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from stage3_aggregation import LLMAggregator

# ── Config ───────────────────────────────────────────────────────────────────
STAGE1_RESULT    = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/test_stage1_short_result.json"
LEAF_CAPTIONS    = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/test_stage2_leaf_result.json"
NONLEAF_CAPTIONS = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/test_stage2_nonleaf_result.json"
PARQUET_PATH     = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/parquet/part-00028.parquet"
OUTPUT_PATH      = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/test_stage3_result.json"
VIDEO_UID        = "zpcK1IzH6b8"

MAX_NODES = 5  # limit for testing; set to None for all

stage3_config = {
    "use_api": True,
    "min_duration_seconds": 4,   # only annotate segments >= 4s
    "num_refine_rounds": 3,      # paper: 3 rounds of Self-Refine
    "api": {
        "provider": "openai",
        "model": "gpt-4o",
        "max_retries": 3,
    },
}

# ── 1. Load video metadata (title, description, ASR) from parquet ────────────
print("Loading video metadata from parquet...")
import pandas as pd

parquet_df = pd.read_parquet(PARQUET_PATH)
video_row = parquet_df[parquet_df["video_uid"] == VIDEO_UID].iloc[0]
metadata = video_row["metadata"]

# Build ASR transcript string from list of {time, text} entries
asr_entries = metadata.get("transcript", [])
try:
    asr_text = " ".join(entry.get("text", "") for entry in asr_entries if isinstance(entry, dict))
except Exception:
    asr_text = str(asr_entries)[:2000]

video_context = {
    "title": metadata.get("title", ""),
    "description": metadata.get("description", ""),
    "asr_transcript": asr_text,
}
print(f"  Title: {video_context['title']}")
print(f"  ASR length: {len(asr_text)} chars")

# ── 2. Load Stage 1 tree + Stage 2 captions ─────────────────────────────────
print("Loading Stage 1 + Stage 2 results...")
with open(STAGE1_RESULT) as f:
    stage1 = json.load(f)
with open(LEAF_CAPTIONS) as f:
    leaf_data = json.load(f)
with open(NONLEAF_CAPTIONS) as f:
    nonleaf_data = json.load(f)

# Build tree_of_captions structure expected by Stage 3
tree_of_captions = {
    "captions": {},
    "tree": {},
}

# Add tree nodes from Stage 1
for node_id, node_info in stage1["tree"].items():
    tree_of_captions["tree"][node_id] = node_info

# Add leaf captions from Stage 2
for cap in leaf_data["captions"]:
    nid = str(cap["node_id"])
    tree_of_captions["captions"][nid] = {
        "caption": cap["caption"],
        "model_used": leaf_data["model"],
        "num_frames": 1,
        "resolution": 320,
    }

# Add non-leaf captions from Stage 2
for cap in nonleaf_data["captions"]:
    nid = str(cap["node_id"])
    tree_of_captions["captions"][nid] = {
        "caption": cap["caption"],
        "model_used": nonleaf_data["model"],
        "num_frames": cap["num_frames_used"],
        "resolution": 320,
    }

captioned_nodes = list(tree_of_captions["captions"].keys())
print(f"Tree: {len(tree_of_captions['tree'])} nodes | Captions available: {len(captioned_nodes)}")

# ── 2. Run Stage 3 on captioned non-leaf nodes ──────────────────────────────
# Only process nodes that have captions AND meet min_duration
eligible = []
for nid in captioned_nodes:
    node = tree_of_captions["tree"].get(nid, {})
    if node.get("duration", 0) >= stage3_config["min_duration_seconds"]:
        eligible.append(nid)

eligible.sort(key=lambda x: -tree_of_captions["tree"][x]["duration"])
if MAX_NODES:
    eligible = eligible[:MAX_NODES]

print(f"Will process {len(eligible)} nodes (duration >= {stage3_config['min_duration_seconds']}s)")
for nid in eligible:
    n = tree_of_captions["tree"][nid]
    has_cap = "yes" if nid in tree_of_captions["captions"] else "no"
    print(f"  [{nid}] L{n['level']} {n['start_time']:.1f}s–{n['end_time']:.1f}s ({n['duration']:.1f}s) caption={has_cap}")

# ── 3. Process each node ────────────────────────────────────────────────────
aggregator = LLMAggregator(stage3_config)
aggregator._init_api_client()

if aggregator.client is None:
    print("\nERROR: No API client initialized. Set ANTHROPIC_API_KEY environment variable.")
    print("  export ANTHROPIC_API_KEY='sk-ant-...'")
    sys.exit(1)

print(f"\nUsing API: {stage3_config['api']['provider']} / {stage3_config['api']['model']}")
print(f"Refine rounds: {stage3_config['num_refine_rounds']}")

results = []
for nid in eligible:
    node = tree_of_captions["tree"][nid]
    print(f"\nProcessing node {nid} ({node['start_time']:.1f}s–{node['end_time']:.1f}s)...")

    annotation = aggregator.process_node(nid, tree_of_captions, context=video_context)
    if annotation:
        results.append({
            "node_id": annotation.node_id,
            "start_time": node["start_time"],
            "end_time": node["end_time"],
            "duration": node["duration"],
            "level": node["level"],
            "brief_action": annotation.brief_action,
            "detailed_action": annotation.detailed_action,
            "actor": annotation.actor,
            "brief_caption": annotation.brief_caption,
            "detailed_caption": annotation.detailed_caption,
        })
        print(f"  Action: {annotation.brief_action}")
        print(f"  Actor: {annotation.actor}")
        print(f"  Brief: {annotation.brief_caption[:100]}")

# ── 4. Save ──────────────────────────────────────────────────────────────────
output = {
    "api_provider": stage3_config["api"]["provider"],
    "api_model": stage3_config["api"]["model"],
    "num_refine_rounds": stage3_config["num_refine_rounds"],
    "num_annotations": len(results),
    "annotations": results,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(results)} annotations to {OUTPUT_PATH}")
print("DONE ✓")
