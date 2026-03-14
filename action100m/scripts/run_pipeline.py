#!/usr/bin/env python3
"""
Action100M Pipeline - Main Orchestration Script

This script runs the complete Action100M data pipeline:
1. Stage 1: Temporal Segmentation using V-JEPA 2
2. Stage 2: Caption Generation using VLMs
3. Stage 3: LLM Aggregation for structured annotations
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from action100m.src.stage1_segmentation import TemporalSegmentationStage
from action100m.src.stage2_captioning import CaptionGenerationStage
from action100m.src.stage3_aggregation import LLMAggregationStage


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_video_files(input_path: str) -> List[str]:
    """Get list of video files from input path."""
    video_extensions = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv", ".wmv"}

    path = Path(input_path)

    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        videos = []
        for ext in video_extensions:
            videos.extend(str(p) for p in path.rglob(f"*{ext}"))
        return sorted(videos)
    else:
        raise ValueError(f"Invalid input path: {input_path}")


def process_single_video(
    video_path: str,
    config: Dict[str, Any],
    output_dir: str,
    stage: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a single video through the pipeline."""

    logger = logging.getLogger(__name__)
    video_name = Path(video_path).stem

    logger.info(f"=" * 60)
    logger.info(f"Processing: {video_path}")
    logger.info(f"=" * 60)

    result = {"video_path": video_path, "video_name": video_name}

    # Stage 1: Temporal Segmentation
    if stage is None or stage == "1" or stage == "all":
        logger.info("=" * 40)
        logger.info("Stage 1: Temporal Segmentation")
        logger.info("=" * 40)

        stage1_config = config["stage1"]
        segmenter = TemporalSegmentationStage(stage1_config)

        # Load model (if available)
        try:
            segmenter.load_model()
        except Exception as e:
            logger.warning(f"Could not load V-JEPA 2 model: {e}")
            logger.warning("Using mock encoder for testing")

        # Process video
        segmentation_result = segmenter.process_video(video_path)

        # Save intermediate result
        output_path = Path(output_dir) / "stage1" / f"{video_name}_segmentation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert segments to serializable format
        serializable = {
            "video_path": segmentation_result["video_path"],
            "total_duration": segmentation_result["total_duration"],
            "fps": segmentation_result["fps"],
            "num_sampled_frames": segmentation_result["num_sampled_frames"],
            "tree": segmentation_result["tree"],
        }

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Saved segmentation to {output_path}")
        result["segmentation"] = segmentation_result

    # Load segmentation result if not just doing Stage 1
    if stage == "2" or stage == "3" or stage == "all":
        if "segmentation" not in result:
            seg_path = Path(output_dir) / "stage1" / f"{video_name}_segmentation.json"
            if seg_path.exists():
                with open(seg_path, "r") as f:
                    result["segmentation"] = json.load(f)
            else:
                raise FileNotFoundError(f"Segmentation result not found: {seg_path}")

    # Stage 2: Caption Generation
    if stage is None or stage == "2" or stage == "all":
        logger.info("=" * 40)
        logger.info("Stage 2: Caption Generation")
        logger.info("=" * 40)

        stage2_config = config["stage2"]
        captioner = CaptionGenerationStage(stage2_config)

        # Load models (if available)
        try:
            captioner.load_models()
        except Exception as e:
            logger.warning(f"Could not load captioning models: {e}")
            logger.warning("Using mock captioner for testing")

        # Load video frames for captioning
        from action100m.src.stage1_segmentation import VideoLoader

        video_loader = VideoLoader(sample_rate=4)
        frames, fps, _ = video_loader.load_video(video_path)

        # Process with segmentation
        caption_result = captioner.process_segmentation(
            frames, fps, result["segmentation"]
        )

        # Save intermediate result
        output_path = Path(output_dir) / "stage2" / f"{video_name}_captions.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(caption_result, f, indent=2)

        logger.info(f"Saved captions to {output_path}")
        result["captions"] = caption_result

    # Load captions if not just doing Stage 3
    if stage == "3":
        if "captions" not in result:
            cap_path = Path(output_dir) / "stage2" / f"{video_name}_captions.json"
            if cap_path.exists():
                with open(cap_path, "r") as f:
                    result["captions"] = json.load(f)
            else:
                raise FileNotFoundError(f"Caption result not found: {cap_path}")

    # Stage 3: LLM Aggregation
    if stage is None or stage == "3" or stage == "all":
        logger.info("=" * 40)
        logger.info("Stage 3: LLM Aggregation")
        logger.info("=" * 40)

        stage3_config = config["stage3"]

        # Check if API key is provided
        if api_key is None and stage3_config.get("use_api", True):
            api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )

        aggregator = LLMAggregationStage(stage3_config, api_key)

        # Process with tree of captions
        tree_of_captions = result.get("captions", {}).get("tree_of_captions")
        if not tree_of_captions:
            raise ValueError("No tree_of_captions found in caption result")

        # Get video context (title, description, ASR)
        context = {
            "title": Path(video_path).stem,
            "description": "",
            "asr_transcript": "",
        }

        annotations = aggregator.process(tree_of_captions, context)

        # Save final result
        output_path = Path(output_dir) / "stage3" / f"{video_name}_annotations.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        aggregator.save_annotations(annotations, output_path)

        logger.info(f"Saved annotations to {output_path}")
        result["annotations"] = [
            {
                "node_id": ann.node_id,
                "brief_action": ann.brief_action,
                "detailed_action": ann.detailed_action,
                "actor": ann.actor,
                "brief_caption": ann.brief_caption,
                "detailed_caption": ann.detailed_caption,
            }
            for ann in annotations
        ]

    logger.info(f"=" * 60)
    logger.info(f"Completed: {video_path}")
    logger.info(f"=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Action100M Pipeline - Process videos with hierarchical segmentation and captioning"
    )

    parser.add_argument("input", help="Input video file or directory containing videos")

    parser.add_argument(
        "-o", "--output", default="output", help="Output directory (default: output)"
    )

    parser.add_argument(
        "-c",
        "--config",
        default="action100m/configs/config.yaml",
        help="Path to configuration file (default: action100m/configs/config.yaml)",
    )

    parser.add_argument(
        "-s",
        "--stage",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Pipeline stage to run: 1 (segmentation), 2 (captioning), 3 (aggregation), or all (default: all)",
    )

    parser.add_argument(
        "--api-key",
        help="API key for LLM aggregation stage (or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var)",
    )

    parser.add_argument(
        "-l",
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Get video files
    video_files = get_video_files(args.input)
    logger.info(f"Found {len(video_files)} video(s) to process")

    if not video_files:
        logger.error("No video files found")
        sys.exit(1)

    # Process each video
    results = []
    for video_path in video_files:
        try:
            result = process_single_video(
                video_path, config, args.output, args.stage, args.api_key
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}", exc_info=True)
            continue

    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Processed {len(results)}/{len(video_files)} videos successfully")
    logger.info(f"Output saved to: {args.output}")
    logger.info("=" * 60)

    # Save summary
    summary_path = Path(args.output) / "summary.json"
    summary = {
        "total_videos": len(video_files),
        "processed_videos": len(results),
        "stage": args.stage,
        "results": [
            {
                "video_path": r["video_path"],
                "video_name": r.get("video_name"),
                "stages_completed": list(r.keys())[1:],
            }
            for r in results
        ],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
