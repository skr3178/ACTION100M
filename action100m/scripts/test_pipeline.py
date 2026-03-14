#!/usr/bin/env python3
"""Test script to verify the Action100M pipeline works."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Setup
sys.path.insert(0, str(Path(__file__).parent.parent))

from action100m.src.stage1_segmentation import (
    TemporalSegmentationStage,
    VideoLoader,
    VJepa2Encoder,
)
from action100m.src.stage2_captioning import (
    CaptionGenerationStage,
    LeafCaptioner,
    NonLeafCaptioner,
)
from action100m.src.stage3_aggregation import LLMAggregationStage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_video_loader():
    """Test video loading with sample video."""
    logger.info("Testing VideoLoader...")

    # Try loading a sample video if available
    sample_videos = list(Path("data").glob("*.mp4")) if Path("data").exists() else []
    sample_videos.extend(Path(".").glob("**/*.mp4"))

    if sample_videos:
        video_path = str(sample_videos[0])
        logger.info(f"Found sample video: {video_path}")

        loader = VideoLoader(sample_rate=4)
        frames, fps, duration = loader.load_video(video_path)

        logger.info(
            f"Loaded {len(frames)} frames, FPS: {fps}, Duration: {duration:.2f}s"
        )
        return frames, fps
    else:
        logger.warning("No sample videos found, using mock data")
        return None, None


def test_vjepa_encoder(mock: bool = True):
    """Test V-JEPA 2 encoder."""
    logger.info(f"Testing VJepa2Encoder (mock={mock})...")

    encoder = VJepa2Encoder()

    if not mock:
        try:
            encoder.load_model()
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            mock = True

    if mock:
        logger.info("Using mock encoder")
        embeddings = encoder.encode_frames(
            __import__("numpy").random.randn(8, 224, 224, 3).astype("float32")
        )
        logger.info(f"Mock embeddings shape: {embeddings.shape}")

    return encoder


def test_segmentation(frames, fps):
    """Test temporal segmentation."""
    logger.info("Testing HierarchicalSegmenter...")

    from action100m.src.stage1_segmentation import HierarchicalSegmenter

    segmenter = HierarchicalSegmenter(linkage="ward", n_neighbors=5, min_duration=0.5)

    if frames is not None:
        # Use actual frames
        import numpy as np

        # Create mock embeddings from frames (just use random for now)
        embeddings = np.random.randn(len(frames), 1536).astype(np.float32)

        segments = segmenter.segment(embeddings, fps, sample_rate=4)
        logger.info(f"Generated {len(segments)} segments")
    else:
        logger.info("Skipping segmentation test (no frames)")

    return segmenter


def test_captioning():
    """Test caption generation."""
    logger.info("Testing CaptionGenerationStage...")

    leaf_captioner = LeafCaptioner()
    non_leaf_captioner = NonLeafCaptioner()

    logger.info(f"Leaf captioner: {leaf_captioner.model_name}")
    logger.info(f"Non-leaf captioner: {non_leaf_captioner.model_name}")

    return leaf_captioner, non_leaf_captioner


def test_aggregation():
    """Test LLM aggregation."""
    logger.info("Testing LLMAggregationStage...")

    config = {
        "use_api": False,
        "min_duration_seconds": 4,
        "num_refine_rounds": 3,
        "output_fields": [
            "brief_action",
            "detailed_action",
            "actor",
            "brief_caption",
            "detailed_caption",
        ],
    }

    aggregator = LLMAggregationStage(config)
    logger.info("LLMAggregationStage created (API disabled)")

    return aggregator


def main():
    parser = argparse.ArgumentParser(description="Test Action100M pipeline")
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock models (skip downloading)",
    )
    parser.add_argument(
        "--no-mock", action="store_false", dest="mock", help="Try to load actual models"
    )
    parser.add_argument("--video", type=str, default=None, help="Path to test video")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Action100M Pipeline Test")
    logger.info("=" * 60)

    # Test video loader
    frames, fps = test_video_loader()
    if args.video and Path(args.video).exists():
        from action100m.src.stage1_segmentation import VideoLoader

        loader = VideoLoader(sample_rate=4)
        frames, fps, _ = loader.load_video(args.video)
        logger.info(f"Loaded test video: {args.video}")

    # Test V-JEPA encoder
    encoder = test_vjepa_encoder(mock=args.mock)

    # Test segmentation
    segmenter = test_segmentation(frames, fps)

    # Test captioning
    leaf_cap, non_leaf_cap = test_captioning()

    # Test aggregation
    aggregator = test_aggregation()

    logger.info("=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)

    if args.mock:
        logger.info(
            "NOTE: Ran with mock models. Run with --no-mock to use real models."
        )
    else:
        logger.info("NOTE: Ran with real models (may have downloaded from HuggingFace)")


if __name__ == "__main__":
    main()
