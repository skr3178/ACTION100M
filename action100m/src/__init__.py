"""Action100M - A video action dataset pipeline."""

__version__ = "0.1.0"

from .stage1_segmentation import TemporalSegmentationStage, Segment, VJepa2Encoder
from .stage2_captioning import (
    CaptionGenerationStage,
    LeafCaptioner,
    NonLeafCaptioner,
    TreeOfCaptions,
)
from .stage3_aggregation import LLMAggregationStage, LLMAggregator, StructuredAnnotation

__all__ = [
    "TemporalSegmentationStage",
    "Segment",
    "VJepa2Encoder",
    "CaptionGenerationStage",
    "LeafCaptioner",
    "NonLeafCaptioner",
    "TreeOfCaptions",
    "LLMAggregationStage",
    "LLMAggregator",
    "StructuredAnnotation",
]
