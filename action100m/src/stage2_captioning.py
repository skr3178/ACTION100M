import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CaptionResult:
    """Result of caption generation for a segment."""

    node_id: int
    caption: str
    model_used: str
    num_frames: int
    resolution: int


QWEN_MODEL_PATH = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/models/qwen3-vl-4b-awq"


class LeafCaptioner:
    """Caption generator for leaf nodes using Qwen3-VL-4B-AWQ.

    Processes middle frame of each leaf segment with:
    - Model: Qwen3-VL-4B-Instruct-AWQ (local)
    - Resolution: 320x320
    - Prompt: "Describe this image in detail."
    - Max tokens: 1024
    """

    def __init__(
        self,
        model_name: str = QWEN_MODEL_PATH,
        resolution: int = 320,
        max_tokens: int = 1024,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.resolution = resolution
        self.max_tokens = max_tokens
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        """Load Qwen3-VL model (AWQ quantized)."""
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            logger.info(f"Loading Qwen3-VL model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
            logger.info("Qwen3-VL model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Qwen3-VL: {e}")
            logger.info("Using mock captioner for testing")
            self.model = None

    @torch.no_grad()
    def caption_frame(
        self, frame: np.ndarray, prompt: str = "Describe this image in detail."
    ) -> str:
        """Generate caption for a single frame using Qwen3-VL chat format.

        Args:
            frame: Frame array (H, W, C)
            prompt: Prompt for captioning

        Returns:
            Generated caption
        """
        if self.model is None:
            return "Mock caption for frame"

        image = Image.fromarray(frame).resize((self.resolution, self.resolution))

        # Qwen3-VL chat template format
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(self.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs, max_new_tokens=self.max_tokens, do_sample=False
        )
        caption = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return caption

    def caption_leaf_nodes(
        self, frames: np.ndarray, leaf_segments: List[Dict[str, Any]]
    ) -> List[CaptionResult]:
        """Caption all leaf nodes in the segmentation tree.

        Args:
            frames: All sampled frames from video
            leaf_segments: List of leaf segment dictionaries

        Returns:
            List of caption results
        """
        results = []

        for segment in tqdm(leaf_segments, desc="Captioning leaf nodes"):
            # Get mid frame
            mid_frame_idx = (segment["start_frame"] + segment["end_frame"]) // 2
            if mid_frame_idx >= len(frames):
                mid_frame_idx = len(frames) - 1

            frame = frames[mid_frame_idx]

            # Generate caption
            caption = self.caption_frame(frame)

            results.append(
                CaptionResult(
                    node_id=segment["node_id"],
                    caption=caption,
                    model_used=self.model_name,
                    num_frames=1,
                    resolution=self.resolution,
                )
            )

        return results


class NonLeafCaptioner:
    """Caption generator for non-leaf nodes using Qwen3-VL-4B-AWQ.

    Processes video segments with:
    - Model: Qwen3-VL-4B-Instruct-AWQ (local)
    - Resolution: 320x320
    - Frames: 32 evenly spaced frames
    - Prompt: "Describe this video in detail."
    - Max tokens: 1024
    """

    def __init__(
        self,
        model_name: str = QWEN_MODEL_PATH,
        resolution: int = 320,
        num_frames: int = 32,
        max_tokens: int = 1024,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.resolution = resolution
        self.num_frames = num_frames
        self.max_tokens = max_tokens
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        """Load Qwen3-VL model (AWQ quantized)."""
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            logger.info(f"Loading Qwen3-VL model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
            logger.info("Qwen3-VL model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Qwen3-VL: {e}")
            logger.info("Using mock captioner for testing")
            self.model = None

    @torch.no_grad()
    def caption_segment(
        self,
        frames: np.ndarray,
        start_time: float,
        end_time: float,
        prompt: str = "Describe this video in detail.",
    ) -> str:
        """Generate caption for a video segment.

        Args:
            frames: All sampled frames from video
            segment_start: Segment start time in seconds
            segment_end: Segment end time in seconds
            prompt: Prompt for captioning

        Returns:
            Generated caption
        """
        if self.model is None:
            return f"Mock caption for segment {start_time:.2f}s - {end_time:.2f}s"

        # Sample evenly spaced frames from the segment
        total_frames_in_segment = len(frames)
        frame_indices = np.linspace(
            0,
            total_frames_in_segment - 1,
            min(self.num_frames, total_frames_in_segment),
            dtype=int,
        )

        segment_frames = [
            Image.fromarray(frames[i]).resize((self.resolution, self.resolution))
            for i in frame_indices
            if i < len(frames)
        ]

        if not segment_frames:
            return f"Mock caption for segment {start_time:.2f}s - {end_time:.2f}s"

        # Qwen3-VL chat template — pass frames as multiple images
        content = [{"type": "image"} for _ in segment_frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=segment_frames, return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs, max_new_tokens=self.max_tokens, do_sample=False
        )
        caption = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return caption

    def caption_non_leaf_nodes(
        self, frames: np.ndarray, fps: float, non_leaf_segments: List[Dict[str, Any]]
    ) -> List[CaptionResult]:
        """Caption all non-leaf nodes in the segmentation tree.

        Args:
            frames: All sampled frames from video
            fps: Frames per second
            non_leaf_segments: List of non-leaf segment dictionaries

        Returns:
            List of caption results
        """
        results = []

        for segment in tqdm(non_leaf_segments, desc="Captioning non-leaf nodes"):
            caption = self.caption_segment(
                frames, segment["start_time"], segment["end_time"]
            )

            results.append(
                CaptionResult(
                    node_id=segment["node_id"],
                    caption=caption,
                    model_used=self.model_name,
                    num_frames=self.num_frames,
                    resolution=self.resolution,
                )
            )

        return results


class TreeOfCaptions:
    """Organizes captions into hierarchical tree structure."""

    def __init__(self):
        self.captions: Dict[int, CaptionResult] = {}
        self.tree: Dict[int, Dict[str, Any]] = {}

    def add_caption(self, caption_result: CaptionResult):
        """Add a caption to the tree."""
        self.captions[caption_result.node_id] = caption_result

    def build_tree(self, segments: Dict[int, Dict[str, Any]]):
        """Build tree structure from segments and captions."""
        self.tree = {}

        for node_id, segment in segments.items():
            caption = self.captions.get(node_id)

            self.tree[node_id] = {
                "node_id": node_id,
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "duration": segment["duration"],
                "parent_id": segment.get("parent_id"),
                "children_ids": segment.get("children_ids", []),
                "level": segment.get("level", 0),
                "is_leaf": segment.get("is_leaf", True),
                "caption": caption.caption if caption else None,
                "model_used": caption.model_used if caption else None,
            }

    def get_node_caption(self, node_id: int) -> Optional[str]:
        """Get caption for a specific node."""
        if node_id in self.captions:
            return self.captions[node_id].caption
        return None

    def get_children_captions(self, node_id: int) -> List[str]:
        """Get captions for all children of a node (depth-first order)."""
        if node_id not in self.tree:
            return []

        children_ids = self.tree[node_id].get("children_ids", [])
        captions = []

        for child_id in children_ids:
            # Get child's caption
            if child_id in self.captions:
                captions.append(self.captions[child_id].caption)
            # Recursively get children's captions
            captions.extend(self.get_children_captions(child_id))

        return captions

    def get_root_caption(self, max_level: int = 2) -> Optional[str]:
        """Get caption from root node at specified max level."""
        # Find root nodes (nodes with no parent)
        roots = [
            node_id
            for node_id, info in self.tree.items()
            if info.get("parent_id") is None
        ]

        if roots:
            root_id = roots[0]
            if self.tree[root_id]["level"] <= max_level:
                return (
                    self.captions.get(root_id).caption
                    if root_id in self.captions
                    else None
                )
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to serializable dictionary."""
        return {
            "captions": {
                node_id: {
                    "caption": result.caption,
                    "model_used": result.model_used,
                    "num_frames": result.num_frames,
                    "resolution": result.resolution,
                }
                for node_id, result in self.captions.items()
            },
            "tree": self.tree,
        }


class CaptionGenerationStage:
    """Stage 2: Caption Generation using VLM models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Leaf node captioner (LLaMA-3.2-Vision-11B)
        leaf_config = config["leaf"]
        self.leaf_captioner = LeafCaptioner(
            model_name=leaf_config["model_name"],
            resolution=leaf_config["resolution"],
            max_tokens=leaf_config["max_tokens"],
            device=leaf_config["device"],
        )

        # Non-leaf node captioner (PerceptionLM-3B)
        non_leaf_config = config["non_leaf"]
        self.non_leaf_captioner = NonLeafCaptioner(
            model_name=non_leaf_config["model_name"],
            resolution=non_leaf_config["resolution"],
            num_frames=non_leaf_config["num_frames"],
            max_tokens=non_leaf_config["max_tokens"],
            device=non_leaf_config["device"],
        )

        self.tree_of_captions = TreeOfCaptions()

    def load_models(self):
        """Load both captioning models."""
        self.leaf_captioner.load_model()
        self.non_leaf_captioner.load_model()

    def process_segmentation(
        self, frames: np.ndarray, fps: float, segmentation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process segmentation result to generate captions.

        Args:
            frames: Video frames
            fps: Frames per second
            segmentation_result: Output from Stage 1

        Returns:
            Updated result with captions
        """
        tree = segmentation_result["tree"]

        # Separate leaf and non-leaf nodes
        leaf_nodes = []
        non_leaf_nodes = []

        for node_id, info in tree.items():
            if info["is_leaf"]:
                leaf_nodes.append(info)
            else:
                non_leaf_nodes.append(info)

        # Caption leaf nodes
        logger.info(f"Captioning {len(leaf_nodes)} leaf nodes")
        leaf_captions = self.leaf_captioner.caption_leaf_nodes(frames, leaf_nodes)

        # Caption non-leaf nodes
        logger.info(f"Captioning {len(non_leaf_nodes)} non-leaf nodes")
        non_leaf_captions = self.non_leaf_captioner.caption_non_leaf_nodes(
            frames, fps, non_leaf_nodes
        )

        # Build tree of captions
        for caption_result in leaf_captions + non_leaf_captions:
            self.tree_of_captions.add_caption(caption_result)

        self.tree_of_captions.build_tree(tree)

        return {
            **segmentation_result,
            "tree_of_captions": self.tree_of_captions.to_dict(),
        }

    def process_video(
        self, video_path: str, segmentation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single video with segmentation results.

        Args:
            video_path: Path to video file
            segmentation_result: Output from Stage 1

        Returns:
            Complete result with captions
        """
        # Load video frames (need full frames for captioning)
        from .stage1_segmentation import VideoLoader

        sample_rate = 4  # Same as Stage 1
        video_loader = VideoLoader(sample_rate=sample_rate)
        frames, fps, _ = video_loader.load_video(video_path)

        return self.process_segmentation(frames, fps, segmentation_result)

    def process_batch(
        self, caption_results: List[Dict[str, Any]], frames: np.ndarray, fps: float
    ) -> List[Dict[str, Any]]:
        """Process batch of segmentation results.

        Args:
            caption_results: List of segmentation results from Stage 1
            frames: Video frames
            fps: Frames per second

        Returns:
            List of results with captions
        """
        results = []

        for result in tqdm(caption_results, desc="Generating captions"):
            try:
                updated = self.process_segmentation(frames, fps, result)
                results.append(updated)
            except Exception as e:
                logger.error(
                    f"Error captioning {result.get('video_path', 'unknown')}: {e}"
                )
                continue

        return results

    def _save_result(self, result: Dict[str, Any], output_dir: str):
        """Save caption result to file."""
        import json
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        video_name = Path(result.get("video_path", "unknown")).stem
        result_path = output_path / f"{video_name}_captions.json"

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved captions to {result_path}")
