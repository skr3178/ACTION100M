import os
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm

try:
    import decord

    decord.bridge.set_bridge("native")
except ImportError:
    decord = None

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """Represents a temporal segment in the video hierarchy."""

    node_id: int
    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float  # seconds
    parent_id: Optional[int] = None
    children_ids: Optional[List[int]] = None
    level: int = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def mid_frame(self) -> int:
        return (self.start_frame + self.end_frame) // 2

    @property
    def is_leaf(self) -> bool:
        return self.children_ids is None or len(self.children_ids) == 0


class VideoLoader:
    """Handles video loading and frame extraction."""

    def __init__(self, sample_rate: int = 4):
        self.sample_rate = sample_rate

    def open_video(self, video_path: str):
        """Open video and return reader + metadata without loading frames.

        Returns:
            Tuple of (VideoReader, fps, total_duration, num_sampled_frames)
        """
        if decord is None:
            raise ImportError("decord is required. Install with: pip install decord")
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        fps = vr.get_avg_fps()
        num_frames = len(vr)
        duration = num_frames / fps
        num_sampled = len(range(0, num_frames, self.sample_rate))
        return vr, fps, duration, num_sampled

    def get_window_frames(self, vr, sampled_start: int, sampled_end: int) -> np.ndarray:
        """Load a specific window of frames on-demand.

        Args:
            vr: decord VideoReader
            sampled_start: start index in sampled-frame space
            sampled_end: end index in sampled-frame space

        Returns:
            Frames array (T, H, W, C)
        """
        real_indices = list(range(
            sampled_start * self.sample_rate,
            sampled_end * self.sample_rate,
            self.sample_rate,
        ))
        real_indices = [min(i, len(vr) - 1) for i in real_indices]
        return vr.get_batch(real_indices).asnumpy()

    def get_frame_at_sampled_idx(self, vr, sampled_idx: int) -> np.ndarray:
        """Load a single frame by sampled index."""
        real_idx = min(sampled_idx * self.sample_rate, len(vr) - 1)
        return vr.get_batch([real_idx]).asnumpy()[0]


class VJepa2Encoder:
    """V-JEPA 2 encoder for video frame embeddings.

    Uses facebook/vjepa2-vitg-fpc64-384 from HuggingFace.
    """

    def __init__(
        self,
        model_name: str = "/media/skr/storage/3DGS/RhodusAI/Action100M/data/models/vjepa2",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        self.crop_size = None

    def load_model(self):
        """Load V-JEPA 2 model and processor from HuggingFace."""
        try:
            from transformers import AutoModel, AutoVideoProcessor

            logger.info(f"Loading V-JEPA 2 model: {self.model_name}")
            self.processor = AutoVideoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name, torch_dtype=self.dtype
            )
            self.model.to(self.device)
            self.model.eval()

            # Get crop size from processor
            if hasattr(self.processor, "crop_size"):
                self.crop_size = self.processor.crop_size.get("height", 384)
            else:
                self.crop_size = 384

            logger.info(
                f"V-JEPA 2 model loaded successfully (crop_size: {self.crop_size})"
            )
        except Exception as e:
            logger.warning(f"Failed to load V-JEPA 2 from HuggingFace: {e}")
            logger.info("Using mock encoder for testing")
            self.model = None

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        """Encode frames to embeddings using V-JEPA 2.

        Args:
            frames: Array of frames (T, H, W, C) in range [0, 255]

        Returns:
            Frame embeddings (T, D)
        """
        if self.model is None:
            # Mock encoding for testing
            return np.random.randn(len(frames), 1408).astype(np.float32)

        num_frames = len(frames)

        # Process through video preprocessor (resize, crop, normalize)
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values_videos"].to(self.device, self.dtype)

        # Forward pass — output: last_hidden_state shape (1, T_tok * N_patches, D)
        # where T_tok = num_frames // tubelet_size, N_patches = spatial patch count
        outputs = self.model(pixel_values)
        hidden = outputs.last_hidden_state  # (1, T_tok * N_patches, D)

        # Reshape to (T_tok, N_patches, D) and average over spatial patches
        tubelet_size = self.model.config.tubelet_size  # 2
        T_tok = num_frames // tubelet_size
        N_patches = hidden.shape[1] // T_tok
        embeddings = hidden.reshape(1, T_tok, N_patches, -1).mean(dim=2)  # (1, T_tok, D)

        # Upsample temporal tokens back to frame resolution by repeating each token
        # tubelet_size times so shape matches the input frame count
        embeddings = embeddings.squeeze(0)  # (T_tok, D)
        embeddings = embeddings.repeat_interleave(tubelet_size, dim=0)  # (num_frames, D)

        return embeddings.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_windows(
        self,
        video_loader,
        vr,
        num_sampled_frames: int,
        window_size: int = 64,
        stride: int = 8,
    ) -> np.ndarray:
        """Encode video with overlapping windows, streaming frames on-demand.

        Args:
            video_loader: VideoLoader instance for on-demand frame fetching
            vr: decord VideoReader
            num_sampled_frames: total number of sampled frames in the video
            window_size: number of sampled frames per window
            stride: sampled-frame stride between windows

        Returns:
            Averaged frame embeddings (num_sampled_frames, D)
        """
        D = self._get_embedding_dim()
        embeddings_accum = np.zeros((num_sampled_frames, D), dtype=np.float32)
        embeddings_count = np.zeros(num_sampled_frames)

        if num_sampled_frames < window_size:
            frames = video_loader.get_window_frames(vr, 0, num_sampled_frames)
            return self.encode_frames(frames)

        checkpoint_path = "/tmp/vjepa2_embeddings_checkpoint.npz"
        start_window = 0

        # Resume from checkpoint if available and shapes match
        if os.path.exists(checkpoint_path):
            ckpt = np.load(checkpoint_path)
            if ckpt["accum"].shape == embeddings_accum.shape:
                embeddings_accum = ckpt["accum"]
                embeddings_count = ckpt["count"]
                start_window = int(ckpt["last_window"]) + 1
                logger.info(f"Resumed from checkpoint at window {start_window}")
            else:
                logger.warning(
                    f"Checkpoint shape {ckpt['accum'].shape} != expected {embeddings_accum.shape}, ignoring stale checkpoint"
                )
                os.remove(checkpoint_path)

        window_starts = list(range(0, num_sampled_frames - window_size + 1, stride))
        for i, start_idx in enumerate(
            tqdm(window_starts[start_window:], desc="Encoding windows",
                 unit="win", initial=start_window, total=len(window_starts))
        ):
            end_idx = start_idx + window_size
            frames = video_loader.get_window_frames(vr, start_idx, end_idx)
            window_embeddings = self.encode_frames(frames)
            embeddings_accum[start_idx:end_idx] += window_embeddings
            embeddings_count[start_idx:end_idx] += 1

            # Checkpoint every 50 windows
            if (start_window + i + 1) % 50 == 0:
                np.savez(checkpoint_path,
                         accum=embeddings_accum, count=embeddings_count,
                         last_window=start_window + i)
                logger.info(f"Checkpoint saved at window {start_window + i + 1}/{len(window_starts)}")

        # Handle tail frames not covered by a full window.
        # Pad to tubelet_size boundary since V-JEPA 2 needs even frame counts.
        last_end = window_starts[-1] + window_size if window_starts else 0
        tail_len = num_sampled_frames - last_end
        if tail_len > 0:
            tubelet = self.model.config.tubelet_size if self.model else 2
            # Round down to nearest tubelet boundary
            usable = (tail_len // tubelet) * tubelet
            if usable > 0:
                frames = video_loader.get_window_frames(vr, last_end, last_end + usable)
                tail_embeddings = self.encode_frames(frames)
                embeddings_accum[last_end:last_end + usable] += tail_embeddings
                embeddings_count[last_end:last_end + usable] += 1

        # Clean up checkpoint on successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("Checkpoint file cleaned up after successful encoding")

        embeddings_count[embeddings_count == 0] = 1
        return embeddings_accum / embeddings_count[:, np.newaxis]

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension from model config."""
        if self.model is not None and hasattr(self.model, "config"):
            return self.model.config.hidden_size
        return 1408  # V-JEPA 2 ViT-g hidden_size


class HierarchicalSegmenter:
    """Performs hierarchical temporal segmentation using agglomerative clustering."""

    def __init__(
        self, linkage: str = "ward", n_neighbors: int = 5, min_duration: float = 0.5
    ):
        self.linkage = linkage
        self.n_neighbors = n_neighbors
        self.min_duration = min_duration
        self.segments: List[Segment] = []

    def segment(
        self, embeddings: np.ndarray, fps: float, sample_rate: int
    ) -> List[Segment]:
        """Perform hierarchical segmentation on frame embeddings.

        Args:
            embeddings: Frame embeddings (T, D)
            fps: Frames per second
            sample_rate: Frame sampling rate

        Returns:
            List of all segments across all hierarchy levels
        """
        from sklearn.cluster import AgglomerativeClustering
        from scipy.sparse import diags

        num_frames = len(embeddings)

        # Temporal chain connectivity: each frame connects only to adjacent frames.
        # This enforces temporally contiguous segments and avoids disconnected
        # component warnings from kneighbors_graph.
        connectivity = diags(
            [1, 1], [-1, 1], shape=(num_frames, num_frames), format="csr"
        )

        # Compute the full dendrogram in one pass.
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            linkage=self.linkage,
            connectivity=connectivity,
            distance_threshold=0,
            compute_full_tree=True,
        )
        clusterer.fit(embeddings)

        # Reconstruct the full tree from children_.
        # children_[i] = [left, right] means internal node (num_frames + i)
        # is the parent of nodes left and right.
        # Nodes 0..num_frames-1 are leaves; num_frames..2*num_frames-2 are internal.
        all_segments: Dict[int, Segment] = {}

        # Create leaf segments (level 0)
        for i in range(num_frames):
            start_frame = i * sample_rate
            end_frame = (i + 1) * sample_rate
            all_segments[i] = Segment(
                node_id=i,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_frame / fps,
                end_time=end_frame / fps,
                children_ids=[],
                level=0,
            )

        # Create internal nodes by walking the merge tree
        for i, (left, right) in enumerate(clusterer.children_):
            node_id = num_frames + i
            left, right = int(left), int(right)
            left_seg = all_segments[left]
            right_seg = all_segments[right]

            level = max(left_seg.level, right_seg.level) + 1
            start_frame = min(left_seg.start_frame, right_seg.start_frame)
            end_frame = max(left_seg.end_frame, right_seg.end_frame)

            all_segments[node_id] = Segment(
                node_id=node_id,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_frame / fps,
                end_time=end_frame / fps,
                parent_id=None,
                children_ids=[left, right],
                level=level,
            )

            # Wire up parent references
            left_seg.parent_id = node_id
            right_seg.parent_id = node_id

        # Keep the full dendrogram — pruning happens as a separate step
        self.segments = list(all_segments.values())

        return self.segments

    def prune(self, min_duration: float = 0.5) -> List[Segment]:
        """Prune the dendrogram so all leaves have duration >= min_duration.

        Walks bottom-up: any node below min_duration is removed, and the
        lowest ancestor >= min_duration becomes the new leaf.
        """
        by_id = {s.node_id: s for s in self.segments}

        # Find new leaf set: for each original leaf, walk up to the first
        # ancestor whose duration >= min_duration.
        new_leaf_ids = set()
        for seg in self.segments:
            if not seg.is_leaf:
                continue
            if seg.duration >= min_duration:
                new_leaf_ids.add(seg.node_id)
                continue
            current = seg
            while current.parent_id is not None and current.parent_id in by_id:
                current = by_id[current.parent_id]
                if current.duration >= min_duration:
                    new_leaf_ids.add(current.node_id)
                    break

        # Collect all ancestors of new leaves (these are the internal nodes to keep)
        keep_ids = set(new_leaf_ids)
        for nid in list(new_leaf_ids):
            current = by_id[nid]
            while current.parent_id is not None and current.parent_id in by_id:
                keep_ids.add(current.parent_id)
                current = by_id[current.parent_id]

        # Rebuild segments with updated children/leaf status
        pruned = []
        for nid in keep_ids:
            seg = by_id[nid]
            kept_children = [c for c in (seg.children_ids or []) if c in keep_ids]
            pruned.append(Segment(
                node_id=seg.node_id,
                start_frame=seg.start_frame,
                end_frame=seg.end_frame,
                start_time=seg.start_time,
                end_time=seg.end_time,
                parent_id=seg.parent_id if seg.parent_id in keep_ids else None,
                children_ids=kept_children,
                level=seg.level,
            ))

        self.segments = pruned
        logger.info(
            f"Pruned to {len(pruned)} nodes: "
            f"{sum(1 for s in pruned if s.is_leaf)} leaves, "
            f"{sum(1 for s in pruned if not s.is_leaf)} internal"
        )
        return self.segments

    def build_tree(self) -> Dict[int, Dict[str, Any]]:
        """Build tree structure from segments.

        Returns:
            Dictionary mapping node_id to segment info
        """
        tree = {}
        for seg in self.segments:
            tree[seg.node_id] = {
                "node_id": seg.node_id,
                "start_frame": seg.start_frame,
                "end_frame": seg.end_frame,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "duration": seg.duration,
                "parent_id": seg.parent_id,
                "children_ids": seg.children_ids,
                "level": seg.level,
                "is_leaf": seg.is_leaf,
            }
        return tree


class TemporalSegmentationStage:
    """Stage 1: Temporal Segmentation using V-JEPA 2."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config["frame_sample_rate"]
        self.window_size = config["window_size"]
        self.window_stride = config["window_stride"]
        self.min_duration = config["min_duration_seconds"]

        self.video_loader = VideoLoader(sample_rate=self.sample_rate)
        self.encoder = VJepa2Encoder(
            model_name=config["model_name"], device=config.get("model_device", "cuda")
        )
        self.segmenter = HierarchicalSegmenter(
            linkage=config["clustering"]["linkage"],
            n_neighbors=config["clustering"]["n_neighbors"],
            min_duration=self.min_duration,
        )

    def load_model(self):
        """Load V-JEPA 2 model."""
        self.encoder.load_model()

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process a single video for temporal segmentation.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with segmentation results
        """
        logger.info(f"Processing video: {video_path}")

        # Open video — metadata only, no frames loaded into RAM
        vr, fps, total_duration, num_sampled = self.video_loader.open_video(video_path)
        logger.info(
            f"Opened video: {num_sampled} sampled frames, FPS: {fps:.2f}, Duration: {total_duration:.2f}s"
        )

        # Stream-encode frames window by window (only 64 frames in RAM at a time)
        embeddings = self.encoder.encode_windows(
            video_loader=self.video_loader,
            vr=vr,
            num_sampled_frames=num_sampled,
            window_size=self.window_size,
            stride=self.window_stride,
        )
        logger.info(f"Generated {len(embeddings)} frame embeddings")

        # Perform hierarchical segmentation
        segments = self.segmenter.segment(embeddings, fps, self.sample_rate)
        logger.info(f"Generated {len(segments)} segments after pruning")

        # Build tree structure
        tree = self.segmenter.build_tree()

        return {
            "video_path": video_path,
            "total_duration": total_duration,
            "fps": fps,
            "num_sampled_frames": num_sampled,
            "num_embeddings": len(embeddings),
            "segments": segments,
            "tree": tree,
            "vr": vr,  # keep reader open for Stage 2 frame access
        }

    def process_batch(
        self, video_paths: List[str], output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple videos.

        Args:
            video_paths: List of video paths
            output_dir: Optional output directory for results

        Returns:
            List of segmentation results
        """
        results = []

        for video_path in tqdm(video_paths, desc="Temporal Segmentation"):
            try:
                result = self.process_video(video_path)
                results.append(result)

                if output_dir:
                    self._save_result(result, output_dir)

            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                continue

        return results

    def _save_result(self, result: Dict[str, Any], output_dir: str):
        """Save segmentation result to file."""
        import json
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        video_name = Path(result["video_path"]).stem
        result_path = output_path / f"{video_name}_segmentation.json"

        # Convert segments to serializable format
        serializable = {
            "video_path": result["video_path"],
            "total_duration": result["total_duration"],
            "fps": result["fps"],
            "num_sampled_frames": result["num_sampled_frames"],
            "tree": result["tree"],
        }

        with open(result_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Saved segmentation to {result_path}")
