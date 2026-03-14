import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

import torch

logger = logging.getLogger(__name__)


@dataclass
class StructuredAnnotation:
    """Structured annotation extracted from LLM."""

    node_id: int
    brief_action: str
    detailed_action: str
    actor: str
    brief_caption: str
    detailed_caption: str
    confidence: float = 1.0


class LLMAggregator:
    """Stage 3: LLM Aggregation using API or local model.

    Extracts structured annotations from Tree-of-Captions:
    - brief_action: Short action description
    - detailed_action: Detailed action description
    - actor: Who performed the action
    - brief_caption: Brief video caption
    - detailed_caption: Detailed video caption

    Performs 3 rounds of Self-Refine for quality improvement.
    """

    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key
        self.use_api = config.get("use_api", True)
        self.min_duration = config.get("min_duration_seconds", 4)
        self.num_refine_rounds = config.get("num_refine_rounds", 3)
        self.output_fields = config.get(
            "output_fields",
            [
                "brief_action",
                "detailed_action",
                "actor",
                "brief_caption",
                "detailed_caption",
            ],
        )

        self.client = None

    def _init_api_client(self):
        """Initialize API client."""
        if not self.use_api:
            return self._init_local_model()

        provider = self.config.get("api", {}).get("provider", "anthropic")

        if provider == "anthropic":
            try:
                import anthropic

                self.client = anthropic.Anthropic(
                    api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY")
                )
                logger.info("Initialized Anthropic API client")
            except ImportError:
                logger.warning("anthropic package not installed")
                self.client = None

        elif provider == "openai":
            try:
                import openai

                self.client = openai.OpenAI(
                    api_key=self.api_key or os.environ.get("OPENAI_API_KEY")
                )
                logger.info("Initialized OpenAI API client")
            except ImportError:
                logger.warning("openai package not installed")
                self.client = None
        else:
            logger.warning(f"Unknown API provider: {provider}")

    def _init_local_model(self):
        """Initialize local model (e.g., Llama-3.1-70B)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = self.config.get("local", {}).get(
                "model_name", "meta-llama/Llama-3.1-70B-Instruct-Q4_K_M"
            )
            device = self.config.get("local", {}).get("device", "cuda")

            logger.info(f"Loading local model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                load_in_4bit=True,
            )
            self.model.eval()
            logger.info("Local model loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load local model: {e}")
            self.model = None

    def _get_dfs_captions(
        self, tree_of_captions: Dict[str, Any], node_id: int, max_depth: int = 5
    ) -> List[str]:
        """Collect children's captions in depth-first order (recursive).

        Args:
            tree_of_captions: Tree structure with captions
            node_id: Current node ID
            max_depth: Maximum recursion depth to prevent excessive context

        Returns:
            List of caption strings in DFS order
        """
        captions = []
        node_info = tree_of_captions["tree"].get(node_id)
        if not node_info or max_depth <= 0:
            return captions

        for child_id in node_info.get("children_ids", []):
            child_id_str = str(child_id)
            # Add this child's caption
            child_caption = (
                tree_of_captions["captions"].get(child_id_str, {}).get("caption", "")
            )
            if child_caption:
                captions.append(child_caption)
            # Recurse into child's children
            captions.extend(
                self._get_dfs_captions(tree_of_captions, child_id_str, max_depth - 1)
            )

        return captions

    def _build_prompt(
        self,
        tree_of_captions: Dict[str, Any],
        node_id: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build prompt for LLM to extract structured annotation."""

        node_info = tree_of_captions["tree"].get(node_id)
        if not node_info:
            return ""

        # Get current node's caption
        current_caption = (
            tree_of_captions["captions"].get(node_id, {}).get("caption", "")
        )

        # Get children's captions in depth-first order (recursive)
        children_captions = self._get_dfs_captions(tree_of_captions, node_id)

        # Get root caption for context (find actual root — node with no parent)
        root_caption = ""
        for nid, info in tree_of_captions["tree"].items():
            if info.get("parent_id") is None:
                root_caption = (
                    tree_of_captions["captions"].get(nid, {}).get("caption", "")
                )
                break

        # Build context info
        context_info = ""
        if context:
            video_title = context.get("title", "")
            video_description = context.get("description", "")
            asr_transcript = context.get("asr_transcript", "")

            if video_title:
                context_info += f"Video title: {video_title}\n"
            if video_description:
                context_info += f"Video description: {video_description}\n"
            if asr_transcript:
                context_info += (
                    f"ASR transcript (truncated): {asr_transcript[:1000]}...\n"
                )

        prompt = f"""You are tasked with extracting structured information from video captions. The video has been temporally segmented into a hierarchy of segments, each with captions.

{context_info}

Current segment spans from {node_info["start_time"]:.2f} to {node_info["end_time"]:.2f} seconds (duration: {node_info["duration"]:.2f}s).

Current segment caption:
{current_caption}

Children segments captions (in order):
{chr(10).join(f"- {cap}" for cap in children_captions) if children_captions else "N/A"}

Root context:
{root_caption if root_caption else "N/A"}

Based on the above information, extract the following structured fields for the current segment:

1. brief_action: A very short action description (1-5 words, e.g., "add flour", "stir batter")
2. detailed_action: Detailed action description (1-2 sentences)
3. actor: Who performed the action. Use the person's name if mentioned in the title, description, or ASR transcript. Otherwise describe them (e.g., "person", "chef", "woman in kitchen")
4. brief_caption: Brief video caption describing what happens (1 sentence)
5. detailed_caption: Detailed video caption with full context (2-3 sentences)

Output in JSON format:
```json
{{
  "brief_action": "...",
  "detailed_action": "...",
  "actor": "...",
  "brief_caption": "...",
  "detailed_caption": "..."
}}
```

Only output the JSON, nothing else."""

        return prompt

    def _call_api(self, prompt: str, model: Optional[str] = None) -> str:
        """Call API to get LLM response."""

        provider = self.config.get("api", {}).get("provider", "anthropic")
        model = model or self.config.get("api", {}).get(
            "model", "claude-sonnet-4-20250514"
        )

        max_retries = self.config.get("api", {}).get("max_retries", 3)

        for attempt in range(max_retries):
            try:
                if provider == "anthropic" and self.client:
                    response = self.client.messages.create(
                        model=model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.content[0].text

                elif provider == "openai" and self.client:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024,
                    )
                    return response.choices[0].message.content

            except Exception as e:
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    raise

        return "{}"

    def _call_local_model(self, prompt: str) -> str:
        """Call local model to get LLM response."""
        if not hasattr(self, "model") or self.model is None:
            return "{}"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=False, temperature=0.0
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the response part after the prompt
        response = response[len(prompt) :].strip()

        return response

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        import re

        # Try to find JSON in the response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try to parse entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Return empty dict if parsing fails
        logger.warning(f"Failed to parse JSON from response: {response[:200]}")
        return {}

    def _refine_annotation(
        self, annotation: Dict[str, Any], tree_of_captions: Dict[str, Any], node_id: int
    ) -> Dict[str, Any]:
        """Refine annotation through Self-Refine rounds."""

        prompt = f"""Review and improve the following annotation for consistency and accuracy:

Current annotation:
{json.dumps(annotation, indent=2)}

Segment info:
{json.dumps(tree_of_captions["tree"].get(node_id, {}), indent=2)}

Check for:
1. Consistency between fields
2. Factual accuracy based on the captions
3. Completeness of information
4. Correct grammar and spelling

Output improved JSON:
```json
{{
  "brief_action": "...",
  "detailed_action": "...",
  "actor": "...",
  "brief_caption": "...",
  "detailed_caption": "..."
}}
```"""

        if self.use_api and self.client:
            response = self._call_api(prompt)
        else:
            response = self._call_local_model(prompt)

        refined = self._parse_json_response(response)

        # Merge with original (keep fields that weren't refined)
        for field in self.output_fields:
            if field not in refined and field in annotation:
                refined[field] = annotation[field]

        return refined

    def process_node(
        self,
        node_id: int,
        tree_of_captions: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[StructuredAnnotation]:
        """Process a single node to extract structured annotation."""

        node_info = tree_of_captions["tree"].get(node_id)
        if not node_info:
            return None

        # Skip nodes shorter than min_duration
        if node_info["duration"] < self.min_duration:
            logger.debug(
                f"Skipping node {node_id} (duration {node_info['duration']:.2f}s < {self.min_duration}s)"
            )
            return None

        # Build initial prompt
        prompt = self._build_prompt(tree_of_captions, node_id, context)

        # Get initial response
        if self.use_api and self.client:
            response = self._call_api(prompt)
        else:
            response = self._call_local_model(prompt)

        # Parse response
        annotation = self._parse_json_response(response)

        # Self-Refine rounds
        for round_idx in range(self.num_refine_rounds - 1):
            annotation = self._refine_annotation(annotation, tree_of_captions, node_id)

        # Create structured annotation
        return StructuredAnnotation(
            node_id=node_id,
            brief_action=annotation.get("brief_action", ""),
            detailed_action=annotation.get("detailed_action", ""),
            actor=annotation.get("actor", ""),
            brief_caption=annotation.get("brief_caption", ""),
            detailed_caption=annotation.get("detailed_caption", ""),
        )

    def process_tree(
        self, tree_of_captions: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> List[StructuredAnnotation]:
        """Process all nodes in the tree to extract structured annotations."""

        # Initialize API client if needed
        if self.client is None:
            self._init_api_client()

        annotations = []

        # Get all node IDs (sorted by level for consistent processing)
        nodes_by_level = {}
        for node_id, info in tree_of_captions["tree"].items():
            level = info.get("level", 0)
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node_id)

        # Process nodes level by level
        for level in sorted(nodes_by_level.keys()):
            node_ids = nodes_by_level[level]
            logger.info(f"Processing {len(node_ids)} nodes at level {level}")

            for node_id in tqdm(node_ids, desc=f"Level {level}"):
                annotation = self.process_node(node_id, tree_of_captions, context)
                if annotation:
                    annotations.append(annotation)

        return annotations

    def save_annotations(
        self, annotations: List[StructuredAnnotation], output_path: str
    ):
        """Save annotations to file."""

        data = {
            "num_annotations": len(annotations),
            "annotations": [
                {
                    "node_id": ann.node_id,
                    "brief_action": ann.brief_action,
                    "detailed_action": ann.detailed_action,
                    "actor": ann.actor,
                    "brief_caption": ann.brief_caption,
                    "detailed_caption": ann.detailed_caption,
                }
                for ann in annotations
            ],
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(annotations)} annotations to {output_path}")


class LLMAggregationStage:
    """Stage 3: LLM Aggregation for structured annotations."""

    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key
        self.aggregator = LLMAggregator(config, api_key)

    def process(
        self, tree_of_captions: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> List[StructuredAnnotation]:
        """Process Tree-of-Captions to generate structured annotations."""

        annotations = self.aggregator.process_tree(tree_of_captions, context)

        return annotations

    def process_video(
        self, caption_result: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process video caption result to generate annotations."""

        tree_of_captions = caption_result.get("tree_of_captions")
        if not tree_of_captions:
            raise ValueError("No tree_of_captions in caption result")

        annotations = self.process(tree_of_captions, context)

        # Add annotations to result
        result = {
            **caption_result,
            "annotations": [
                {
                    "node_id": ann.node_id,
                    "brief_action": ann.brief_action,
                    "detailed_action": ann.detailed_action,
                    "actor": ann.actor,
                    "brief_caption": ann.brief_caption,
                    "detailed_caption": ann.detailed_caption,
                }
                for ann in annotations
            ],
        }

        return result
