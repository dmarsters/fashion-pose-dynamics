"""
Layer 1: Pure taxonomy lookup tools (0 tokens).

All tools in this module return pre-computed catalog data.
No LLM inference required.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from utils.coordinates import apply_modifier
from utils.keyword_index import (
    build_keyword_index,
    get_keywords_for_pose,
    get_keywords_by_category,
)


DATA_DIR = Path(__file__).parent.parent / "data"

# Load catalogs at import time
_pose_catalog: Dict = {}
_head_positions: Dict = {}
_lighting_types: Dict = {}
_visual_vocabulary: Dict = {}
_keyword_index: Dict = {}


def _load_data():
    """Load all JSON data files into module-level caches."""
    global _pose_catalog, _head_positions, _lighting_types
    global _visual_vocabulary, _keyword_index

    with open(DATA_DIR / "pose_catalog.json") as f:
        data = json.load(f)
        _pose_catalog = data.get("poses", {})

    with open(DATA_DIR / "head_positions.json") as f:
        data = json.load(f)
        _head_positions = data.get("head_positions", {})

    with open(DATA_DIR / "lighting_types.json") as f:
        data = json.load(f)
        _lighting_types = data.get("lighting_types", {})

    with open(DATA_DIR / "visual_vocabulary.json") as f:
        _visual_vocabulary = json.load(f)

    _keyword_index = build_keyword_index(_pose_catalog)


def get_pose_catalog() -> Dict:
    """Access the loaded pose catalog."""
    if not _pose_catalog:
        _load_data()
    return _pose_catalog


def get_head_position_catalog() -> Dict:
    """Access the loaded head positions."""
    if not _head_positions:
        _load_data()
    return _head_positions


def get_lighting_catalog() -> Dict:
    """Access the loaded lighting types."""
    if not _lighting_types:
        _load_data()
    return _lighting_types


def get_keyword_index_data() -> Dict:
    """Access the built keyword index."""
    if not _keyword_index:
        _load_data()
    return _keyword_index


def register_taxonomy_tools(mcp):
    """Register all Layer 1 taxonomy tools on the FastMCP server."""

    @mcp.tool()
    def get_pose(pose_name: str, modifier: str = "") -> dict:
        """Look up a specific pose by name and return its complete geometric specification.

        Layer 1: Pure taxonomy lookup (0 tokens).

        Args:
            pose_name: Canonical pose name (e.g. "contrapposto", "broken_doll",
                       "crossover_walk", "look_back")
            modifier: Optional modifier ("subtle", "exaggerated", "editorial")
                      Scales coordinates toward extremes.

        Returns full 7D coordinates, compositional geometry, keywords,
        and body surface map properties.
        """
        catalog = get_pose_catalog()
        pose = catalog.get(pose_name)
        if not pose:
            available = sorted(catalog.keys())
            return {"error": f"Pose '{pose_name}' not found.", "available_poses": available}

        result = dict(pose)
        result["pose_name"] = pose_name

        if modifier:
            result["coordinates"] = apply_modifier(pose["coordinates"], modifier)
            result["modifier_applied"] = modifier

        return result

    @mcp.tool()
    def list_poses(category: str = "", sort_by: str = "") -> dict:
        """List all poses with coordinates, optionally filtered by category.

        Layer 1: Pure taxonomy lookup (0 tokens).

        Args:
            category: Filter by "foundation", "arms", "walking", "editorial",
                      "seated" (empty = all)
            sort_by: Optional sort by any parameter name
                     (e.g. "dynamic_tension", "negative_space_area")
        """
        catalog = get_pose_catalog()
        poses = []
        for name, data in catalog.items():
            if category and data.get("category") != category:
                continue
            poses.append({
                "pose_name": name,
                "category": data.get("category", ""),
                "coordinates": data.get("coordinates", {}),
                "keywords": data.get("keywords", []),
            })

        if sort_by:
            poses.sort(
                key=lambda p: p["coordinates"].get(sort_by, 0.0),
                reverse=True,
            )

        categories = sorted(set(d.get("category", "") for d in catalog.values()))
        return {
            "count": len(poses),
            "available_categories": categories,
            "poses": poses,
        }

    @mcp.tool()
    def get_pose_vocabulary() -> dict:
        """Return complete pose visual vocabulary organized by category.

        Layer 1: Pure taxonomy lookup (0 tokens).

        Categories: body_axis, limb_vectors, negative_space_shapes,
        weight_distribution, gaze_patterns, dynamic_indicators, surface_states.
        """
        if not _visual_vocabulary:
            _load_data()
        return _visual_vocabulary

    @mcp.tool()
    def get_keywords(pose_name: str = "", category: str = "") -> dict:
        """Return keyword associations for poses.

        Layer 1: Pure taxonomy lookup (0 tokens).

        Args:
            pose_name: Specific pose to get keywords for (e.g. "contrapposto").
            category: Filter by keyword category — "mood", "genre", "geometry",
                      "era", "photographer", "movement". Empty = all.

        If both empty, returns the complete keyword index.
        If only category, returns reverse lookup from aesthetic intent to
        candidate poses.
        """
        catalog = get_pose_catalog()
        kw_index = get_keyword_index_data()

        if pose_name:
            keywords = get_keywords_for_pose(pose_name, catalog, kw_index)
            if not keywords:
                return {"error": f"Pose '{pose_name}' not found."}
            return {"pose_name": pose_name, "keywords": keywords}

        if category:
            keywords = get_keywords_by_category(category, kw_index)
            return {"category": category, "count": len(keywords), "keywords": keywords}

        # Full index
        summary = {}
        for kw, entries in kw_index.items():
            summary[kw] = {
                "pose_count": len(entries),
                "poses": [e["pose"] for e in entries],
            }
        return {"total_keywords": len(summary), "index": summary}

    @mcp.tool()
    def get_head_positions() -> dict:
        """List all head position entries with gaze/chin/tilt parameters.

        Layer 1: Pure taxonomy lookup (0 tokens).

        Head positions compose with body poses — they're independent
        parameter sets that overlay.
        """
        positions = get_head_position_catalog()
        return {
            "count": len(positions),
            "positions": {
                name: {
                    "gaze_deviation": data["gaze_deviation"],
                    "chin_angle": data["chin_angle"],
                    "head_tilt": data["head_tilt"],
                    "focal_properties": data["focal_properties"],
                    "keywords": data["keywords"],
                }
                for name, data in positions.items()
            },
        }
