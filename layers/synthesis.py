"""
Layer 3: Structured enhancement data for Claude synthesis (~100-200 tokens).

Runs the full computation chain and returns structured data that
Claude synthesizes into explicit image-generation prompts.
"""

from typing import Dict, Optional

from utils.coordinates import PARAMETER_NAMES, apply_modifier
from utils.keyword_index import search_by_keywords, tokenize_intent
from layers.taxonomy import (
    get_pose_catalog,
    get_head_position_catalog,
    get_lighting_catalog,
    get_keyword_index_data,
)


def _resolve_pose(intent: str, pose_name: str, catalog: Dict) -> tuple:
    """Resolve a pose from explicit name or intent decomposition.

    Returns (pose_name, pose_data) or (None, error_dict).
    """
    if pose_name:
        pose = catalog.get(pose_name)
        if pose:
            return pose_name, pose
        return None, {"error": f"Pose '{pose_name}' not found."}

    if intent:
        kw_index = get_keyword_index_data()
        tokens = tokenize_intent(intent)
        if tokens:
            results = search_by_keywords(tokens, kw_index)
            if results:
                top_name = results[0]["pose"]
                return top_name, catalog[top_name]

    return None, {"error": "No pose resolved. Provide pose_name or intent."}


def _build_geometry_spec(pose_data: Dict, head_data: Dict) -> Dict:
    """Build explicit geometric specification from pose + head data."""
    coords = pose_data.get("coordinates", {})
    geometry = pose_data.get("geometry", {})

    # Convert normalized values to degree descriptions
    axis_degrees = coords.get("primary_axis_angle", 0.0) * 45
    hip_pct = coords.get("hip_displacement", 0.0) * 100
    gaze_degrees = head_data.get("gaze_deviation", 0.33) * 90

    shoulder_angle = geometry.get("shoulder_line_angle", 0)
    hip_angle = geometry.get("hip_line_angle", 0)

    spec = {
        "body_axis": f"{axis_degrees:.0f} degrees from vertical",
        "hip_displacement": f"{hip_pct:.0f}% of shoulder width",
        "shoulder_line": f"{shoulder_angle} degrees from horizontal",
        "hip_line": f"{hip_angle} degrees from horizontal",
        "primary_diagonal": geometry.get("primary_diagonal", ""),
        "counter_diagonal": geometry.get("counter_diagonal", ""),
        "gaze_vector": f"{gaze_degrees:.0f} degrees off camera axis",
        "negative_space_shapes": geometry.get("negative_space_shapes", []),
        "focal_anchor": geometry.get("focal_anchor", ""),
    }

    if "torso_twist" in geometry:
        spec["torso_twist"] = f"{geometry['torso_twist']} degrees of spinal rotation"
    if "stride_vector" in geometry:
        spec["stride_vector"] = geometry["stride_vector"]
    if "implied_force_vector" in geometry:
        spec["implied_force_vector"] = geometry["implied_force_vector"]

    return spec


def _build_lighting_spec(pose_data: Dict, lighting_data: Dict) -> Dict:
    """Build lighting specification from pose + lighting interaction."""
    if not lighting_data:
        return {}

    body_map = pose_data.get("body_surface_map", {})
    neg_shapes = pose_data.get("geometry", {}).get("negative_space_shapes", [])

    return {
        "type": lighting_data.get("source_position", ""),
        "shadow_character": lighting_data.get("shadow_character", ""),
        "highlight_zones": lighting_data.get("highlight_zones", []),
        "negative_space_behavior": lighting_data.get("negative_space_behavior", ""),
        "mood": lighting_data.get("mood", []),
    }


def register_synthesis_tools(mcp):
    """Register all Layer 3 synthesis tools on the FastMCP server."""

    @mcp.tool()
    def enhance_pose_prompt(
        intent: str = "",
        pose_name: str = "",
        head_position: str = "three_quarter_turn",
        lighting_type: str = "",
        frame_aspect: str = "2:3",
        camera_angle: str = "eye_level",
        camera_distance: str = "full_body",
        intensity: float = 0.7,
    ) -> dict:
        """Full pipeline: intent + pose + lighting + frame -> structured
        enhancement data for Claude synthesis.

        Layer 3: Provides structured data (~100-200 tokens).

        Runs the computation chain:
        1. Resolve pose (from name or intent decomposition)
        2. Overlay head position
        3. Compute frame composition
        4. Compute lighting interaction
        5. Export body surface map (for cross-domain consumers)
        6. Compile geometric specification vocabulary

        NOTE: Garment interaction is NOT computed here. When composing
        with garment-dynamics, use aesthetics-dynamics-core to compose
        the two domains. This server exports body_surface_map data that
        garment-dynamics consumes.

        Args:
            intent: Natural language description (used if pose_name empty)
            pose_name: Explicit pose name (takes priority over intent)
            head_position: Head position overlay
            lighting_type: Lighting pattern (empty = no lighting computed)
            frame_aspect: Frame aspect ratio
            camera_angle: Camera angle
            camera_distance: Camera distance / crop
            intensity: Enhancement intensity 0.0-1.0 (default 0.7)
        """
        catalog = get_pose_catalog()
        resolved_name, pose_data = _resolve_pose(intent, pose_name, catalog)

        if resolved_name is None:
            return pose_data  # Error dict

        # Apply intensity as modifier
        coords = dict(pose_data["coordinates"])
        if intensity < 0.4:
            coords = apply_modifier(coords, "subtle")
        elif intensity > 0.8:
            coords = apply_modifier(coords, "exaggerated")

        # Head position
        heads = get_head_position_catalog()
        head = heads.get(head_position, {})

        # Geometry spec
        geometry_spec = _build_geometry_spec(pose_data, head)

        # Lighting
        lighting_spec = {}
        if lighting_type:
            lighting_catalog = get_lighting_catalog()
            lighting = lighting_catalog.get(lighting_type)
            if lighting:
                lighting_spec = _build_lighting_spec(pose_data, lighting)

        # Body surface map
        body_map = pose_data.get("body_surface_map", {})
        head_surface = head.get("surface_effect", {})

        # Frame info
        frame_info = {
            "aspect": frame_aspect,
            "camera_angle": camera_angle,
            "camera_distance": camera_distance,
        }

        # Compose result
        result = {
            "resolved_pose": resolved_name,
            "resolved_from": "name" if pose_name else "intent",
            "coordinates": {k: round(v, 4) for k, v in coords.items()},
            "geometry_spec": geometry_spec,
            "head_position": {
                "name": head_position,
                "chin_angle": head.get("chin_angle", 0),
                "head_tilt": head.get("head_tilt", 0),
                "eye_contact": head.get("focal_properties", {}).get("eye_contact", ""),
            },
            "frame": frame_info,
            "body_surface_map": body_map,
            "keywords": pose_data.get("keywords", []),
        }

        if head_surface:
            result["head_surface_effect"] = head_surface

        if lighting_spec:
            result["lighting"] = lighting_spec

        if intent and not pose_name:
            result["intent_tokens"] = tokenize_intent(intent)

        result["cross_domain_composition"] = {
            "domain_id": "fashion_pose",
            "parameter_names": PARAMETER_NAMES,
            "bounds": [0.0, 1.0],
            "note": "Garment interaction computed via aesthetics-dynamics-core composition with garment_dynamics domain.",
        }

        return result
