"""
Layer 2: Deterministic computation tools (0 tokens).

Pose-composition, pose-lighting, body surface map export,
distance/trajectory/intent decomposition.
"""

from typing import Dict, List, Optional

from utils.coordinates import (
    PARAMETER_NAMES,
    euclidean_distance,
    interpolate,
    find_nearest,
    apply_modifier,
)
from utils.keyword_index import (
    search_by_keywords,
    tokenize_intent,
)
from layers.taxonomy import (
    get_pose_catalog,
    get_head_position_catalog,
    get_lighting_catalog,
    get_keyword_index_data,
)


# --- Frame geometry helpers ---

FRAME_ASPECTS = {
    "1:1": (1.0, 1.0),
    "2:3": (2.0, 3.0),
    "3:4": (3.0, 4.0),
    "4:5": (4.0, 5.0),
    "9:16": (9.0, 16.0),
    "16:9": (16.0, 9.0),
}

CAMERA_ANGLE_MODIFIERS = {
    "eye_level": {"vertical_shift": 0, "foreshortening": "none", "power_read": "neutral"},
    "low_angle": {"vertical_shift": -15, "foreshortening": "legs-elongated", "power_read": "dominant"},
    "high_angle": {"vertical_shift": 15, "foreshortening": "head-emphasized", "power_read": "diminished"},
    "dutch": {"vertical_shift": 0, "foreshortening": "diagonal-frame", "power_read": "unsettled"},
}

CAMERA_DISTANCE_CROPS = {
    "full_body": {"crop_top": 0.0, "crop_bottom": 1.0, "visible": "head-to-feet"},
    "three_quarter": {"crop_top": 0.0, "crop_bottom": 0.75, "visible": "head-to-mid-calf"},
    "waist_up": {"crop_top": 0.0, "crop_bottom": 0.45, "visible": "head-to-waist"},
    "close_up": {"crop_top": 0.0, "crop_bottom": 0.2, "visible": "head-and-shoulders"},
}


def _compute_thirds_occupancy(coords: Dict[str, float], geometry: Dict) -> Dict:
    """Determine which rule-of-thirds intersections the pose occupies."""
    hip_disp = coords.get("hip_displacement", 0.0)
    axis_angle = coords.get("primary_axis_angle", 0.0)

    # Map hip displacement to horizontal position
    # 0 = centered, positive = right of center
    h_position = "center"
    if hip_disp > 0.3:
        h_position = "right-third"
    elif hip_disp > 0.15:
        h_position = "right-of-center"

    focal = geometry.get("focal_anchor", "center-face")
    return {
        "head_position": "upper-third",
        "hip_apex": h_position,
        "focal_anchor": focal,
        "primary_diagonal_crosses": "upper-left-to-lower-right" if axis_angle > 0.2 else "near-vertical",
    }


def _compute_visual_weight(coords: Dict[str, float]) -> Dict:
    """Compute visual weight distribution in frame."""
    hip_disp = coords.get("hip_displacement", 0.0)
    weight_asym = coords.get("weight_asymmetry", 0.0)
    neg_space = coords.get("negative_space_area", 0.0)

    left_right = 0.5 + (hip_disp * 0.3)  # Shifted toward displaced side
    top_bottom = 0.4  # Poses generally weight upper body

    return {
        "left_right_balance": round(left_right, 2),
        "top_bottom_balance": round(top_bottom, 2),
        "negative_space_lightens": neg_space > 0.5,
        "weight_concentration": "single-side" if weight_asym > 0.6 else "distributed",
    }


def _compute_lighting_on_pose(pose_data: Dict, lighting_data: Dict, head_data: Dict) -> Dict:
    """Compute shadow/highlight map from pose geometry + lighting."""
    coords = pose_data.get("coordinates", {})
    geometry = pose_data.get("geometry", {})
    body_map = pose_data.get("body_surface_map", {})

    light_angle = lighting_data.get("angle_from_camera", 0)
    shadow_char = lighting_data.get("shadow_character", "")
    highlight_zones = lighting_data.get("highlight_zones", [])
    neg_behavior = lighting_data.get("negative_space_behavior", "")

    # Determine which negative spaces get shadow vs light
    neg_shapes = geometry.get("negative_space_shapes", [])
    surface_facing = body_map.get("surface_facing", [])

    shadow_map = {
        "facial_pattern": shadow_char,
        "body_shadow_side": "camera-right" if light_angle > 0 else "camera-left" if light_angle < 0 else "below",
        "negative_space_fill": neg_behavior,
        "negative_spaces_affected": neg_shapes,
    }

    highlight_map = {
        "zones": highlight_zones,
        "surfaces_catching_light": [s for s in surface_facing if "faces-camera" in s or "faces-up" in s],
    }

    stretched = body_map.get("stretched_surfaces", [])
    compressed = body_map.get("compressed_surfaces", [])

    return {
        "shadow_map": shadow_map,
        "highlight_map": highlight_map,
        "stretched_surfaces_highlight": "stretched surfaces catch more light due to smooth planes",
        "compressed_surfaces_shadow": "compression folds create shadow traps",
        "lighting_mood": lighting_data.get("mood", []),
    }


def register_computation_tools(mcp):
    """Register all Layer 2 computation tools on the FastMCP server."""

    @mcp.tool()
    def compute_pose_composition(
        pose_name: str,
        head_position: str = "three_quarter_turn",
        frame_aspect: str = "2:3",
        camera_angle: str = "eye_level",
        camera_distance: str = "full_body",
    ) -> dict:
        """Compute resulting compositional geometry from pose + frame + camera.

        Layer 2: Deterministic computation (0 tokens).

        Takes pose parameters and computes resulting diagonal vectors,
        negative space positions, rule-of-thirds occupancy, sight line
        trajectory, silhouette bounding box, and visual weight distribution.

        Args:
            pose_name: Any canonical pose name
            head_position: Head position overlay
            frame_aspect: "1:1", "2:3", "3:4", "4:5", "9:16", "16:9"
            camera_angle: "eye_level", "low_angle", "high_angle", "dutch"
            camera_distance: "full_body", "three_quarter", "waist_up", "close_up"
        """
        catalog = get_pose_catalog()
        pose = catalog.get(pose_name)
        if not pose:
            return {"error": f"Pose '{pose_name}' not found."}

        heads = get_head_position_catalog()
        head = heads.get(head_position, {})

        coords = pose["coordinates"]
        geometry = pose.get("geometry", {})

        # Frame
        aspect = FRAME_ASPECTS.get(frame_aspect, (2.0, 3.0))
        cam_mod = CAMERA_ANGLE_MODIFIERS.get(camera_angle, CAMERA_ANGLE_MODIFIERS["eye_level"])
        crop = CAMERA_DISTANCE_CROPS.get(camera_distance, CAMERA_DISTANCE_CROPS["full_body"])

        # Compute gaze vector (pose deviation + head deviation)
        base_gaze = coords.get("gaze_deviation", 0.0)
        head_gaze = head.get("gaze_deviation", 0.33)
        # Combined gaze is head override when available
        effective_gaze = head_gaze

        # Sight line
        gaze_degrees = effective_gaze * 90
        gaze_desc = f"{gaze_degrees:.0f} degrees off camera axis"
        if effective_gaze < 0.1:
            gaze_terminates = "direct-at-camera"
        elif effective_gaze < 0.4:
            gaze_terminates = "intersects-upper-third-line"
        elif effective_gaze < 0.7:
            gaze_terminates = "exits-frame-at-edge"
        else:
            gaze_terminates = "parallel-to-frame-edge-profile"

        thirds = _compute_thirds_occupancy(coords, geometry)
        weight = _compute_visual_weight(coords)

        # Diagonal summary
        diagonals = {
            "primary": geometry.get("primary_diagonal", ""),
            "counter": geometry.get("counter_diagonal", ""),
            "shoulder_line": f"{geometry.get('shoulder_line_angle', 0)} degrees from horizontal",
            "hip_line": f"{geometry.get('hip_line_angle', 0)} degrees from horizontal",
        }

        return {
            "pose_name": pose_name,
            "head_position": head_position,
            "frame": {
                "aspect": frame_aspect,
                "orientation": "portrait" if aspect[0] < aspect[1] else "landscape" if aspect[0] > aspect[1] else "square",
            },
            "camera": {
                "angle": camera_angle,
                "distance": camera_distance,
                "modifier": cam_mod,
                "crop": crop,
            },
            "diagonals": diagonals,
            "negative_space": {
                "shapes": geometry.get("negative_space_shapes", []),
                "total_area_normalized": coords.get("negative_space_area", 0.0),
            },
            "thirds_occupancy": thirds,
            "sight_line": {
                "gaze_angle": gaze_desc,
                "terminates_at": gaze_terminates,
                "head_chin_angle": head.get("chin_angle", 0),
                "head_tilt": head.get("head_tilt", 0),
            },
            "visual_weight": weight,
        }

    @mcp.tool()
    def compute_pose_lighting_interaction(
        pose_name: str,
        lighting_type: str,
        light_angle: float = 45.0,
        head_position: str = "three_quarter_turn",
    ) -> dict:
        """Compute shadow geometry and highlight placement from pose + lighting.

        Layer 2: Deterministic computation (0 tokens).

        Args:
            pose_name: Any canonical pose name
            lighting_type: "butterfly", "rembrandt", "split", "loop",
                           "broad", "short", "rim", "clamshell",
                           "side_window", "overhead_harsh"
            light_angle: Degrees from camera axis (horizontal plane)
            head_position: Head position overlay
        """
        catalog = get_pose_catalog()
        pose = catalog.get(pose_name)
        if not pose:
            return {"error": f"Pose '{pose_name}' not found."}

        lighting_catalog = get_lighting_catalog()
        lighting = lighting_catalog.get(lighting_type)
        if not lighting:
            return {"error": f"Lighting '{lighting_type}' not found.", "available": sorted(lighting_catalog.keys())}

        heads = get_head_position_catalog()
        head = heads.get(head_position, {})

        result = _compute_lighting_on_pose(pose, lighting, head)
        result["pose_name"] = pose_name
        result["lighting_type"] = lighting_type
        result["light_angle_override"] = light_angle
        result["head_position"] = head_position

        return result

    @mcp.tool()
    def get_body_surface_map(
        pose_name: str,
        head_position: str = "three_quarter_turn",
    ) -> dict:
        """Export body surface geometry for cross-domain composition.

        Layer 2: Deterministic computation (0 tokens).

        This is the primary interface for garment-dynamics and other
        domains that need body spatial state. Returns stretched/compressed
        surfaces, angled planes, joint bends, gravity anchors, surface
        facing directions, and motion vectors.

        Args:
            pose_name: Any canonical pose name
            head_position: Head position overlay (affects neck/shoulder surfaces)
        """
        catalog = get_pose_catalog()
        pose = catalog.get(pose_name)
        if not pose:
            return {"error": f"Pose '{pose_name}' not found."}

        body_map = dict(pose.get("body_surface_map", {}))

        # Overlay head position effects on neck/shoulder surfaces
        heads = get_head_position_catalog()
        head = heads.get(head_position, {})
        surface_effect = head.get("surface_effect", {})

        if surface_effect:
            body_map["head_overlay"] = {
                "head_position": head_position,
                "neck_stretch": surface_effect.get("neck_stretch", "neutral"),
                "neck_compression": surface_effect.get("neck_compression", "neutral"),
                "face_plane_angle": surface_effect.get("face_plane_angle", 0),
            }

        body_map["pose_name"] = pose_name
        body_map["coordinates"] = pose.get("coordinates", {})
        body_map["cross_domain_hint"] = ["garment_dynamics", "stage_lighting"]

        return body_map

    @mcp.tool()
    def compute_pose_distance(pose_a: str, pose_b: str) -> dict:
        """Euclidean distance between two poses in 7D parameter space.

        Layer 2: Deterministic computation (0 tokens).
        """
        catalog = get_pose_catalog()
        pa = catalog.get(pose_a)
        pb = catalog.get(pose_b)

        if not pa:
            return {"error": f"Pose '{pose_a}' not found."}
        if not pb:
            return {"error": f"Pose '{pose_b}' not found."}

        dist = euclidean_distance(pa["coordinates"], pb["coordinates"])

        # Per-parameter breakdown
        breakdown = {}
        for p in PARAMETER_NAMES:
            diff = abs(pa["coordinates"].get(p, 0.0) - pb["coordinates"].get(p, 0.0))
            breakdown[p] = round(diff, 4)

        return {
            "pose_a": pose_a,
            "pose_b": pose_b,
            "euclidean_distance": round(dist, 4),
            "parameter_differences": breakdown,
            "most_different": max(breakdown, key=breakdown.get),
            "most_similar": min(breakdown, key=breakdown.get),
        }

    @mcp.tool()
    def find_nearby_poses(pose_name: str, max_results: int = 5) -> dict:
        """Find poses nearest to the given pose in 7D parameter space.

        Layer 2: Deterministic distance computation (0 tokens).
        """
        catalog = get_pose_catalog()
        pose = catalog.get(pose_name)
        if not pose:
            return {"error": f"Pose '{pose_name}' not found."}

        results = find_nearest(
            pose["coordinates"], catalog, max_results=max_results, exclude=pose_name
        )

        return {
            "reference_pose": pose_name,
            "nearest": [
                {"pose": name, "distance": round(dist, 4)}
                for name, dist in results
            ],
        }

    @mcp.tool()
    def compute_pose_trajectory(
        pose_a: str,
        pose_b: str,
        steps: int = 10,
    ) -> dict:
        """Smooth interpolation between two poses through 7D space.

        Layer 2: Deterministic interpolation (0 tokens).

        Useful for: animation keyframes, editorial series showing
        progression, or finding intermediate poses.
        """
        catalog = get_pose_catalog()
        pa = catalog.get(pose_a)
        pb = catalog.get(pose_b)

        if not pa:
            return {"error": f"Pose '{pose_a}' not found."}
        if not pb:
            return {"error": f"Pose '{pose_b}' not found."}

        trajectory = interpolate(pa["coordinates"], pb["coordinates"], steps=steps)

        # For each step, find the nearest named pose
        waypoints = []
        for i, point in enumerate(trajectory):
            nearest = find_nearest(point, catalog, max_results=1)
            nearest_name = nearest[0][0] if nearest else "unknown"
            nearest_dist = nearest[0][1] if nearest else 0.0
            waypoints.append({
                "step": i,
                "t": round(i / max(steps - 1, 1), 3),
                "coordinates": {k: round(v, 4) for k, v in point.items()},
                "nearest_named_pose": nearest_name,
                "distance_to_nearest": round(nearest_dist, 4),
            })

        total_dist = euclidean_distance(pa["coordinates"], pb["coordinates"])

        return {
            "pose_a": pose_a,
            "pose_b": pose_b,
            "steps": steps,
            "total_distance": round(total_dist, 4),
            "trajectory": waypoints,
        }

    @mcp.tool()
    def decompose_pose_intent(description: str) -> dict:
        """Decompose natural language into pose 7D coordinates via keyword matching.

        Layer 2: Deterministic classification (0 tokens).

        Args:
            description: Natural language (e.g. "confident powerful stance",
                         "flowing romantic movement", "angular editorial geometry")

        Returns matching poses ranked by keyword overlap score.
        """
        kw_index = get_keyword_index_data()
        tokens = tokenize_intent(description)

        if not tokens:
            return {
                "error": "No meaningful keywords extracted from description.",
                "description": description,
            }

        results = search_by_keywords(tokens, kw_index)

        # Add coordinates for top results
        catalog = get_pose_catalog()
        enriched = []
        for r in results[:10]:
            pose = catalog.get(r["pose"], {})
            enriched.append({
                "pose": r["pose"],
                "score": r["score"],
                "matched_keywords": r["matched_keywords"],
                "coordinates": pose.get("coordinates", {}),
                "category": pose.get("category", ""),
            })

        return {
            "description": description,
            "extracted_tokens": tokens,
            "matches": enriched,
        }
