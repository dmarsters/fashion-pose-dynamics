"""
Coordinate utilities for 7D pose parameter space.

Distance computation, interpolation, and nearest-neighbor search.
All operations are deterministic and zero-token cost.
"""

import math
from typing import Dict, List, Tuple


PARAMETER_NAMES = [
    "primary_axis_angle",
    "hip_displacement",
    "weight_asymmetry",
    "joint_articulation",
    "negative_space_area",
    "gaze_deviation",
    "dynamic_tension",
]


def euclidean_distance(coords_a: Dict[str, float], coords_b: Dict[str, float]) -> float:
    """Euclidean distance between two coordinate dicts in 7D space."""
    total = 0.0
    for p in PARAMETER_NAMES:
        diff = coords_a.get(p, 0.0) - coords_b.get(p, 0.0)
        total += diff * diff
    return math.sqrt(total)


def interpolate(
    coords_a: Dict[str, float],
    coords_b: Dict[str, float],
    steps: int = 10,
) -> List[Dict[str, float]]:
    """Linear interpolation between two coordinate dicts.

    Returns `steps` points including start and end.
    """
    if steps < 2:
        steps = 2
    trajectory = []
    for i in range(steps):
        t = i / (steps - 1)
        point = {}
        for p in PARAMETER_NAMES:
            va = coords_a.get(p, 0.0)
            vb = coords_b.get(p, 0.0)
            point[p] = va + t * (vb - va)
        trajectory.append(point)
    return trajectory


def find_nearest(
    target_coords: Dict[str, float],
    catalog: Dict[str, Dict],
    max_results: int = 5,
    exclude: str = "",
) -> List[Tuple[str, float]]:
    """Find nearest poses by Euclidean distance in 7D space.

    Args:
        target_coords: 7D coordinate dict to search from.
        catalog: Full pose catalog dict (name -> pose entry).
        max_results: Number of results to return.
        exclude: Pose name to exclude (typically the query pose itself).

    Returns:
        List of (pose_name, distance) tuples sorted by distance.
    """
    distances = []
    for name, pose in catalog.items():
        if name == exclude:
            continue
        coords = pose.get("coordinates", {})
        dist = euclidean_distance(target_coords, coords)
        distances.append((name, dist))
    distances.sort(key=lambda x: x[1])
    return distances[:max_results]


def apply_modifier(
    coords: Dict[str, float],
    modifier: str,
) -> Dict[str, float]:
    """Apply a named modifier to coordinates.

    Modifiers scale coordinates toward extremes or center.
    """
    modified = dict(coords)

    if modifier == "subtle":
        # Pull all values toward 0.3 (conservative center)
        for p in PARAMETER_NAMES:
            v = modified.get(p, 0.0)
            modified[p] = v * 0.6 + 0.3 * 0.4
    elif modifier == "exaggerated":
        # Push values away from center
        for p in PARAMETER_NAMES:
            v = modified.get(p, 0.0)
            center = 0.5
            modified[p] = max(0.0, min(1.0, center + (v - center) * 1.5))
    elif modifier == "editorial":
        # Boost dynamic_tension and joint_articulation
        modified["dynamic_tension"] = min(1.0, modified.get("dynamic_tension", 0.0) + 0.15)
        modified["joint_articulation"] = min(1.0, modified.get("joint_articulation", 0.0) + 0.10)
        modified["negative_space_area"] = min(1.0, modified.get("negative_space_area", 0.0) + 0.10)

    return modified
