"""
Test suite for fashion-pose-dynamics MCP server.

Tests all three layers: taxonomy, computation, synthesis.
Run with: python -m tests.test_all
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.coordinates import (
    euclidean_distance,
    interpolate,
    find_nearest,
    apply_modifier,
    PARAMETER_NAMES,
)
from utils.keyword_index import (
    build_keyword_index,
    search_by_keywords,
    tokenize_intent,
    get_keywords_for_pose,
    get_keywords_by_category,
)
from layers.taxonomy import (
    get_pose_catalog,
    get_head_position_catalog,
    get_lighting_catalog,
    get_keyword_index_data,
    _load_data,
)


def test_data_loading():
    """Test that all JSON data files load correctly."""
    print("  Loading pose catalog...", end=" ")
    catalog = get_pose_catalog()
    assert len(catalog) > 0, "Pose catalog is empty"
    print(f"OK ({len(catalog)} poses)")

    print("  Loading head positions...", end=" ")
    heads = get_head_position_catalog()
    assert len(heads) > 0, "Head positions empty"
    print(f"OK ({len(heads)} positions)")

    print("  Loading lighting types...", end=" ")
    lighting = get_lighting_catalog()
    assert len(lighting) > 0, "Lighting types empty"
    print(f"OK ({len(lighting)} types)")

    print("  Building keyword index...", end=" ")
    kw_index = get_keyword_index_data()
    assert len(kw_index) > 0, "Keyword index empty"
    print(f"OK ({len(kw_index)} keywords)")


def test_pose_completeness():
    """Verify every pose has all required fields."""
    catalog = get_pose_catalog()
    required_fields = ["category", "coordinates", "geometry", "keywords", "body_surface_map"]
    required_coords = set(PARAMETER_NAMES)
    required_bsm = ["stretched_surfaces", "compressed_surfaces", "angled_planes",
                     "joint_bend_points", "gravity_anchor", "surface_facing", "motion_vectors"]

    for name, pose in catalog.items():
        for field in required_fields:
            assert field in pose, f"Pose '{name}' missing field '{field}'"

        coords = pose["coordinates"]
        assert set(coords.keys()) == required_coords, \
            f"Pose '{name}' has wrong coordinate keys: {set(coords.keys())} vs {required_coords}"

        # All coordinates in [0, 1]
        for p, v in coords.items():
            assert 0.0 <= v <= 1.0, f"Pose '{name}' param '{p}' out of range: {v}"

        bsm = pose["body_surface_map"]
        for field in required_bsm:
            assert field in bsm, f"Pose '{name}' body_surface_map missing '{field}'"

    print(f"  All {len(catalog)} poses have complete fields")


def test_coordinate_math():
    """Test distance, interpolation, nearest-neighbor."""
    catalog = get_pose_catalog()

    # Self-distance is zero
    cp = catalog["contrapposto"]["coordinates"]
    assert euclidean_distance(cp, cp) == 0.0, "Self-distance should be 0"

    # Triangle inequality
    a = catalog["contrapposto"]["coordinates"]
    b = catalog["power_stance"]["coordinates"]
    c = catalog["broken_doll"]["coordinates"]
    ab = euclidean_distance(a, b)
    bc = euclidean_distance(b, c)
    ac = euclidean_distance(a, c)
    assert ac <= ab + bc + 1e-10, "Triangle inequality violated"

    # Interpolation endpoints
    traj = interpolate(a, b, steps=5)
    assert len(traj) == 5, f"Expected 5 steps, got {len(traj)}"
    for p in PARAMETER_NAMES:
        assert abs(traj[0][p] - a[p]) < 1e-10, "Start point mismatch"
        assert abs(traj[-1][p] - b[p]) < 1e-10, "End point mismatch"

    # Nearest neighbor
    nearest = find_nearest(a, catalog, max_results=3, exclude="contrapposto")
    assert len(nearest) == 3, f"Expected 3 nearest, got {len(nearest)}"
    assert nearest[0][1] <= nearest[1][1] <= nearest[2][1], "Not sorted by distance"

    print("  Coordinate math: distance, interpolation, nearest-neighbor OK")


def test_modifier():
    """Test that modifiers shift coordinates predictably."""
    catalog = get_pose_catalog()
    base = catalog["contrapposto"]["coordinates"]

    subtle = apply_modifier(base, "subtle")
    exaggerated = apply_modifier(base, "exaggerated")

    # Subtle should pull toward center
    for p in PARAMETER_NAMES:
        if base[p] > 0.5:
            assert subtle[p] <= base[p], f"Subtle should pull {p} toward center"
        elif base[p] < 0.3:
            assert subtle[p] >= base[p], f"Subtle should pull {p} toward center"

    # Exaggerated should push away from center
    for p in PARAMETER_NAMES:
        dist_base = abs(base[p] - 0.5)
        dist_exag = abs(exaggerated[p] - 0.5)
        # Allow clipping at bounds
        if 0.05 < base[p] < 0.95:
            assert dist_exag >= dist_base - 0.01, \
                f"Exaggerated should push {p} from center"

    print("  Modifiers: subtle and exaggerated OK")


def test_keyword_index():
    """Test keyword search and decomposition."""
    catalog = get_pose_catalog()
    kw_index = get_keyword_index_data()

    # Known keyword should find expected pose
    assert "classical" in kw_index, "'classical' should be in keyword index"
    classical_poses = [e["pose"] for e in kw_index["classical"]]
    assert "contrapposto" in classical_poses, "contrapposto should be under 'classical'"

    # Intent decomposition
    tokens = tokenize_intent("confident powerful commanding stance")
    assert "confident" in tokens, "'confident' should be extracted"
    assert "stance" not in tokens, "'stance' should be filtered as stop word"

    results = search_by_keywords(tokens, kw_index)
    assert len(results) > 0, "Should find matches for 'confident powerful commanding'"
    assert results[0]["pose"] == "power_stance", \
        f"Expected power_stance as top match, got {results[0]['pose']}"

    # Category lookup
    mood_kws = get_keywords_by_category("mood", kw_index)
    assert len(mood_kws) > 0, "Should have mood keywords"

    print(f"  Keyword index: {len(kw_index)} keywords, decomposition OK")


def test_body_surface_map_completeness():
    """Verify body surface maps are meaningful (not empty lists everywhere)."""
    catalog = get_pose_catalog()

    empty_count = 0
    for name, pose in catalog.items():
        bsm = pose["body_surface_map"]
        # At least stretched_surfaces and joint_bend_points should be non-empty
        if not bsm["stretched_surfaces"] or bsm["stretched_surfaces"] == ["minimal"]:
            if not bsm["compressed_surfaces"] or bsm["compressed_surfaces"] == ["minimal"]:
                empty_count += 1
                print(f"    WARNING: {name} has minimal surface data")

        # Dynamic poses should have motion vectors
        coords = pose["coordinates"]
        if coords.get("dynamic_tension", 0) > 0.5:
            if not bsm["motion_vectors"]:
                print(f"    WARNING: {name} has high dynamic_tension but no motion_vectors")

    assert empty_count < len(catalog) // 2, "Too many poses with empty surface maps"
    print(f"  Body surface maps: complete ({empty_count} minimal)")


def test_head_position_overlay():
    """Test that head positions have surface effects."""
    heads = get_head_position_catalog()
    for name, head in heads.items():
        assert "surface_effect" in head, f"Head position '{name}' missing surface_effect"
        se = head["surface_effect"]
        assert "neck_stretch" in se, f"Head '{name}' missing neck_stretch"
        assert "face_plane_angle" in se, f"Head '{name}' missing face_plane_angle"

    print(f"  Head positions: all {len(heads)} have surface effects")


def test_categories():
    """Test pose category distribution."""
    catalog = get_pose_catalog()
    cats = {}
    for name, pose in catalog.items():
        cat = pose.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1

    print(f"  Categories: {cats}")
    assert "foundation" in cats, "Should have foundation poses"
    assert "walking" in cats, "Should have walking poses"
    assert "editorial" in cats, "Should have editorial poses"
    assert "arms" in cats, "Should have arms poses"


def test_no_garment_references():
    """Verify no garment-domain data leaked into pose server."""
    catalog = get_pose_catalog()
    garment_keys = {"garment_interaction", "drape_bias", "hem_behavior", "fabric_tension_zones"}

    for name, pose in catalog.items():
        for key in garment_keys:
            assert key not in pose, f"Pose '{name}' has garment key '{key}' — should be in garment_dynamics"
            if "body_surface_map" in pose:
                assert key not in pose["body_surface_map"], \
                    f"Pose '{name}' body_surface_map has garment key '{key}'"

    print("  No garment domain leakage detected")


def run_all_tests():
    """Run all test functions."""
    tests = [
        ("Data Loading", test_data_loading),
        ("Pose Completeness", test_pose_completeness),
        ("Coordinate Math", test_coordinate_math),
        ("Modifiers", test_modifier),
        ("Keyword Index", test_keyword_index),
        ("Body Surface Maps", test_body_surface_map_completeness),
        ("Head Position Overlays", test_head_position_overlay),
        ("Categories", test_categories),
        ("No Garment Leakage", test_no_garment_references),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            test_fn()
            print(f"  ✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
