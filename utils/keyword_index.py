"""
Inverted keyword index for pose intent decomposition.

Maps keywords to poses with strength scores and category faceting.
All operations are deterministic and zero-token cost.
"""

from typing import Dict, List, Tuple, Optional


# Keyword categories
KEYWORD_CATEGORIES = {
    "mood": [
        "confident", "commanding", "elegant", "dramatic", "intimate", "thoughtful",
        "pensive", "aggressive", "mysterious", "flirtatious", "assertive", "guarded",
        "cool", "reserved", "relaxed", "casual", "nonchalant", "vulnerable",
        "expressive", "sensual", "haughty", "melancholy", "introspective",
    ],
    "genre": [
        "editorial", "commercial", "runway", "catalog", "street-style", "beauty",
        "cosmetics", "high-fashion", "vogue", "avant-garde", "street",
    ],
    "geometry": [
        "angular", "sinuous", "symmetrical", "asymmetric", "s-curve",
        "architectural", "sculptural", "graphic", "constructed", "geometric",
        "zigzag", "spiral", "torsion", "elongated", "wide-silhouette",
    ],
    "era": [
        "renaissance", "classical", "bauhaus", "contemporary", "90s",
        "modern", "ancient", "greek",
    ],
    "photographer": [
        "avedon", "meisel", "testino", "newton", "leibovitz", "penn",
    ],
    "movement": [
        "dynamic", "static", "caught-in-motion", "windswept", "strut",
        "hip-sway", "departure", "approaching", "freedom", "liberation",
    ],
}


def _categorize_keyword(keyword: str) -> str:
    """Determine which category a keyword belongs to."""
    for cat, words in KEYWORD_CATEGORIES.items():
        if keyword in words:
            return cat
    return "uncategorized"


def build_keyword_index(pose_catalog: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    """Build inverted index from keyword -> [{pose, strength, category}].

    Strength is computed as 1.0 / number_of_keywords_on_that_pose,
    so poses with fewer, more specific keywords get higher strength.
    """
    index: Dict[str, List[Dict]] = {}

    for pose_name, pose_data in pose_catalog.items():
        keywords = pose_data.get("keywords", [])
        if not keywords:
            continue
        strength = 1.0 / len(keywords)
        for kw in keywords:
            cat = _categorize_keyword(kw)
            entry = {
                "pose": pose_name,
                "strength": round(strength, 4),
                "category": cat,
            }
            if kw not in index:
                index[kw] = []
            index[kw].append(entry)

    return index


def search_by_keywords(
    query_keywords: List[str],
    keyword_index: Dict[str, List[Dict]],
) -> List[Dict]:
    """Score poses by keyword overlap with query.

    Returns poses ranked by cumulative strength across matching keywords.
    """
    scores: Dict[str, float] = {}
    matched_kws: Dict[str, List[str]] = {}

    for kw in query_keywords:
        kw_lower = kw.lower().strip()
        if kw_lower in keyword_index:
            for entry in keyword_index[kw_lower]:
                pose = entry["pose"]
                scores[pose] = scores.get(pose, 0.0) + entry["strength"]
                if pose not in matched_kws:
                    matched_kws[pose] = []
                matched_kws[pose].append(kw_lower)

    results = []
    for pose, score in sorted(scores.items(), key=lambda x: -x[1]):
        results.append({
            "pose": pose,
            "score": round(score, 4),
            "matched_keywords": matched_kws.get(pose, []),
        })

    return results


def get_keywords_for_pose(
    pose_name: str,
    pose_catalog: Dict[str, Dict],
    keyword_index: Dict[str, List[Dict]],
) -> List[Dict]:
    """Get all keywords for a specific pose with categories."""
    pose = pose_catalog.get(pose_name)
    if not pose:
        return []

    keywords = pose.get("keywords", [])
    results = []
    for kw in keywords:
        cat = _categorize_keyword(kw)
        # Find other poses sharing this keyword
        related = []
        if kw in keyword_index:
            for entry in keyword_index[kw]:
                if entry["pose"] != pose_name:
                    related.append(entry["pose"])
        results.append({
            "term": kw,
            "category": cat,
            "related_poses": related,
        })
    return results


def get_keywords_by_category(
    category: str,
    keyword_index: Dict[str, List[Dict]],
) -> List[Dict]:
    """Get all keywords in a specific category with their pose associations."""
    results = []
    for kw, entries in keyword_index.items():
        cat = _categorize_keyword(kw)
        if cat == category:
            poses = [{"pose": e["pose"], "strength": e["strength"]} for e in entries]
            results.append({
                "term": kw,
                "category": cat,
                "poses": poses,
            })
    results.sort(key=lambda x: x["term"])
    return results


def tokenize_intent(description: str) -> List[str]:
    """Split a natural language description into keyword tokens.

    Strips common filler words and normalizes.
    """
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "of", "in", "to", "for", "with", "on", "at", "from", "by",
        "about", "as", "into", "through", "during", "before", "after",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "this", "that", "these",
        "those", "i", "me", "my", "we", "our", "you", "your", "it",
        "its", "they", "them", "their", "want", "like", "looking",
        "need", "give", "make", "create", "show", "pose", "shot",
        "photo", "image", "picture", "style", "feel", "look",
        "stance", "position", "standing", "sitting", "model",
    }

    tokens = description.lower().replace(",", " ").replace(".", " ").split()
    # Also split hyphenated words to match both forms
    expanded = []
    for t in tokens:
        expanded.append(t)
        if "-" in t:
            expanded.extend(t.split("-"))
    return [t for t in expanded if t not in stop_words and len(t) > 1]
