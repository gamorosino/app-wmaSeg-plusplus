"""Category and geometry priors for streamline segmentation."""

import numpy as np

from .atlas import atlas_roi_nums_from_coords, inflate_labels, roi_from_atlas_nums
from .streamlines import (
    segment_fascicle_from_connectome, midpoint_roi_segment, streamline_lengths,
)
from .classification import (
    _empty_classification, concat_classification_criteria, resort_classification,
)


# ---------------------------------------------------------------------------
# ROI label sets (matching bsc_streamlineCategoryPriors_v6 definitions)
# ---------------------------------------------------------------------------

_GREY_MATTER_ROIS = (
    list(range(101 + 11000, 175 + 11000 + 1))
    + list(range(101 + 12000, 175 + 12000 + 1))
)
_LEFT_ROIS = list(range(101 + 11000, 175 + 11000 + 1)) + [26, 17, 18, 7, 8] + list(range(10, 14))
_RIGHT_ROIS = list(range(101 + 12000, 175 + 12000 + 1)) + [46, 47] + list(range(49, 55)) + [58]
_SUBCORTICAL_ROIS = list(range(10, 14)) + list(range(17, 21)) + [26, 58, 27] + list(range(49, 57)) + [59]
_SPINE_ROIS = [16, 28, 60]
_CEREBELLUM_ROIS = [8, 47, 7, 46]
_VENTRICLE_ROIS = [31, 63, 4, 43, 14, 24, 15, 44, 5, 62, 30, 80, 72]
_WM_ROIS = [41, 2]
_CC_ROIS = list(range(251, 256))
_UNKNOWN_ROIS = [0, 2000, 1000] + list(range(77, 83)) + [24, 42, 3]
_OPTIC_CROI = [85]
_PERIC_ROI = [167 + 11000, 167 + 12000]

_FRONTAL_3DIGIT = [124, 148, 118, 165, 101, 154, 105, 115, 154, 155, 115, 170,
                   129, 146, 153, 164, 106, 116, 108, 131, 171, 112, 150, 104,
                   169, 114, 113, 116, 107, 163, 139, 132, 140]
_TEMPORAL_3DIGIT = [144, 134, 138, 137, 173, 174, 135, 175, 121, 151, 123, 162, 133]
_OCCIPITAL_3DIGIT = [120, 119, 111, 158, 166, 143, 145, 159, 152, 122, 162, 161, 121, 160, 102]
_PARIETAL_3DIGIT = [157, 127, 168, 136, 126, 125, 156, 128, 141, 172, 147, 109, 103, 130, 110]
_INSULA_3DIGIT = [117, 149]

def _both_sides(nums_3digit):
    return [n + 11000 for n in nums_3digit] + [n + 12000 for n in nums_3digit]

_FRONTAL_ROIS = set(_both_sides(_FRONTAL_3DIGIT))
_TEMPORAL_ROIS = set(_both_sides(_TEMPORAL_3DIGIT))
_OCCIPITAL_ROIS = set(_both_sides(_OCCIPITAL_3DIGIT))
_PARIETAL_ROIS = set(_both_sides(_PARIETAL_3DIGIT))
_INSULA_ROIS = set(_both_sides(_INSULA_3DIGIT))

_LEFT_SET = frozenset(_LEFT_ROIS)
_RIGHT_SET = frozenset(_RIGHT_ROIS)
_GREY_SET = frozenset(_GREY_MATTER_ROIS)
_VALID_SIDE_SET = _LEFT_SET | _RIGHT_SET
_EXCLUDE_SIDE_SET = frozenset(
    _UNKNOWN_ROIS + list(_PERIC_ROI) + _CC_ROIS + _OPTIC_CROI + _WM_ROIS + _SPINE_ROIS + _VENTRICLE_ROIS
)


def _label_to_region(label):
    """Map an atlas label to a lobar region string."""
    if label in _FRONTAL_ROIS:
        return "frontal"
    if label in _TEMPORAL_ROIS:
        return "temporal"
    if label in _OCCIPITAL_ROIS:
        return "occipital"
    if label in _PARIETAL_ROIS:
        return "parietal"
    if label in frozenset(_SUBCORTICAL_ROIS):
        return "subcortical"
    if label in frozenset(_SPINE_ROIS):
        return "spinal"
    if label in frozenset(_INSULA_ROIS):
        return "insula"
    if label in frozenset(_CEREBELLUM_ROIS):
        return "cerebellum"
    if label in frozenset(_CC_ROIS):
        return "CorpusCallosum"
    if label in frozenset(_VENTRICLE_ROIS):
        return "ventricle"
    if label in frozenset(_UNKNOWN_ROIS):
        return "unlabeled"
    if label in frozenset(_WM_ROIS):
        return "whiteMatter"
    if label in frozenset(_PERIC_ROI):
        return "pericallosal"
    if label in frozenset(_OPTIC_CROI):
        return "OpticChi"
    return None


# ---------------------------------------------------------------------------
# Superficial fibers (bsc_segmentSuperficialFibers)
# ---------------------------------------------------------------------------

def segment_superficial_fibers(streamlines, atlas):
    """Identify U-fibers (superficial): short, midpoint in grey matter."""
    n = len(streamlines)
    classification = _empty_classification(n)
    wm_lut = [2, 41]
    length_limit = 30.0

    inflated = inflate_labels(atlas, 2)
    grey_matter_nums = list(range(101 + 11000, 175 + 11000 + 1)) + list(range(101 + 12000, 175 + 12000 + 1))
    grey_roi = roi_from_atlas_nums(inflated, grey_matter_nums, smooth_kernel=1)

    midpoint_bool = midpoint_roi_segment(streamlines, grey_roi)
    lengths = streamline_lengths(streamlines)

    side_labels = ["left", "right"]
    for leftright in [0, 1]:
        other_wm_idx = wm_lut[1 - leftright]
        wm_degrade = roi_from_atlas_nums(inflated, [wm_lut[leftright]], smooth_kernel=1)
        other_wm_full = roi_from_atlas_nums(atlas, [other_wm_idx], smooth_kernel=5)

        not_wm = ~segment_fascicle_from_connectome(streamlines, [wm_degrade, other_wm_full], ["not", "not"])
        classification = concat_classification_criteria(
            classification,
            f"{side_labels[leftright]}Ufibers",
            lengths < length_limit,
            midpoint_bool,
            not_wm,
        )

    return classification


# ---------------------------------------------------------------------------
# Category priors (bsc_streamlineCategoryPriors_v6)
# ---------------------------------------------------------------------------

def streamline_category_priors(streamlines, atlas, inflate_iter=2):
    """Classify each streamline by lobe-to-lobe category name.

    Mirrors bsc_streamlineCategoryPriors_v6.
    Returns a classification dict.
    """
    n = len(streamlines)
    print("computing superficial fibers...")
    superficial_cls = segment_superficial_fibers(streamlines, atlas)

    print("inflating atlas for category priors...")
    inflated = inflate_labels(atlas, inflate_iter)

    ep1 = np.zeros((3, n), dtype=np.float64)
    ep2 = np.zeros((3, n), dtype=np.float64)
    for i, s in enumerate(streamlines):
        ep1[:, i] = s[:, 0]
        ep2[:, i] = s[:, -1]

    print("getting endpoint identities...")
    id1 = atlas_roi_nums_from_coords(inflated, ep1)  # (N,)
    id2 = atlas_roi_nums_from_coords(inflated, ep2)  # (N,)

    # Compute per-streamline boolean properties
    exclude = np.array([
        id2[i] in _EXCLUDE_SIDE_SET or id1[i] in _EXCLUDE_SIDE_SET
        for i in range(n)
    ], dtype=bool)
    include = np.array([
        id2[i] in _VALID_SIDE_SET or id1[i] in _VALID_SIDE_SET
        for i in range(n)
    ], dtype=bool)
    valid_u = np.array([
        (id2[i] in _GREY_SET or id1[i] in _GREY_SET) and not exclude[i]
        for i in range(n)
    ], dtype=bool)
    left_bool = np.array([
        id1[i] in _LEFT_SET and id2[i] in _LEFT_SET
        for i in range(n)
    ], dtype=bool)
    right_bool = np.array([
        id1[i] in _RIGHT_SET and id2[i] in _RIGHT_SET
        for i in range(n)
    ], dtype=bool)
    inter_hemi_bool = np.array([
        (id1[i] in _LEFT_SET and id2[i] in _RIGHT_SET) or
        (id1[i] in _RIGHT_SET and id2[i] in _LEFT_SET)
        for i in range(n)
    ], dtype=bool)
    single_left = np.array([
        (id1[i] in _LEFT_SET) ^ (id2[i] in _LEFT_SET)
        for i in range(n)
    ], dtype=bool)
    single_right = np.array([
        (id1[i] in _RIGHT_SET) ^ (id2[i] in _RIGHT_SET)
        for i in range(n)
    ], dtype=bool)
    interhemi_flag = inter_hemi_bool & include

    print("building stream names...")
    stream_names = []
    for i in range(n):
        r1 = _label_to_region(id1[i])
        r2 = _label_to_region(id2[i])
        if r1 is None or r2 is None:
            stream_names.append("unlabeled")
            continue

        terms = sorted([r1, r2])
        base_name = f"{terms[0]}_to_{terms[1]}"

        if interhemi_flag[i]:
            base_name += "_interHemi"

        is_superficial = superficial_cls["index"][i] > 0
        if is_superficial and valid_u[i]:
            base_name += "_ufiber"

        if (left_bool[i] or single_left[i]) and include[i] and not interhemi_flag[i]:
            base_name = "left" + base_name
        elif (right_bool[i] or single_right[i]) and include[i] and not interhemi_flag[i]:
            base_name = "right" + base_name

        if interhemi_flag[i] and valid_u[i] and is_superficial:
            base_name = "MaskFailure"

        stream_names.append(base_name)

    print(f"building classification from {len(set(stream_names))} categories...")
    classification = _empty_classification(n)
    _SUMMARIZE = {"CorpusCallosum", "unlabeled", "OpticChi", "ventricle",
                  "whiteMatter", "pericallosal"}

    unique_names = sorted(set(stream_names))
    for cat in unique_names:
        # Check if this category contains a summarized name
        summary_match = [s for s in _SUMMARIZE if s in cat]
        if summary_match:
            summary_key = summary_match[0]
            mask = np.array([sn == cat for sn in stream_names], dtype=bool)
            classification = concat_classification_criteria(classification, summary_key, mask)
        else:
            mask = np.array([sn == cat for sn in stream_names], dtype=bool)
            classification = concat_classification_criteria(classification, cat, mask)

    classification = resort_classification(classification)
    print("categorical segmentation complete")
    return classification


# ---------------------------------------------------------------------------
# Geometry priors (bsc_streamlineGeometryPriors / ConnectomeTestQ_v2)
# ---------------------------------------------------------------------------

def streamline_geometry_priors(streamlines):
    """Compute asymmetry and efficiency ratio priors.

    Mirrors bsc_streamlineGeometryPriors + ConnectomeTestQ_v2.
    Returns (asym_prior, eff_prior) classification dicts.
    """
    n = len(streamlines)
    eff_ratios = np.full(n, np.nan)
    asym_ratios = np.full(n, np.nan)

    for i, s in enumerate(streamlines):
        num_nodes = s.shape[1]
        if num_nodes < 5:
            continue
        mid = num_nodes // 2
        first_half = s[:, :mid]
        second_half = s[:, mid:]

        ep1 = s[:, 0]
        ep2 = s[:, -1]
        mid_pt = s[:, mid]

        def arc_len(seg):
            return float(np.sqrt((np.diff(seg, axis=1) ** 2).sum(axis=0)).sum())

        len1 = arc_len(first_half)
        len2 = arc_len(second_half)
        total_len = len1 + len2
        if total_len == 0:
            continue

        disp1 = float(np.linalg.norm(ep1 - mid_pt))
        disp2 = float(np.linalg.norm(ep2 - mid_pt))
        full_disp = float(np.linalg.norm(ep1 - ep2))

        dr1 = disp1 / len1 if len1 > 0 else 0
        dr2 = disp2 / len2 if len2 > 0 else 0

        eff_ratios[i] = full_disp / total_len
        asym_ratios[i] = (dr1 - dr2) ** 2

    increment = 0.05
    asym_prior = _empty_classification(n)
    eff_prior = _empty_classification(n)

    for lo in np.arange(0.0, 1.0, increment):
        hi = lo + increment
        label = f"{lo:.2f}_to_{hi:.2f}"
        asym_prior = concat_classification_criteria(
            asym_prior, f"{label}_asym",
            ~np.isnan(asym_ratios) & (asym_ratios > lo) & (asym_ratios <= hi),
        )
        eff_prior = concat_classification_criteria(
            eff_prior, f"{label}_eff",
            ~np.isnan(eff_ratios) & (eff_ratios > lo) & (eff_ratios <= hi),
        )

    return asym_prior, eff_prior
