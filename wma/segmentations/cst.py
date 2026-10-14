"""Corticospinal tract segmentation (bsc_segmentCST)."""

import numpy as np

from ..atlas import inflate_labels, roi_from_atlas_nums, plane_from_roi
from ..streamlines import segment_fascicle_from_connectome
from ..classification import (
    _empty_classification, concat_classification_criteria,
)

_SIDE_LABELS = ["left", "right"]
_CB_ROI_NUMS = [7, 8, 46, 47]
_THAL_LUT = [10, 49]
_WM_LUT = [2, 41]


def segment_cst(streamlines, atlas, category_prior):
    """Segment left and right corticospinal tract."""
    n = len(streamlines)
    classification = _empty_classification(n)
    inflated = inflate_labels(atlas, 2)

    # nonSubCortBool: streamlines whose category is NOT a subcortical category
    non_subcort_indices = [
        i + 1 for i, name in enumerate(category_prior["names"]) if "subcortical" not in name
    ]
    non_subcort_bool = np.isin(category_prior["index"], non_subcort_indices)

    # nonCebBool: streamlines whose category is NOT a cerebellar category
    non_ceb_indices = [
        i + 1 for i, name in enumerate(category_prior["names"]) if "cerebellum" not in name
    ]
    non_ceb_bool = np.isin(category_prior["index"], non_ceb_indices)

    cb_roi = roi_from_atlas_nums(inflated, _CB_ROI_NUMS, smooth_kernel=1)
    spine_roi = roi_from_atlas_nums(inflated, [16], smooth_kernel=1)

    for leftright in [0, 1]:
        sidenum = 10000 + (leftright + 1) * 1000

        thal_ant_plane = plane_from_roi(_THAL_LUT[leftright], "anterior", atlas)
        motor_roi = roi_from_atlas_nums(
            inflated, [d + sidenum for d in [168, 128, 146, 129, 170, 103]], smooth_kernel=1
        )
        other_wm_id = 41 if leftright == 0 else 2
        other_wm = roi_from_atlas_nums(atlas, [other_wm_id], smooth_kernel=1)

        cst_bool = segment_fascicle_from_connectome(
            streamlines,
            [motor_roi, spine_roi, other_wm, cb_roi, thal_ant_plane],
            ["endpoints", "and", "not", "not", "not"],
        )

        classification = concat_classification_criteria(
            classification,
            f"{_SIDE_LABELS[leftright]}CST",
            non_subcort_bool, non_ceb_bool, cst_bool,
        )

    return classification
