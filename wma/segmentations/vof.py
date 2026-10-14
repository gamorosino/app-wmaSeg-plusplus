"""Vertical occipital fasciculus segmentation (bsc_segmentVOF_v4)."""

import numpy as np

from ..atlas import roi_from_atlas_nums, plane_from_roi, modify_roi
from ..streamlines import (
    segment_fascicle_from_connectome, apply_endpoint_criteria, apply_midpoint_criteria,
    tract_by_endpoint_rois,
)
from ..classification import (
    _empty_classification, concat_classification_criteria, extract_stream_ind_by_name,
)

_SIDE_LABELS = ["left", "right"]
_INFERIOR_OCCIP_ROI_NUMS = [143, 102, 122, 152, 161, 162]
_SUPERIOR_OCCIP_ROI_NUMS = [120, 121, 158, 111, 166, 159]
_WM_LUT = [2, 41]


def segment_vof(streamlines, atlas, category_prior):
    """Segment left and right VOF."""
    n = len(streamlines)
    classification = _empty_classification(n)

    for leftright in [0, 1]:
        sidenum = 10000 + (leftright + 1) * 1000

        inferior_occip_roi = roi_from_atlas_nums(
            atlas, [d + sidenum for d in _INFERIOR_OCCIP_ROI_NUMS], smooth_kernel=1
        )
        superior_occip_roi = roi_from_atlas_nums(
            atlas, [d + sidenum for d in _SUPERIOR_OCCIP_ROI_NUMS], smooth_kernel=1
        )

        inferior_bool = tract_by_endpoint_rois(streamlines, inferior_occip_roi, inferior_occip_roi)
        superior_bool = tract_by_endpoint_rois(streamlines, superior_occip_roi, superior_occip_roi)
        criteria_bool = tract_by_endpoint_rois(streamlines, inferior_occip_roi, superior_occip_roi)

        both_endpoints_inf = apply_endpoint_criteria(
            streamlines, np.array([0.0, 0.0, 0.0]), "inferior", "both"
        )
        sup_endpoints_criter = apply_endpoint_criteria(
            streamlines, np.array([0.0, 0.0, 3.0]), "superior", "one"
        )
        inf_endpoints_criter = apply_endpoint_criteria(
            streamlines, np.array([0.0, 0.0, -3.0]), "inferior", "one"
        )

        vof_bottom_plane = plane_from_roi(162 + sidenum, "superior", atlas)
        vof_inferior_plane = plane_from_roi(159 + sidenum, "inferior", atlas)
        vof_post_limit = plane_from_roi(111 + sidenum, "anterior", atlas)

        vof_midpoint_bool = apply_midpoint_criteria(
            streamlines, vof_post_limit, "posterior", vof_bottom_plane, "superior",
            vof_inferior_plane, "inferior"
        )
        vof_candidate_bool = segment_fascicle_from_connectome(
            streamlines, [vof_post_limit], ["not"]
        )

        occipital_occipital_bool = (
            extract_stream_ind_by_name(
                category_prior, f"{_SIDE_LABELS[leftright]}occipital_to_occipital"
            ) |
            extract_stream_ind_by_name(
                category_prior, f"{_SIDE_LABELS[leftright]}occipital_to_occipital_ufiber"
            )
        )

        vof_bool = (
            ~inferior_bool & ~superior_bool & occipital_occipital_bool &
            criteria_bool & vof_candidate_bool & sup_endpoints_criter & inf_endpoints_criter
        )

        classification = concat_classification_criteria(
            classification, f"{_SIDE_LABELS[leftright]}VOF", vof_bool
        )

    return classification
