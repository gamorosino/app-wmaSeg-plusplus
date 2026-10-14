"""Aslant tract segmentation (bsc_segmentAslant)."""

import numpy as np

from ..atlas import roi_from_atlas_nums, plane_from_roi, modify_roi, make_planar_roi_legacy
from ..streamlines import (
    segment_fascicle_from_connectome, apply_midpoint_criteria, tract_by_endpoint_rois,
)
from ..classification import (
    _empty_classification, concat_classification_criteria, extract_stream_ind_by_name,
)

_SIDE_LABELS = ["left", "right"]


def segment_aslant(streamlines, atlas, category_prior):
    """Segment left and right Aslant tracts."""
    n = len(streamlines)
    classification = _empty_classification(n)

    not_hemi = make_planar_roi_legacy(atlas, 0, "x")

    for leftright in [0, 1]:
        sidenum = 10000 + (leftright + 1) * 1000

        superior_roi = roi_from_atlas_nums(atlas, [116 + sidenum], smooth_kernel=7)
        lateral_roi = roi_from_atlas_nums(atlas, [114 + sidenum, 112 + sidenum], smooth_kernel=1)

        lat_border = plane_from_roi(104 + sidenum, "medial", atlas)
        lateral_roi = modify_roi(atlas, lateral_roi, lat_border, "lateral")

        keep = tract_by_endpoint_rois(streamlines, superior_roi, lateral_roi)

        ant_peri_call = plane_from_roi(167 + sidenum, "anterior", atlas)
        ant_par_sup = plane_from_roi(103 + sidenum, "anterior", atlas)

        prevent_anterior_bool = apply_midpoint_criteria(streamlines, ant_peri_call, "posterior")

        bounded_ind = segment_fascicle_from_connectome(
            streamlines, [ant_peri_call, ant_par_sup, not_hemi], ["not", "not", "not"]
        )

        fronto_frontal_bool = (
            extract_stream_ind_by_name(category_prior, f"{_SIDE_LABELS[leftright]}frontal_to_frontal") |
            extract_stream_ind_by_name(category_prior, f"{_SIDE_LABELS[leftright]}frontal_to_frontal_ufiber")
        )

        classification = concat_classification_criteria(
            classification,
            f"{_SIDE_LABELS[leftright]}Aslant",
            keep, bounded_ind, fronto_frontal_bool, prevent_anterior_bool,
        )

    return classification
