"""Cingulum segmentation (bsc_segmentCingulum_v3)."""

import numpy as np

from ..atlas import plane_from_roi, modify_roi
from ..streamlines import segment_fascicle_from_connectome
from ..classification import (
    _empty_classification, concat_classification_criteria, extract_stream_ind_by_name,
)

_THAL_IDS = [10, 49]
_SIDE_LABELS = ["left", "right"]


def segment_cingulum(streamlines, atlas, category_prior):
    """Segment left and right cingulum."""
    n = len(streamlines)
    classification = _empty_classification(n)

    for leftright in [0, 1]:
        sidenum = 10000 + (leftright + 1) * 1000
        thal_top = plane_from_roi(_THAL_IDS[leftright], "superior", atlas)
        cc_post_limit = plane_from_roi(251, "posterior", atlas)
        cc_ant_limit = plane_from_roi(254, "anterior", atlas)

        post_cc_top_thal = modify_roi(atlas, cc_post_limit, thal_top, "superior")
        cc_interior1 = modify_roi(atlas, thal_top, cc_post_limit, "anterior")
        cc_interior2 = modify_roi(atlas, cc_interior1, cc_ant_limit, "posterior")

        peri_c_sup_lim = plane_from_roi(156 + sidenum, "superior", atlas)
        peri_c_lat_lim = plane_from_roi(167 + sidenum, "lateral", atlas)
        cc_mid_ant = plane_from_roi(252, "anterior", atlas)

        cing_inc_plane = modify_roi(atlas, cc_mid_ant, peri_c_lat_lim, "medial")
        cing_sup_exc_plane = modify_roi(atlas, cc_mid_ant, peri_c_sup_lim, "superior")

        _, cing_seg_bool = None, segment_fascicle_from_connectome(
            streamlines,
            [cing_inc_plane, cing_sup_exc_plane, cc_interior2],
            ["and", "not", "not"],
        )

        parieto_frontal = extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}frontal_to_parietal"
        )
        fronto_temporal = extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}frontal_to_temporal"
        )
        class_bool = parieto_frontal | fronto_temporal
        cing_rem_bool = cing_seg_bool & class_bool

        classification = concat_classification_criteria(
            classification,
            f"{_SIDE_LABELS[leftright]}cingulum",
            cing_rem_bool,
        )

    return classification
