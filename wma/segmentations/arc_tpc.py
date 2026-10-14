"""Posterior Arc and TPC segmentation (bsc_segpArcTPC)."""

import numpy as np

from ..atlas import roi_from_atlas_nums, plane_from_roi, modify_roi, intersect_rois
from ..streamlines import segment_fascicle_from_connectome, tract_by_endpoint_rois
from ..classification import (
    _empty_classification, concat_classification_criteria, reconcile_classifications,
)

_SIDE_LABELS = ["left", "right"]
_PARIETAL_ROIS3 = [157, 127, 168, 136, 126, 125]
_TEMPORAL_ROIS3 = [121, 161, 137, 162, 138, 173]
_PUT_IND = [12, 51]
_VENT_IND = [4, 43]


def segment_arc_tpc(streamlines, atlas):
    """Segment left/right pArc and left/right TPC."""
    n = len(streamlines)
    cls_parc = _empty_classification(n)
    cls_tpc = _empty_classification(n)

    for leftright in [0, 1]:
        sidenum = 10000 + (leftright + 1) * 1000

        parietal_roi = roi_from_atlas_nums(
            atlas, [d + sidenum for d in _PARIETAL_ROIS3], smooth_kernel=1
        )
        temporal_roi = roi_from_atlas_nums(
            atlas, [d + sidenum for d in _TEMPORAL_ROIS3], smooth_kernel=1
        )

        post_put = plane_from_roi(_PUT_IND[leftright], "posterior", atlas)
        temporal_roi = modify_roi(atlas, temporal_roi, post_put, "posterior")

        # Vertical traverse: must pass through supPlane and infPlane, not frontPost
        sup_plane = plane_from_roi(156 + sidenum, "inferior", atlas)
        inf_plane = plane_from_roi(136 + sidenum, "inferior", atlas)
        front_post = plane_from_roi(116 + sidenum, "posterior", atlas)
        front_post = modify_roi(atlas, front_post, sup_plane, "superior")

        vert_ind = segment_fascicle_from_connectome(
            streamlines, [sup_plane, inf_plane, front_post], ["and", "and", "not"]
        )

        endpoint_ind = tract_by_endpoint_rois(streamlines, parietal_roi, temporal_roi)

        # SPL WM = intersect(precuneus inflate=27, IPS inflate=27)
        tpc_mid = roi_from_atlas_nums(atlas, [130 + sidenum], smooth_kernel=27)
        tpc_lat = roi_from_atlas_nums(atlas, [157 + sidenum], smooth_kernel=27)
        tpc_wm = intersect_rois(tpc_mid, tpc_lat)

        spl_ind = segment_fascicle_from_connectome(streamlines, [tpc_wm], ["and"])

        # Exclusion: fusTopPost = top of lingual sulcus, posterior of ventricle
        fus_top = plane_from_roi(162 + sidenum, "superior", atlas)
        vent_post = plane_from_roi(_VENT_IND[leftright], "posterior", atlas)
        fus_top_post = modify_roi(atlas, fus_top, vent_post, "posterior")

        post_remove = segment_fascicle_from_connectome(streamlines, [fus_top_post], ["not"])

        cls_parc = concat_classification_criteria(
            cls_parc,
            f"{_SIDE_LABELS[leftright]}pArc",
            endpoint_ind, vert_ind, ~spl_ind, post_remove,
        )
        cls_tpc = concat_classification_criteria(
            cls_tpc,
            f"{_SIDE_LABELS[leftright]}TPC",
            endpoint_ind, vert_ind, spl_ind,
        )

    return reconcile_classifications(cls_parc, cls_tpc)
