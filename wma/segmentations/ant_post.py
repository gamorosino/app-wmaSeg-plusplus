"""Anterior/posterior tract segmentation (bsc_segmentAntPostTracts_v3).

Segments: cingulum (internally), Uncinate, IFOF, Arc, SLF1&2, SLF3.
"""

import numpy as np

from ..atlas import plane_from_roi, modify_roi, merge_rois, roi_from_atlas_nums, make_planar_roi_legacy, multi_intersect_rois
from ..streamlines import segment_fascicle_from_connectome, apply_endpoint_criteria, endpoint_atlas_criteria
from ..classification import (
    _empty_classification, concat_classification_criteria, extract_stream_ind_by_name,
)
from .cingulum import segment_cingulum

_SIDE_LABELS = ["left", "right"]
_LENTI_LUT = [[12, 13], [51, 52]]
_PAL_LUT = [13, 52]
_THAL_LUT = [10, 49]
_AMIG_LUT = [18, 54]


def segment_ant_post_tracts(streamlines, atlas, category_prior, eff_prior):
    """Segment UF, IFOF, Arcuate, SLF1&2, SLF3 per hemisphere."""
    n = len(streamlines)
    inter_hemi_not = make_planar_roi_legacy(atlas, 0, "x")

    # Cingulum is segmented first and used as exclusion
    classification = segment_cingulum(streamlines, atlas, category_prior)
    cingulum_bool = (
        (classification["index"] == (classification["names"].index("rightcingulum") + 1)
         if "rightcingulum" in classification["names"] else np.zeros(n, dtype=bool)) |
        (classification["index"] == (classification["names"].index("leftcingulum") + 1)
         if "leftcingulum" in classification["names"] else np.zeros(n, dtype=bool))
    )

    for leftright in [0, 1]:
        sidenum = 10000 + (leftright + 1) * 1000

        thal_top = plane_from_roi(_THAL_LUT[leftright], "superior", atlas)
        thal_post = plane_from_roi(_THAL_LUT[leftright], "posterior", atlas)
        amig_post = plane_from_roi(_AMIG_LUT[leftright], "posterior", atlas)

        # ---- Uncinate ----
        unc_endpoint_criteria = apply_endpoint_criteria(
            streamlines,
            thal_top, "inferior", "both",
            amig_post, "anterior", "both",
        )
        unc_seg_bool = segment_fascicle_from_connectome(
            streamlines, [thal_top, amig_post, thal_post], ["not", "not", "not"]
        )
        fronto_temporal = (
            extract_stream_ind_by_name(category_prior, f"{_SIDE_LABELS[leftright]}frontal_to_temporal") |
            extract_stream_ind_by_name(category_prior, f"{_SIDE_LABELS[leftright]}frontal_to_temporal_ufiber")
        )
        classification = concat_classification_criteria(
            classification,
            f"{_SIDE_LABELS[leftright]}Uncinate",
            fronto_temporal, unc_seg_bool, unc_endpoint_criteria,
        )

        # ---- IFOF ----
        cc_post_limit = plane_from_roi(251, "posterior", atlas)
        cc_ant_limit = plane_from_roi(255, "anterior", atlas)

        post_cc_top_thal = modify_roi(atlas, cc_post_limit, thal_top, "superior")
        cc_interior1 = modify_roi(atlas, thal_top, cc_post_limit, "anterior")
        cc_interior2 = modify_roi(atlas, cc_interior1, cc_ant_limit, "posterior")
        ant_cc_top_thal = modify_roi(atlas, cc_ant_limit, thal_top, "superior")

        cc_carve_out = merge_rois(post_cc_top_thal, cc_interior2)
        cc_carve_out = merge_rois(cc_carve_out, ant_cc_top_thal)

        ant_temp_plane = plane_from_roi(173 + sidenum, "anterior", atlas)
        inf_cc_limit = plane_from_roi(255, "inferior", atlas)
        inf_temp_roi = modify_roi(atlas, ant_temp_plane, inf_cc_limit, "inferior")

        ifof_bool = segment_fascicle_from_connectome(
            streamlines, [inf_temp_roi, cc_carve_out], ["and", "not"]
        )
        fronto_occipital = extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}frontal_to_occipital"
        )
        classification = concat_classification_criteria(
            classification,
            f"{_SIDE_LABELS[leftright]}IFOF",
            fronto_occipital, ifof_bool,
        )

        # ---- Arcuate ----
        post_cing_not = roi_from_atlas_nums(atlas, [108 + sidenum], smooth_kernel=5)
        lat_fis_inf = plane_from_roi(141 + sidenum, "inferior", atlas)
        ins_post = plane_from_roi(150 + sidenum, "posterior", atlas)
        temp_trans_v_top = plane_from_roi(133 + sidenum, "superior", atlas)

        post_lat_fis_inf = modify_roi(atlas, lat_fis_inf, cc_post_limit, "posterior")
        top_arc_and = modify_roi(atlas, ins_post, temp_trans_v_top, "superior")

        arc_bool = segment_fascicle_from_connectome(
            streamlines, [post_lat_fis_inf, top_arc_and, post_cing_not], ["and", "and", "not"]
        )
        classification = concat_classification_criteria(
            classification,
            f"{_SIDE_LABELS[leftright]}Arc",
            fronto_temporal, arc_bool, ~cingulum_bool,
        )

        # ---- SLF 1&2 / SLF 3 ----
        parieto_frontal = extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}frontal_to_parietal"
        )
        cc_mid_limit = plane_from_roi(252, "anterior", atlas)
        slf12_exclude = modify_roi(atlas, cc_interior2, cc_mid_limit, "posterior")

        interior_wall_bool = endpoint_atlas_criteria(
            streamlines, atlas,
            [116, 109, 108, 107, 167, 147, 172, 130, 110, 106],  # sidenum added below
            "either",
        )
        # Note: bsc_endpointAtlasCriteria adds sidenum internally; here we pass absolute IDs
        interior_wall_bool = endpoint_atlas_criteria(
            streamlines, atlas,
            [n + sidenum for n in [116, 109, 108, 107, 167, 147, 172, 130, 110, 106]],
            "either",
        )

        slf3_intersection = multi_intersect_rois(atlas, 19, 112 + sidenum, 150 + sidenum)
        pal_ant = plane_from_roi(_PAL_LUT[leftright], "anterior", atlas)
        front_sinf_limit = plane_from_roi(155 + sidenum, "inferior", atlas)
        slf3_fix = modify_roi(atlas, pal_ant, front_sinf_limit, "superior")

        slf12_seg_bool = segment_fascicle_from_connectome(
            streamlines, [slf12_exclude, top_arc_and], ["not", "and"]
        )
        slf3_bool = segment_fascicle_from_connectome(
            streamlines, [slf3_intersection, post_lat_fis_inf, slf3_fix], ["endpoints", "not", "not"]
        )

        slf12_bool = slf12_seg_bool & parieto_frontal & ~cingulum_bool & ~ifof_bool & ~interior_wall_bool
        slf3_bool = slf3_bool & parieto_frontal & ~ifof_bool & ~interior_wall_bool

        classification = concat_classification_criteria(
            classification, f"{_SIDE_LABELS[leftright]}SLF1And2", slf12_bool
        )
        classification = concat_classification_criteria(
            classification, f"{_SIDE_LABELS[leftright]}SLF3", slf3_bool
        )

    return classification
