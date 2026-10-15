"""MdLF and ILF segmentation (bsc_segmentMdLF_ILF_v4)."""

import numpy as np

from ..atlas import inflate_labels, roi_from_atlas_nums, plane_from_roi, modify_roi, intersect_rois
from ..streamlines import (
    segment_fascicle_from_connectome, apply_endpoint_criteria, apply_midpoint_criteria,
    tract_by_endpoint_rois,
)
from ..classification import (
    _empty_classification, concat_classification_criteria, extract_stream_ind_by_name,
    reconcile_classifications,
)

_SIDE_LABELS = ["left", "right"]
_AMYG_IDS = [18, 54]
_SUB_CORT_IDS = [[5, 12, 13, 17, 18], [44, 51, 52, 53, 54]]


def segment_mdlf_ilf(streamlines, atlas, category_prior):
    """Segment left/right ILF and MDLFang."""
    n = len(streamlines)
    inflated = inflate_labels(atlas, 2)

    cls_ilf = _empty_classification(n)
    cls_mdlf_ang = _empty_classification(n)

    for leftright in [0, 1]:
        sidenum = 10000 + (leftright + 1) * 1000

        # Occipito ROI (inflated)
        oct_roi = roi_from_atlas_nums(
            inflated, [n + sidenum for n in [120, 111, 166, 143, 119, 158, 122, 102, 119, 159]],
            smooth_kernel=1,
        )
        ant_limit = plane_from_roi(157 + sidenum, "anterior", atlas)
        oct_roi = modify_roi(atlas, oct_roi, ant_limit, "posterior")

        # Parietal ROI (inflated, inflate=5)
        parietal_roi = roi_from_atlas_nums(
            inflated, [d + sidenum for d in [157, 127, 168, 136, 126, 125]], smooth_kernel=5
        )

        # Lateral temporal ROI (inflated, inflate=9)
        lat_temp_roi = roi_from_atlas_nums(
            inflated, [d + sidenum for d in [134, 144, 174, 135]], smooth_kernel=9
        )
        amyg_post = plane_from_roi(_AMYG_IDS[leftright], "posterior", atlas)
        lat_temp_roi = modify_roi(atlas, lat_temp_roi, amyg_post, "anterior")

        # --- ILF ---
        occipito_temporal_bool = extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}occipital_to_temporal"
        )

        ling_gy_ant = plane_from_roi(162 + sidenum, "anterior", atlas)
        posterior_ilf_ant = plane_from_roi(110 + sidenum, "posterior", atlas)

        ilf_ant_endpoint = apply_endpoint_criteria(streamlines, ling_gy_ant, "anterior", "one")
        ilf_post_endpoint = apply_endpoint_criteria(streamlines, posterior_ilf_ant, "posterior", "one")

        ins_post = plane_from_roi(150 + sidenum, "posterior", atlas)
        temp_trans_v_top = plane_from_roi(133 + sidenum, "superior", atlas)
        top_arc_and = modify_roi(atlas, ins_post, temp_trans_v_top, "superior")

        cing_remove_roi = roi_from_atlas_nums(inflated, [109 + sidenum, 110 + sidenum], smooth_kernel=5)

        ilf_bool = segment_fascicle_from_connectome(
            streamlines, [top_arc_and, cing_remove_roi], ["not", "not"]
        )

        cls_ilf = concat_classification_criteria(
            cls_ilf,
            f"{_SIDE_LABELS[leftright]}ILF",
            occipito_temporal_bool, ilf_ant_endpoint, ilf_post_endpoint, ilf_bool,
        )

        # --- MDLFang / MDLFspl ---
        mdlf_ind = tract_by_endpoint_rois(streamlines, parietal_roi, lat_temp_roi)

        sub_cort_roi = roi_from_atlas_nums(atlas, _SUB_CORT_IDS[leftright], smooth_kernel=1)
        anterior_mdlf_limit = plane_from_roi(172 + sidenum, "anterior", atlas)
        lat_fis = plane_from_roi(150 + sidenum, "superior", atlas)
        mdlf_front_rm = modify_roi(atlas, anterior_mdlf_limit, lat_fis, "superior")

        remove_mdlf_ind = segment_fascicle_from_connectome(
            streamlines, [sub_cort_roi, mdlf_front_rm], ["not", "not"]
        )

        temporo_parietal_bool = extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}parietal_to_temporal"
        )

        # SPL WM = intersect of precuneus inflate=27 and IPS inflate=27
        spl_mid = roi_from_atlas_nums(atlas, [130 + sidenum], smooth_kernel=27)
        spl_lat = roi_from_atlas_nums(atlas, [157 + sidenum], smooth_kernel=27)
        spl_wm = intersect_rois(spl_mid, spl_lat)

        spl_ind = segment_fascicle_from_connectome(streamlines, [spl_wm], ["and"])

        subcort_top = plane_from_roi(sub_cort_roi, "superior", atlas)
        bad_midpoints_a = apply_midpoint_criteria(
            streamlines, anterior_mdlf_limit, "anterior", subcort_top, "superior", amyg_post, "posterior"
        )
        bad_midpoints_b = apply_midpoint_criteria(streamlines, amyg_post, "anterior")

        cls_mdlf_ang = concat_classification_criteria(
            cls_mdlf_ang,
            f"{_SIDE_LABELS[leftright]}MDLFang",
            mdlf_ind, remove_mdlf_ind, ~bad_midpoints_a, ~bad_midpoints_b, ~spl_ind,
            temporo_parietal_bool,
        )

    base = _empty_classification(n)
    out = reconcile_classifications(base, cls_ilf)
    out = reconcile_classifications(out, cls_mdlf_ang)
    return out
