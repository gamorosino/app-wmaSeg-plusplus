"""Optic radiation segmentation (bsc_opticRadiationSeg_V7)."""

import numpy as np

from ..atlas import roi_from_atlas_nums, plane_from_roi, modify_roi, intersect_rois, make_planar_roi_legacy
from ..streamlines import segment_fascicle_from_connectome, apply_endpoint_criteria
from ..classification import (
    _empty_classification, concat_classification_criteria, extract_stream_ind_by_name,
)

_SIDE_LABELS = ["left", "right"]
_THAL_LUT = [10, 49]
_PAL_LUT = [13, 52]
_HIPPO_LUT = [17, 53]
_VENT_LUT = [4, 43]
_WM_LUT = [2, 41]


def segment_optic_radiation(streamlines, atlas, expand_seg_bool, category_prior):
    """Segment left/right meyer and baum loops of the optic radiation."""
    n = len(streamlines)
    classification = _empty_classification(n)

    for leftright in [0, 1]:
        sidenum = 10000 + (leftright + 1) * 1000

        wm = roi_from_atlas_nums(atlas, [_WM_LUT[leftright]], smooth_kernel=1)
        thalamic_roi = roi_from_atlas_nums(atlas, [_THAL_LUT[leftright]], smooth_kernel=7)

        pre_occip_plane = plane_from_roi(172 + sidenum, "posterior", atlas)
        thal_ant = plane_from_roi(_THAL_LUT[leftright], "anterior", atlas)
        thal_lat = plane_from_roi(_THAL_LUT[leftright], "lateral", atlas)
        hippo_top = plane_from_roi(_HIPPO_LUT[leftright], "superior", atlas)
        thal_post = plane_from_roi(_THAL_LUT[leftright], "posterior", atlas)

        post_cc_plane = plane_from_roi(251, "posterior", atlas)

        post_cing_gy_roi = roi_from_atlas_nums(atlas, [110 + sidenum], smooth_kernel=15)
        med_thal_wm = intersect_rois(wm, post_cing_gy_roi)
        med_thal_wm_cut = modify_roi(atlas, med_thal_wm, post_cc_plane, "posterior")

        hipp_clip = modify_roi(atlas, thal_post, hippo_top, "superior")
        thal_lat_plost = modify_roi(atlas, thal_lat, thal_post, "anterior")

        pal_roi = roi_from_atlas_nums(atlas, [_PAL_LUT[leftright]], smooth_kernel=5)
        pal_wm = intersect_rois(wm, pal_roi)

        put_post_plane = plane_from_roi(_PAL_LUT[leftright], "posterior", atlas)
        ant_thal_exclude_bool = segment_fascicle_from_connectome(
            streamlines, [put_post_plane], ["not"]
        )
        extreme_ant_exclude = segment_fascicle_from_connectome(
            streamlines, [thal_ant], ["not"]
        )

        thal_bool = segment_fascicle_from_connectome(streamlines, [thalamic_roi], ["endpoints"])
        clip_ind = segment_fascicle_from_connectome(streamlines, [hipp_clip], ["not"])

        occip_sub = extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}occipital_to_subcortical"
        )

        lat_thal_ind = segment_fascicle_from_connectome(streamlines, [thal_lat_plost], ["and"])

        meyer_bool = thal_bool & clip_ind & occip_sub & lat_thal_ind & extreme_ant_exclude
        classification = concat_classification_criteria(
            classification, f"{_SIDE_LABELS[leftright]}meyer", meyer_bool
        )

        baum_bool = thal_bool & ~meyer_bool & occip_sub & lat_thal_ind & ant_thal_exclude_bool
        classification = concat_classification_criteria(
            classification, f"{_SIDE_LABELS[leftright]}baum", baum_bool
        )

        if expand_seg_bool:
            optic_bool = apply_endpoint_criteria(streamlines, pre_occip_plane, "posterior", "one")
            vent_expand = roi_from_atlas_nums(atlas, [_VENT_LUT[leftright]], smooth_kernel=5)
            vent_expand_wm = intersect_rois(vent_expand, wm)
            calcerine_expand = roi_from_atlas_nums(atlas, [145 + sidenum], smooth_kernel=5)
            calcerine_expand_wm = intersect_rois(vent_expand, calcerine_expand)
            clacerine_vent_wm = intersect_rois(vent_expand_wm, calcerine_expand_wm)
            hippo_plane = plane_from_roi(_HIPPO_LUT[leftright], "superior", atlas)
            clacerine_vent_wm_cut = modify_roi(atlas, clacerine_vent_wm, hippo_plane, "inferior")

            med_thal_optic_bool = segment_fascicle_from_connectome(
                streamlines,
                [pre_occip_plane, thalamic_roi, thal_ant, med_thal_wm_cut, pal_wm,
                 make_planar_roi_legacy(atlas, 0, "x"), clacerine_vent_wm_cut],
                ["and", "endpoints", "not", "and", "not", "not", "not"],
            )
            calcerine_optic_bool = segment_fascicle_from_connectome(
                streamlines,
                [pre_occip_plane, thalamic_roi, thal_ant, med_thal_wm_cut, pal_wm,
                 make_planar_roi_legacy(atlas, 0, "x"), clacerine_vent_wm_cut],
                ["and", "endpoints", "not", "and", "not", "not", "and"],
            )
            pal_wm_optic_bool = segment_fascicle_from_connectome(
                streamlines,
                [pre_occip_plane, thalamic_roi, thal_ant, med_thal_wm_cut, pal_wm,
                 make_planar_roi_legacy(atlas, 0, "x")],
                ["and", "endpoints", "not", "not", "and", "not"],
            )

            classification = concat_classification_criteria(
                classification,
                f"{_SIDE_LABELS[leftright]}medThalOptic",
                ~meyer_bool, optic_bool, ~baum_bool, med_thal_optic_bool,
                ~pal_wm_optic_bool, ~calcerine_optic_bool,
            )
            classification = concat_classification_criteria(
                classification,
                f"{_SIDE_LABELS[leftright]}calcerineOptic",
                ~meyer_bool, optic_bool, ~baum_bool, ~med_thal_optic_bool,
                ~pal_wm_optic_bool, calcerine_optic_bool,
            )
            classification = concat_classification_criteria(
                classification,
                f"{_SIDE_LABELS[leftright]}medPalOptic",
                ~meyer_bool, optic_bool, ~baum_bool, ~med_thal_optic_bool,
                pal_wm_optic_bool, ~calcerine_optic_bool,
            )

    return classification
