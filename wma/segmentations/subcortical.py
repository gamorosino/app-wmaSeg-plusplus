"""Subcortical tract segmentation (bsc_segmentSubCortical_v2)."""

import numpy as np

from ..atlas import inflate_labels, roi_from_atlas_nums, plane_from_roi, modify_roi, merge_rois
from ..streamlines import segment_fascicle_from_connectome, apply_endpoint_criteria
from ..classification import (
    _empty_classification, concat_classification_criteria, extract_stream_ind_by_name,
)

_SIDE_LABELS = ["left", "right"]
_LENTI_LUT = [[12, 13], [51, 52]]
_PUT_LUT = [12, 51]
_PAL_LUT = [13, 52]
_THALAMUS_LUT = [10, 49]


def segment_subcortical(streamlines, atlas, category_prior, eff_prior):
    """Segment frontoThalamic, temporoThalamic, parietoThalamic, motorThalamic, spinoThalamic."""
    n = len(streamlines)
    classification = _empty_classification(n)
    inflated = inflate_labels(atlas, 2)

    frontal_3digit = [124, 148, 118, 165, 101, 154, 105, 115, 154, 155, 170, 129, 146, 153,
                      164, 106, 116, 108, 131, 171, 112, 150, 104, 169, 114, 113, 103, 107,
                      117, 132, 139, 140, 142, 163]
    temporal_3digit = [144, 134, 138, 137, 173, 174, 135, 175, 121, 151, 123, 162, 133, 149]
    occipital_3digit = [120, 119, 111, 158, 166, 143, 145, 159, 152, 122, 162, 161, 102, 160]
    parietal_3digit = [157, 127, 168, 136, 126, 125, 156, 128, 141, 172, 147, 109, 110, 130]
    motor_3digit = [168, 128, 146, 129, 170, 103]

    for leftright in [0, 1]:
        sidenum = 10000 + (leftright + 1) * 1000

        frontal_roi = roi_from_atlas_nums(inflated, [d + sidenum for d in frontal_3digit], smooth_kernel=1)
        temporal_roi = roi_from_atlas_nums(inflated, [d + sidenum for d in temporal_3digit], smooth_kernel=1)
        occipital_roi = roi_from_atlas_nums(inflated, [d + sidenum for d in occipital_3digit], smooth_kernel=1)
        parietal_roi = roi_from_atlas_nums(inflated, [d + sidenum for d in parietal_3digit], smooth_kernel=1)

        lenti_roi = roi_from_atlas_nums(inflated, _LENTI_LUT[leftright], smooth_kernel=1)
        thalamic_roi = roi_from_atlas_nums(inflated, [_THALAMUS_LUT[leftright]], smooth_kernel=1)
        pal_roi = roi_from_atlas_nums(inflated, [_PAL_LUT[leftright]], smooth_kernel=1)
        put_roi = roi_from_atlas_nums(inflated, [_PUT_LUT[leftright]], smooth_kernel=1)

        fronto_motor_limit = plane_from_roi(170 + sidenum, "anterior", inflated)
        front_med_roi = roi_from_atlas_nums(inflated, [116 + sidenum], smooth_kernel=1)
        fronto_med_motor_roi = modify_roi(inflated, front_med_roi, fronto_motor_limit, "posterior")
        motor_roi_base = roi_from_atlas_nums(inflated, [d + sidenum for d in motor_3digit], smooth_kernel=1)
        motor_roi = merge_rois(fronto_med_motor_roi, motor_roi_base)

        inferior_thal_limit = plane_from_roi(thalamic_roi, "inferior", atlas)
        anterior_thal_limit = plane_from_roi(thalamic_roi, "anterior", atlas)
        posterior_thal_limit = plane_from_roi(thalamic_roi, "posterior", atlas)
        superior_thal_limit = plane_from_roi(thalamic_roi, "superior", atlas)
        lateral_thalamic_limit = plane_from_roi(thalamic_roi, "lateral", atlas)
        posterior_lenti_limit = plane_from_roi(_LENTI_LUT[leftright], "posterior", atlas)
        ant_pal_limit = plane_from_roi(_PAL_LUT[leftright], "anterior", atlas)
        post_pal_limit = plane_from_roi(_PAL_LUT[leftright], "posterior", atlas)
        inf_pal_limit = plane_from_roi(_PAL_LUT[leftright], "inferior", atlas)

        ant_spine_limit = plane_from_roi(16, "anterior", atlas)
        post_put_exclude = modify_roi(inflated, put_roi, ant_spine_limit, "posterior")

        superior_frontal_limit = plane_from_roi(112 + sidenum, "superior", atlas)
        anterior_frontal_limit = plane_from_roi(155 + sidenum, "anterior", atlas)
        anterior_superior_frontal_limit = modify_roi(inflated, superior_frontal_limit, anterior_frontal_limit, "posterior")

        post_peri_c = plane_from_roi(167 + sidenum, "posterior", atlas)

        posterior_inferior_exclude = modify_roi(inflated, inferior_thal_limit, anterior_thal_limit, "posterior")
        posterior_superior_exclude = modify_roi(inflated, superior_thal_limit, anterior_thal_limit, "posterior")
        posterior_lateral_exclude = modify_roi(inflated, lateral_thalamic_limit, anterior_thal_limit, "posterior")
        anterior_lateral_exclude = modify_roi(inflated, ant_pal_limit, lateral_thalamic_limit, "lateral")
        middle_lateral_exclude = modify_roi(inflated, posterior_lenti_limit, lateral_thalamic_limit, "medial")
        post_pal_lateral_exclude = modify_roi(inflated, lateral_thalamic_limit, post_pal_limit, "posterior")
        anterior_medial_exclude = modify_roi(inflated, anterior_thal_limit, lateral_thalamic_limit, "medial")

        _, posterior_thal_limit_bool = None, segment_fascicle_from_connectome(
            streamlines, [posterior_thal_limit], ["not"]
        )
        _, fronto_motor_bool = None, segment_fascicle_from_connectome(
            streamlines, [motor_roi], ["endpoints"]
        )
        _, frontal_bool = None, segment_fascicle_from_connectome(
            streamlines,
            [frontal_roi, posterior_inferior_exclude, posterior_superior_exclude, posterior_lateral_exclude],
            ["endpoints", "not", "not", "not"],
        )
        _, temporal_bool = None, segment_fascicle_from_connectome(
            streamlines, [temporal_roi], ["endpoints"]
        )
        _, occipital_bool = None, segment_fascicle_from_connectome(
            streamlines, [occipital_roi], ["endpoints"]
        )
        _, parietal_bool = None, segment_fascicle_from_connectome(
            streamlines, [parietal_roi, posterior_inferior_exclude, anterior_thal_limit], ["endpoints", "not", "not"]
        )
        _, fronto_thalamic_bool = None, segment_fascicle_from_connectome(
            streamlines,
            [thalamic_roi, posterior_lateral_exclude, posterior_inferior_exclude, middle_lateral_exclude, anterior_superior_frontal_limit],
            ["endpoints", "not", "not", "not", "not"],
        )
        _, temporo_thalamic_bool = None, segment_fascicle_from_connectome(
            streamlines,
            [thalamic_roi, post_peri_c, anterior_lateral_exclude, superior_thal_limit],
            ["endpoints", "not", "not", "not"],
        )
        _, parieto_thalamic_bool = None, segment_fascicle_from_connectome(
            streamlines,
            [thalamic_roi, ant_pal_limit, inferior_thal_limit],
            ["endpoints", "not", "not"],
        )
        _, motor_thalamic_bool = None, segment_fascicle_from_connectome(
            streamlines,
            [thalamic_roi, inferior_thal_limit, anterior_lateral_exclude],
            ["endpoints", "not", "not"],
        )
        _, spino_thalamic_bool = None, segment_fascicle_from_connectome(
            streamlines,
            [thalamic_roi, superior_thal_limit, anterior_thal_limit],
            ["endpoints", "not", "not"],
        )

        fronto_endpoint_bool = apply_endpoint_criteria(
            streamlines, anterior_thal_limit, "anterior", "one"
        )
        lateral_thal_endpoint_bool = apply_endpoint_criteria(
            streamlines, inf_pal_limit, "inferior", "one"
        )

        frontal_subcortical_bool = extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}frontal_to_subcortical"
        )
        parietal_subcortical_bool = extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}parietal_to_subcortical"
        )
        temporal_subcortical_bool = extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}subcortical_to_temporal"
        )
        spinal_subcortical_bool = extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}spinal_to_subcortical"
        )

        motor_bool_prior = parietal_subcortical_bool | frontal_subcortical_bool

        classification = concat_classification_criteria(
            classification,
            f"{_SIDE_LABELS[leftright]}frontoThalamic",
            fronto_thalamic_bool, fronto_endpoint_bool, frontal_bool, ~fronto_motor_bool, frontal_subcortical_bool,
        )
        classification = concat_classification_criteria(
            classification,
            f"{_SIDE_LABELS[leftright]}temporoThalamic",
            temporo_thalamic_bool, temporal_bool, temporal_subcortical_bool, lateral_thal_endpoint_bool,
        )
        classification = concat_classification_criteria(
            classification,
            f"{_SIDE_LABELS[leftright]}parietoThalamic",
            parieto_thalamic_bool, parietal_bool, parietal_subcortical_bool,
        )
        classification = concat_classification_criteria(
            classification,
            f"{_SIDE_LABELS[leftright]}motorThalamic",
            motor_thalamic_bool, ~frontal_bool, fronto_motor_bool, motor_bool_prior,
        )
        classification = concat_classification_criteria(
            classification,
            f"{_SIDE_LABELS[leftright]}spinoThalamic",
            spino_thalamic_bool, spinal_subcortical_bool,
        )

    return classification
