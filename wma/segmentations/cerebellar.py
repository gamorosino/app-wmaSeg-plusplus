"""Cerebellar tract segmentation (bsc_segmentCerebellarTracts_v2)."""

import numpy as np

from ..atlas import inflate_labels, roi_from_atlas_nums, plane_from_roi, modify_roi, merge_rois
from ..streamlines import segment_fascicle_from_connectome
from ..classification import (
    _empty_classification, concat_classification_criteria, extract_stream_ind_by_name,
)

_SIDE_LABELS = ["left", "right"]
_CB_ROI_NUMS = [[7, 8], [46, 47]]
_THALAMUS_LUT = [10, 49]


def segment_cerebellar_tracts(streamlines, atlas, experimental_bool, category_prior):
    """Segment cerebellar tracts."""
    n = len(streamlines)
    classification = _empty_classification(n)
    inflated = inflate_labels(atlas, 2)

    spine_limit = plane_from_roi(16, "anterior", atlas)
    posterior_streams = segment_fascicle_from_connectome(streamlines, [spine_limit], ["not"])

    for leftright in [0, 1]:
        sidenum = 10000 + (leftright + 1) * 1000

        cb_roi = roi_from_atlas_nums(inflated, _CB_ROI_NUMS[leftright], smooth_kernel=1)
        thal_roi = roi_from_atlas_nums(inflated, [_THALAMUS_LUT[leftright]], smooth_kernel=1)
        ant_ceb_split = plane_from_roi(_THALAMUS_LUT[leftright], "anterior", atlas)
        thal_top = plane_from_roi(_THALAMUS_LUT[leftright], "superior", atlas)

        fronto_motor_limit = plane_from_roi(170 + sidenum, "anterior", inflated)
        front_med_roi = roi_from_atlas_nums(inflated, [116 + sidenum], smooth_kernel=1)
        fronto_med_motor_roi = modify_roi(inflated, front_med_roi, fronto_motor_limit, "posterior")
        motor_roi_base = roi_from_atlas_nums(
            inflated, [d + sidenum for d in [168, 128, 146, 129, 170, 103]], smooth_kernel=1
        )
        motor_roi = merge_rois(fronto_med_motor_roi, motor_roi_base)

        anterio_fronto_bool = segment_fascicle_from_connectome(
            streamlines, [cb_roi, ant_ceb_split], ["endpoints", "and"]
        )
        middle_fronto_bool = segment_fascicle_from_connectome(
            streamlines, [cb_roi, ant_ceb_split], ["endpoints", "not"]
        )
        thal_ceb_bool = segment_fascicle_from_connectome(
            streamlines, [cb_roi, thal_roi, spine_limit, thal_top],
            ["endpoints", "endpoints", "not", "not"],
        )
        motor_ceb_bool = segment_fascicle_from_connectome(
            streamlines, [cb_roi, motor_roi], ["endpoints", "endpoints"]
        )
        this_ceb_bool = segment_fascicle_from_connectome(streamlines, [cb_roi], ["endpoints"])

        motor_ceb_bool = motor_ceb_bool & (
            extract_stream_ind_by_name(category_prior, f"{_SIDE_LABELS[leftright]}cerebellum_to_frontal") |
            extract_stream_ind_by_name(category_prior, f"{_SIDE_LABELS[leftright]}cerebellum_to_parietal")
        )
        anterio_fronto_bool = anterio_fronto_bool & extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}cerebellum_to_frontal"
        )
        middle_fronto_bool = (
            ~motor_ceb_bool & middle_fronto_bool &
            extract_stream_ind_by_name(category_prior, f"{_SIDE_LABELS[leftright]}cerebellum_to_frontal")
        )
        thal_ceb_bool = thal_ceb_bool & extract_stream_ind_by_name(
            category_prior, f"{_SIDE_LABELS[leftright]}cerebellum_to_subcortical"
        )
        occipito_ceb_bool = (
            posterior_streams & this_ceb_bool &
            extract_stream_ind_by_name(category_prior, f"{_SIDE_LABELS[leftright]}cerebellum_to_occipital")
        )
        parieto_ceb_bool = (
            posterior_streams & this_ceb_bool & ~motor_ceb_bool &
            extract_stream_ind_by_name(category_prior, f"{_SIDE_LABELS[leftright]}cerebellum_to_parietal")
        )

        classification = concat_classification_criteria(
            classification, f"{_SIDE_LABELS[leftright]}MotorCerebellar", motor_ceb_bool
        )
        classification = concat_classification_criteria(
            classification, f"{_SIDE_LABELS[leftright]}AnterioFrontoCerebellar", anterio_fronto_bool
        )
        classification = concat_classification_criteria(
            classification, f"{_SIDE_LABELS[leftright]}ThalamicoCerebellar", thal_ceb_bool
        )
        classification = concat_classification_criteria(
            classification, f"{_SIDE_LABELS[leftright]}OccipitoCerebellar", occipito_ceb_bool
        )
        classification = concat_classification_criteria(
            classification, f"{_SIDE_LABELS[leftright]}ParietoCerebellar", parieto_ceb_bool
        )

        if experimental_bool:
            classification = concat_classification_criteria(
                classification, f"{_SIDE_LABELS[leftright]}MiddleFrontoCerebellar", middle_fronto_bool
            )

        # Contralateral connections
        other_side = 1 - leftright
        other_cb_roi = roi_from_atlas_nums(inflated, _CB_ROI_NUMS[other_side], smooth_kernel=1)
        other_wm_id = 41 if leftright == 0 else 2
        other_wm = roi_from_atlas_nums(atlas, [other_wm_id], smooth_kernel=1)

        not_these = segment_fascicle_from_connectome(streamlines, [other_wm], ["and"])

        contra_anterio_fronto_bool = segment_fascicle_from_connectome(
            streamlines, [other_cb_roi, ant_ceb_split], ["endpoints", "and"]
        )
        contra_middle_fronto_bool = segment_fascicle_from_connectome(
            streamlines, [other_cb_roi, ant_ceb_split], ["endpoints", "not"]
        )
        contra_thal_ceb_bool = segment_fascicle_from_connectome(
            streamlines, [other_cb_roi, thal_roi, spine_limit], ["endpoints", "endpoints", "not"]
        )
        contra_motor_ceb_bool = segment_fascicle_from_connectome(
            streamlines, [other_cb_roi, motor_roi], ["endpoints", "endpoints"]
        )
        contra_this_ceb_bool = segment_fascicle_from_connectome(
            streamlines, [other_cb_roi], ["endpoints"]
        )

        contra_motor_ceb_bool = ~not_these & contra_motor_ceb_bool & (
            extract_stream_ind_by_name(category_prior, "cerebellum_to_frontal_interHemi") |
            extract_stream_ind_by_name(category_prior, "cerebellum_to_parietal_interHemi")
        )
        contra_anterio_fronto_bool = (
            ~not_these & contra_anterio_fronto_bool &
            extract_stream_ind_by_name(category_prior, "cerebellum_to_frontal_interHemi")
        )
        contra_middle_fronto_bool = (
            ~not_these & ~contra_motor_ceb_bool & contra_middle_fronto_bool &
            extract_stream_ind_by_name(category_prior, "cerebellum_to_frontal_interHemi")
        )
        contra_thal_ceb_bool = (
            ~not_these & contra_thal_ceb_bool &
            extract_stream_ind_by_name(category_prior, "cerebellum_to_subcortical_interHemi")
        )
        contra_occipito_ceb_bool = (
            ~not_these & posterior_streams & contra_this_ceb_bool &
            extract_stream_ind_by_name(category_prior, "cerebellum_to_occipital_interHemi")
        )
        contra_parieto_ceb_bool = (
            ~not_these & posterior_streams & contra_this_ceb_bool & ~contra_motor_ceb_bool &
            extract_stream_ind_by_name(category_prior, "cerebellum_to_parietal_interHemi")
        )

        classification = concat_classification_criteria(
            classification, f"{_SIDE_LABELS[leftright]}ContraMotorCerebellar", contra_motor_ceb_bool
        )
        classification = concat_classification_criteria(
            classification, f"{_SIDE_LABELS[leftright]}ContraAnterioFrontoCerebellar",
            contra_anterio_fronto_bool
        )

        if experimental_bool:
            classification = concat_classification_criteria(
                classification, f"{_SIDE_LABELS[leftright]}ContraOccipitoCerebellar",
                contra_occipito_ceb_bool
            )
            classification = concat_classification_criteria(
                classification, f"{_SIDE_LABELS[leftright]}ContraParietoCerebellar",
                contra_parieto_ceb_bool
            )
            classification = concat_classification_criteria(
                classification, f"{_SIDE_LABELS[leftright]}ContraMiddleFrontoBoolCerebellar",
                contra_middle_fronto_bool
            )
            classification = concat_classification_criteria(
                classification, f"{_SIDE_LABELS[leftright]}ContraThalamicoCerebellar",
                contra_thal_ceb_bool
            )

    return classification
