"""Corpus callosum segmentation (bsc_segmentCorpusCallosum_v3)."""

import numpy as np

from ..atlas import roi_from_atlas_nums, plane_from_roi, modify_roi, merge_rois
from ..streamlines import segment_fascicle_from_connectome, apply_endpoint_criteria, apply_midpoint_criteria
from ..classification import (
    _empty_classification, concat_classification_criteria, extract_stream_ind_by_name,
)


def segment_corpus_callosum(streamlines, atlas, experimental_bool, category_prior):
    """Segment forceps minor, forceps major, parietal CC, middle frontal CC."""
    n = len(streamlines)
    classification = _empty_classification(n)

    full_cc_roi = roi_from_atlas_nums(atlas, [255, 254, 253, 252, 251], smooth_kernel=13)
    _, exclude_cc_bool = None, segment_fascicle_from_connectome(
        streamlines, [full_cc_roi], ["endpoints"]
    )
    ventricle_roi = roi_from_atlas_nums(atlas, [4, 43], smooth_kernel=7)
    _, exclude_ventri_bool = None, segment_fascicle_from_connectome(
        streamlines, [ventricle_roi], ["endpoints"]
    )

    lat_put_left = plane_from_roi(12, "lateral", atlas)
    lat_put_right = plane_from_roi(51, "lateral", atlas)

    # ---- Forceps Minor ----
    anterior_cc_roi = roi_from_atlas_nums(atlas, [255], smooth_kernel=0)
    cc_bottom = plane_from_roi(255, "inferior", atlas)
    ant_mid_cc_plane_ant = plane_from_roi(254, "anterior", atlas)
    cc_extrem_ant_plane_ant = plane_from_roi(255, "anterior", atlas)

    peri_c_roi = plane_from_roi([167 + 11000, 167 + 12000], "anterior", atlas)
    front_middle_plane = plane_from_roi([106 + 11000, 106 + 12000], "anterior", atlas)
    front_middle_top_plane = plane_from_roi([154 + 11000, 154 + 12000], "superior", atlas)
    olf_top = plane_from_roi([164 + 11000, 164 + 12000], "superior", atlas)
    cc_ant_plane = plane_from_roi(255, "anterior", atlas)

    forceps_minior_bool = apply_endpoint_criteria(
        streamlines,
        peri_c_roi, "anterior", "both",
        olf_top, "superior", "both",
        front_middle_top_plane, "inferior", "both",
    )

    anterior_cc_bool = segment_fascicle_from_connectome(
        streamlines, [anterior_cc_roi, ant_mid_cc_plane_ant, peri_c_roi], ["and", "not", "and"]
    )
    frontal_cc_bool = extract_stream_ind_by_name(category_prior, "frontal_to_frontal_interHemi")

    forceps_minor_bool = (
        forceps_minior_bool & anterior_cc_bool & ~exclude_cc_bool & ~exclude_ventri_bool & frontal_cc_bool
    )
    classification = concat_classification_criteria(
        classification, "forcepsMinor", forceps_minor_bool
    )

    # ---- Forceps Major ----
    lat_temp_post_limit = plane_from_roi([161 + 11000, 161 + 12000], "posterior", atlas)
    posterior_cc_inf_limit = plane_from_roi(251, "inferior", atlas)
    post_cc_inferior_roi_cut = modify_roi(
        atlas, posterior_cc_inf_limit, lat_temp_post_limit, "anterior"
    )
    cc_post_midpoints = apply_midpoint_criteria(streamlines, lat_temp_post_limit, "anterior")

    posterior_cc_bool = segment_fascicle_from_connectome(
        streamlines, [post_cc_inferior_roi_cut], ["not"]
    )
    posterior_cc_bool2 = extract_stream_ind_by_name(category_prior, "occipital_to_occipital_interHemi")
    forceps_major_bool = cc_post_midpoints & posterior_cc_bool2
    classification = concat_classification_criteria(
        classification, "forcepsMajor", forceps_major_bool
    )

    # ---- Parietal CC ----
    parietal_cc_posterior_limit = plane_from_roi(253, "posterior", atlas)
    parietal_cc_inferior_limit = plane_from_roi(251, "inferior", atlas)
    parietal_cc_min_top_limit = plane_from_roi(252, "superior", atlas)
    parietal_endpoints_bool = apply_endpoint_criteria(
        streamlines, parietal_cc_min_top_limit, "superior", "both"
    )

    parietal_cc_bool = segment_fascicle_from_connectome(
        streamlines,
        [parietal_cc_posterior_limit, parietal_cc_inferior_limit, lat_put_left, lat_put_right],
        ["not", "not", "not", "not"],
    )
    par_interhemi_bool = extract_stream_ind_by_name(category_prior, "parietal_to_parietal_interHemi")
    parietal_cc_bool = (
        par_interhemi_bool & ~forceps_minor_bool & ~forceps_major_bool &
        parietal_cc_bool & parietal_endpoints_bool
    )
    classification = concat_classification_criteria(
        classification, "parietalCC", parietal_cc_bool
    )

    # ---- Middle Frontal CC ----
    fronto_separate = plane_from_roi([118 + 11000, 118 + 12000], "anterior", atlas)
    posterior_fronto_cc_limit = plane_from_roi(255, "posterior", atlas)
    middle_frontal_not = plane_from_roi(254, "inferior", atlas)
    middle_frontal_limit = plane_from_roi(254, "anterior", atlas)
    anterio_frontal_not = modify_roi(atlas, middle_frontal_not, middle_frontal_limit, "posterior")

    anterio_frontal_bool = segment_fascicle_from_connectome(
        streamlines,
        [lat_put_left, lat_put_right, fronto_separate, anterio_frontal_not, posterior_fronto_cc_limit],
        ["not", "not", "and", "not", "not"],
    )
    middle_frontal_bool = segment_fascicle_from_connectome(
        streamlines,
        [lat_put_left, lat_put_right, fronto_separate, middle_frontal_not],
        ["not", "not", "not", "not"],
    )
    middle_frontal_endpoint_bool = apply_endpoint_criteria(
        streamlines, fronto_separate, "anterior", "both"
    )
    middle_frontal_interhemi_bool = extract_stream_ind_by_name(
        category_prior, "frontal_to_frontal_interHemi"
    )

    middle_frontal_bool = (
        middle_frontal_interhemi_bool & ~forceps_minor_bool & ~forceps_major_bool &
        ~parietal_cc_bool & middle_frontal_bool & ~middle_frontal_endpoint_bool
    )
    anterio_frontal_bool = (
        middle_frontal_interhemi_bool & ~forceps_minor_bool & ~forceps_major_bool &
        ~parietal_cc_bool & ~middle_frontal_bool & anterio_frontal_bool & middle_frontal_endpoint_bool
    )

    classification = concat_classification_criteria(
        classification, "middleFrontalCC", middle_frontal_bool
    )
    if experimental_bool:
        classification = concat_classification_criteria(
            classification, "anterioFrontalCC", anterio_frontal_bool
        )

    return classification
