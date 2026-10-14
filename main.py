#!/usr/bin/env python3
"""White matter segmentation pipeline — Python port of main.m."""

import json
import os
import random

import numpy as np

from wma.io import load_tractogram, load_atlas, save_classification_mat, save_classification_json
from wma.atlas import deisland_labels
from wma.priors import streamline_category_priors, streamline_geometry_priors
from wma.classification import (
    reconcile_classifications, resort_classification, make_fgs_from_classification,
    _empty_classification,
)
from wma.segmentations.ant_post import segment_ant_post_tracts
from wma.segmentations.corpus_callosum import segment_corpus_callosum
from wma.segmentations.subcortical import segment_subcortical
from wma.segmentations.aslant import segment_aslant
from wma.segmentations.mdlf_ilf import segment_mdlf_ilf
from wma.segmentations.arc_tpc import segment_arc_tpc
from wma.segmentations.optic_radiation import segment_optic_radiation
from wma.segmentations.cerebellar import segment_cerebellar_tracts
from wma.segmentations.vof import segment_vof
from wma.segmentations.cst import segment_cst
from wma.segmentations.cingulum import segment_cingulum


def main():
    with open("config.json") as f:
        config = json.load(f)

    print("Loading tractogram...")
    streamlines = load_tractogram(config["track"])

    print("Loading atlas...")
    atlas_raw = load_atlas("aparc.a2009s+aseg.nii.gz")

    print("fnDeislandLabels_v3...")
    atlas = deisland_labels(atlas_raw, max_isle_size=5, replace_val=0)

    n = len(streamlines)
    classification = _empty_classification(n)

    print("Creating priors...")
    category_prior = streamline_category_priors(streamlines, atlas, inflate_iter=2)
    _, eff_prior = streamline_geometry_priors(streamlines)

    print("1) segment_ant_post_tracts...")
    out = segment_ant_post_tracts(streamlines, atlas, category_prior, eff_prior)
    classification = reconcile_classifications(classification, out)

    print("2) segment_corpus_callosum...")
    out = segment_corpus_callosum(streamlines, atlas, experimental_bool=True,
                                  category_prior=category_prior)
    classification = reconcile_classifications(classification, out)

    print("3) segment_subcortical...")
    out = segment_subcortical(streamlines, atlas, category_prior, eff_prior)
    classification = reconcile_classifications(classification, out)

    print("4) segment_aslant...")
    out = segment_aslant(streamlines, atlas, category_prior)
    classification = reconcile_classifications(classification, out)

    print("5) segment_mdlf_ilf...")
    out = segment_mdlf_ilf(streamlines, atlas, category_prior)
    classification = reconcile_classifications(classification, out)

    print("6) segment_arc_tpc...")
    out = segment_arc_tpc(streamlines, atlas)
    classification = reconcile_classifications(classification, out)

    print("7) segment_optic_radiation...")
    out = segment_optic_radiation(streamlines, atlas, expand_seg_bool=False,
                                  category_prior=category_prior)
    classification = reconcile_classifications(classification, out)

    print("8) segment_cerebellar_tracts...")
    out = segment_cerebellar_tracts(streamlines, atlas, experimental_bool=False,
                                    category_prior=category_prior)
    classification = reconcile_classifications(classification, out)

    print("9) segment_vof...")
    out = segment_vof(streamlines, atlas, category_prior)
    classification = reconcile_classifications(classification, out)

    print("10) segment_cst...")
    out = segment_cst(streamlines, atlas, category_prior)
    classification = reconcile_classifications(classification, out)

    print("11) segment_cingulum (final pass)...")
    out = segment_cingulum(streamlines, atlas, category_prior)
    classification = reconcile_classifications(classification, out)

    print("12) resort_classification...")
    classification = resort_classification(classification)

    print("13) Generating output...")
    os.makedirs("classification", exist_ok=True)
    save_classification_mat(classification, "classification/classification.mat")

    fg_classified = make_fgs_from_classification(classification, streamlines)

    tracts_path = "classification/tracts"
    os.makedirs(tracts_path, exist_ok=True)

    all_tracts = []
    for it, fg in enumerate(fg_classified):
        tract = {"name": fg["name"], "color": fg.get("color", [200, 200, 200])}
        fiber_count = min(1000, len(fg["fibers"]))
        indices = random.sample(range(len(fg["fibers"])), fiber_count)
        tract["coords"] = [
            np.round(fg["fibers"][i].T, 3).tolist() for i in indices
        ]
        tract_path = os.path.join(tracts_path, f"{it + 1}.json")
        with open(tract_path, "w") as f:
            json.dump(tract, f)
        print(f"Saved tract.json for {fg['name']}")
        all_tracts.append({
            "name": fg["name"],
            "color": fg.get("color", [200, 200, 200]),
            "filename": f"{it + 1}.json",
        })

    with open(os.path.join(tracts_path, "tracts.json"), "w") as f:
        json.dump(all_tracts, f)

    print("All done.")


if __name__ == "__main__":
    main()
