#!/usr/bin/env python3
"""White matter segmentation pipeline — Python port of main.m."""

import json
import multiprocessing as mp
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

# ---------------------------------------------------------------------------
# Module-level globals for multiprocessing (set before Pool creation; workers
# inherit them via fork — no pickling of large data structures).
# ---------------------------------------------------------------------------
_g_streamlines = None
_g_atlas = None
_g_category_prior = None
_g_eff_prior = None

_STEP_NAMES = [
    "segment_ant_post_tracts",
    "segment_corpus_callosum",
    "segment_subcortical",
    "segment_aslant",
    "segment_mdlf_ilf",
    "segment_arc_tpc",
    "segment_optic_radiation",
    "segment_cerebellar_tracts",
    "segment_vof",
    "segment_cst",
    "segment_cingulum",
]


def _seg_dispatch(task_id):
    """Run one segmentation function; large data inherited via fork."""
    s, a, cp, ep = _g_streamlines, _g_atlas, _g_category_prior, _g_eff_prior
    if task_id == 0:
        return segment_ant_post_tracts(s, a, cp, ep)
    if task_id == 1:
        return segment_corpus_callosum(s, a, experimental_bool=True, category_prior=cp)
    if task_id == 2:
        return segment_subcortical(s, a, cp, ep)
    if task_id == 3:
        return segment_aslant(s, a, cp)
    if task_id == 4:
        return segment_mdlf_ilf(s, a, cp)
    if task_id == 5:
        return segment_arc_tpc(s, a)
    if task_id == 6:
        return segment_optic_radiation(s, a, expand_seg_bool=False, category_prior=cp)
    if task_id == 7:
        return segment_cerebellar_tracts(s, a, experimental_bool=False, category_prior=cp)
    if task_id == 8:
        return segment_vof(s, a, cp)
    if task_id == 9:
        return segment_cst(s, a, cp)
    if task_id == 10:
        return segment_cingulum(s, a, cp)
    raise ValueError(f"Unknown task_id: {task_id}")


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

    # Set module globals before forking so workers inherit the data copy-on-write.
    global _g_streamlines, _g_atlas, _g_category_prior, _g_eff_prior
    _g_streamlines = streamlines
    _g_atlas = atlas
    _g_category_prior = category_prior
    _g_eff_prior = eff_prior

    n_workers = min(11, os.cpu_count() or 1)

    if n_workers > 1:
        print(f"Running {len(_STEP_NAMES)} segmentation steps in parallel ({n_workers} workers)...")
        with mp.Pool(processes=n_workers) as pool:
            seg_results = pool.map(_seg_dispatch, range(len(_STEP_NAMES)))
    else:
        print(f"Running {len(_STEP_NAMES)} segmentation steps sequentially...")
        seg_results = [_seg_dispatch(i) for i in range(len(_STEP_NAMES))]

    for i, out in enumerate(seg_results):
        print(f"Reconciling: {_STEP_NAMES[i]}")
        classification = reconcile_classifications(classification, out)

    print("resort_classification...")
    classification = resort_classification(classification)

    print("Generating output...")
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
