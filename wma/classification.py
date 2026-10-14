"""Classification structure operations."""

import numpy as np


def _empty_classification(n_streamlines):
    return {"names": [], "index": np.zeros(n_streamlines, dtype=np.int32)}


# ---------------------------------------------------------------------------
# bsc_concatClassificationCriteria
# ---------------------------------------------------------------------------

def concat_classification_criteria(classification, name, *boolean_vecs):
    """AND all boolean vectors, add the resulting tract to classification.

    Mirrors bsc_concatClassificationCriteria.
    """
    n = len(classification["index"])
    combined = np.ones(n, dtype=bool)
    for bv in boolean_vecs:
        bv = np.asarray(bv, dtype=bool).ravel()
        if len(bv) != n:
            raise ValueError(f"Boolean vector length {len(bv)} != {n}")
        combined &= bv

    new_cls = _empty_classification(n)
    new_cls["names"] = [name]
    new_cls["index"][combined] = 1

    return reconcile_classifications(classification, new_cls)


# ---------------------------------------------------------------------------
# bsc_reconcileClassifications
# ---------------------------------------------------------------------------

def reconcile_classifications(base, add):
    """Merge two classification structures.

    Mirrors bsc_reconcileClassifications.
    """
    base_names = list(base["names"])
    add_names = list(add["names"])
    base_n = len(base_names)
    add_n = len(add_names)

    base_idx = base["index"].copy()
    add_idx = add["index"].copy()

    n = len(base_idx)
    all_names = base_names + [n for n in add_names if n not in base_names]
    unique_names = list(dict.fromkeys(base_names + add_names))

    out = _empty_classification(n)

    if len(unique_names) == base_n + add_n:
        # All names are unique: simple concat + offset
        out["names"] = base_names + add_names
        out["index"] = base_idx.copy()
        shifted_add = add_idx.copy()
        shifted_add[shifted_add > 0] += base_n
        out["index"][shifted_add > 0] = shifted_add[shifted_add > 0]
    else:
        # Some names overlap: unify
        out["names"] = unique_names
        for i, name in enumerate(unique_names, start=1):
            # indices in base
            if name in base_names:
                b_idx = base_names.index(name) + 1
                out["index"][base_idx == b_idx] = i
            # indices in add
            if name in add_names:
                a_idx = add_names.index(name) + 1
                out["index"][add_idx == a_idx] = i

    return out


# ---------------------------------------------------------------------------
# bsc_extractStreamIndByName
# ---------------------------------------------------------------------------

def extract_stream_ind_by_name(classification, tract_name):
    """Return boolean (N,) for streamlines classified as tract_name."""
    names = classification["names"]
    if tract_name not in names:
        return np.zeros(len(classification["index"]), dtype=bool)
    idx = names.index(tract_name) + 1
    return (classification["index"] == idx)


# ---------------------------------------------------------------------------
# wma_classificationStrucGrouping
# ---------------------------------------------------------------------------

_SIDE_PREFIXES = ["left", "left ", "l_", "l ", "right", "right ", "r_", "r "]


def _strip_side(name):
    lower = name.lower()
    for pfx in _SIDE_PREFIXES:
        if lower.startswith(pfx):
            return name[len(pfx):]
    return name


def classification_struc_grouping(classification):
    """Group left/right tract pairs, returning a merged classification."""
    names = classification["names"]
    idx = classification["index"]
    n = len(idx)

    truncated = [_strip_side(n) for n in names]
    unique_truncated = list(dict.fromkeys(truncated))

    grouping = _empty_classification(n)
    grouping["names"] = unique_truncated

    for gi, stem in enumerate(unique_truncated, start=1):
        both_pos = [i + 1 for i, t in enumerate(truncated) if t == stem]
        for pos in both_pos:
            grouping["index"][idx == pos] = gi

    return grouping


# ---------------------------------------------------------------------------
# wma_resortClassificationStruc
# ---------------------------------------------------------------------------

def resort_classification(classification):
    """Re-order tracts: singletons first, then left/right pairs alphabetically.

    Mirrors wma_resortClassificationStruc.
    """
    grouping = classification_struc_grouping(classification)
    names = classification["names"]
    idx = classification["index"]

    out_names = []
    out_index = np.zeros(len(idx), dtype=np.int32)

    group_counts = {}
    exclude_singleton = set()
    for gi, stem in enumerate(grouping["names"], start=1):
        members = [n for i, n in enumerate(names) if _strip_side(n) == stem]
        group_counts[gi] = len(members)
        if stem not in names:
            exclude_singleton.add(gi)

    singletons = [gi for gi, c in group_counts.items()
                  if c == 1 and gi not in exclude_singleton]
    pairs = [gi for gi in range(1, len(grouping["names"]) + 1)
             if gi not in singletons]
    order = singletons + pairs

    for gi in order:
        stem = grouping["names"][gi - 1]
        members = sorted([n for n in names if _strip_side(n) == stem])
        for member in members:
            out_names.append(member)
            orig_pos = names.index(member) + 1
            out_index[idx == orig_pos] = len(out_names)

    return {"names": out_names, "index": out_index}


# ---------------------------------------------------------------------------
# bsc_makeFGsFromClassification_v4
# ---------------------------------------------------------------------------

def make_fgs_from_classification(classification, streamlines):
    """Build a list of tract dicts, one per named tract.

    Each dict has: 'name', 'colorRgb', 'fibers' (list of (3,N) arrays).
    """
    names = classification["names"]
    idx = classification["index"]
    n_tracts = len(names)
    rng = np.random.default_rng(0)
    colors = rng.random((n_tracts, 3))

    fgs = []
    for i, name in enumerate(names, start=1):
        mask = idx == i
        fg = {
            "name": name,
            "colorRgb": colors[i - 1],
            "fibers": [streamlines[j] for j in np.where(mask)[0]],
        }
        fgs.append(fg)
        print(f"creating tract {i} ({name}) with {mask.sum()} streamlines")

    return fgs
