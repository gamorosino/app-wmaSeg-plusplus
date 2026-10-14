"""Streamline operations: reorient, endpoint/midpoint criteria, ROI intersection."""

import numpy as np
from scipy.spatial import cKDTree

from .atlas import _xform_coords, roi_from_atlas_nums, atlas_roi_nums_from_coords, inflate_labels


# ---------------------------------------------------------------------------
# Streamline reorientation (bsc_reorientStreamline / bsc_reorientFiber)
# ---------------------------------------------------------------------------

def reorient_streamline(fiber):
    """Flip a (3, N) streamline so the larger-coordinate endpoint comes first."""
    disp = np.abs(fiber[:, 0] - fiber[:, -1])
    ax = int(np.argmax(disp))
    if fiber[ax, 0] < fiber[ax, -1]:
        return fiber[:, ::-1]
    return fiber


def reorient_fibers(streamlines):
    """Apply reorient_streamline to every streamline in a list."""
    return [reorient_streamline(s) for s in streamlines]


# ---------------------------------------------------------------------------
# Endpoint / midpoint extraction helpers
# ---------------------------------------------------------------------------

def _endpoints(streamlines):
    """Return (N,3) arrays of first and last endpoints."""
    n = len(streamlines)
    ep1 = np.zeros((n, 3), dtype=np.float64)
    ep2 = np.zeros((n, 3), dtype=np.float64)
    for i, s in enumerate(streamlines):
        ep1[i] = s[:, 0]
        ep2[i] = s[:, -1]
    return ep1, ep2


def _midpoints(streamlines):
    """Return (N,3) array of midpoints."""
    n = len(streamlines)
    mids = np.zeros((n, 3), dtype=np.float64)
    for i, s in enumerate(streamlines):
        mid_idx = s.shape[1] // 2
        if mid_idx == 0:
            mid_idx = 0
        mids[i] = s[:, mid_idx]
    return mids


# ---------------------------------------------------------------------------
# Endpoint criteria (bsc_applyEndpointCriteria)
# ---------------------------------------------------------------------------

def apply_endpoint_criteria(streamlines, *args):
    """Filter streamlines by endpoint spatial criteria.

    Args are triplets: (ref_coord_or_plane, direction_str, 'both'|'one'|'neither'|'either')
    direction_str: 'superior', 'inferior', 'anterior', 'posterior', 'medial', 'lateral'
    ref_coord_or_plane: (3,) array or ROI dict (mean of coords used)

    Returns boolean (N,) array.
    """
    ep1, ep2 = _endpoints(streamlines)
    result = np.ones(len(streamlines), dtype=bool)

    n_triplets = len(args) // 3
    for t in range(n_triplets):
        ref_raw = args[t * 3]
        direction = args[t * 3 + 1].lower()
        side = args[t * 3 + 2].lower()

        if isinstance(ref_raw, dict):
            c = ref_raw["coords"]
            ref = np.array([c[:, 0].mean(), c[:, 1].mean(), c[:, 2].mean()])
        else:
            ref = np.asarray(ref_raw, dtype=float).ravel()

        lr_flag = ref[0] < 0

        if direction in ("top", "superior"):
            dim, valence = 2, 1
        elif direction in ("bottom", "inferior"):
            dim, valence = 2, -1
        elif direction == "anterior":
            dim, valence = 1, 1
        elif direction == "posterior":
            dim, valence = 1, -1
        elif direction == "medial":
            dim = 0
            valence = 1 if lr_flag else -1
        elif direction == "lateral":
            dim = 0
            valence = -1 if lr_flag else 1
        else:
            raise ValueError(f"Unknown direction: {direction}")

        e1_vals = ep1[:, dim]
        e2_vals = ep2[:, dim]
        ref_val = float(ref[dim])

        if valence == 1:
            meets = np.column_stack([e1_vals > ref_val, e2_vals > ref_val])
        else:
            meets = np.column_stack([e1_vals < ref_val, e2_vals < ref_val])

        n_meet = meets.sum(axis=1)
        if side == "both":
            cur = n_meet == 2
        elif side == "one":
            cur = n_meet == 1
        elif side == "neither":
            cur = n_meet == 0
        elif side == "either":
            cur = n_meet >= 1
        else:
            raise ValueError(f"Unknown side: {side}")

        result &= cur

    return result


# ---------------------------------------------------------------------------
# Midpoint criteria (bsc_applyMidpointCriteria)
# ---------------------------------------------------------------------------

def apply_midpoint_criteria(streamlines, *args):
    """Filter streamlines by midpoint spatial criteria.

    Args are pairs: (ref_coord_or_plane, direction_str)
    Returns boolean (N,) array.
    """
    mids = _midpoints(streamlines)
    result = None

    n_pairs = len(args) // 2
    for p in range(n_pairs):
        ref_raw = args[p * 2]
        direction = args[p * 2 + 1].lower()

        if isinstance(ref_raw, dict):
            c = ref_raw["coords"]
            ref = np.array([c[:, 0].mean(), c[:, 1].mean(), c[:, 2].mean()])
        else:
            ref = np.asarray(ref_raw, dtype=float).ravel()

        lr_flag = ref[0] < 0
        ref_val: float

        if direction in ("top", "superior"):
            ref_val = float(ref[2])
            cur = mids[:, 2] > ref_val
        elif direction in ("bottom", "inferior"):
            ref_val = float(ref[2])
            cur = mids[:, 2] < ref_val
        elif direction == "anterior":
            ref_val = float(ref[1])
            cur = mids[:, 1] > ref_val
        elif direction == "posterior":
            ref_val = float(ref[1])
            cur = mids[:, 1] < ref_val
        elif direction == "medial":
            ref_val = float(ref[0])
            cur = mids[:, 0] > ref_val if lr_flag else mids[:, 0] < ref_val
        elif direction == "lateral":
            ref_val = float(ref[0])
            cur = mids[:, 0] < ref_val if lr_flag else mids[:, 0] > ref_val
        else:
            raise ValueError(f"Unknown direction: {direction}")

        result = cur if result is None else result & cur

    return result if result is not None else np.ones(len(streamlines), dtype=bool)


# ---------------------------------------------------------------------------
# ROI-streamline intersection distance (wma_SegmentFascicleFromConnectome)
# ---------------------------------------------------------------------------

def _node_spacing(streamlines):
    """Estimate typical node spacing from the first streamline."""
    if not streamlines or streamlines[0].shape[1] < 2:
        return 1.0
    s = streamlines[0]
    diffs = np.diff(s, axis=1)
    return float(np.mean(np.sqrt((diffs ** 2).sum(axis=0))))


def _intersects_roi(stream, tree, distance, endpoint_only=False):
    """Check if a single streamline intersects an ROI KD-tree within distance."""
    if endpoint_only:
        nodes = np.array([stream[:, 0], stream[:, -1]])  # (2, 3)
    else:
        nodes = stream.T  # (N, 3)
    dists, _ = tree.query(nodes, k=1)
    return bool(np.any(dists <= distance))


def segment_fascicle_from_connectome(streamlines, rois, operations):
    """Apply AND/NOT/endpoints operations sequentially to streamlines.

    Parameters
    ----------
    streamlines : list of (3, N) float arrays
    rois        : list of ROI dicts  (key 'coords' : (M, 3))
    operations  : list of str  ('and', 'not', 'endpoints', 'both_endpoints',
                                 'verticalTraverse', 'spl', 'postNot', 'exclusions')
                  Non-standard names are treated as 'and'.

    Returns
    -------
    keep_bool : (len(streamlines),) bool array
    """
    spacing = _node_spacing(streamlines)
    dist = 0.87 if spacing <= 1.2 else spacing * 0.6

    current_idx = np.arange(len(streamlines))

    for roi, op in zip(rois, operations):
        op_norm = op.lower()
        if len(current_idx) == 0:
            break

        roi_coords = roi["coords"] if isinstance(roi, dict) else roi
        if len(roi_coords) == 0:
            if op_norm not in ("not",):
                current_idx = np.array([], dtype=int)
            continue

        tree = cKDTree(roi_coords)
        endpoint_only = op_norm in ("endpoints", "both_endpoints")

        hits = np.array([
            _intersects_roi(streamlines[i], tree, dist, endpoint_only)
            for i in current_idx
        ])

        if op_norm == "not":
            current_idx = current_idx[~hits]
        else:
            current_idx = current_idx[hits]

    keep_bool = np.zeros(len(streamlines), dtype=bool)
    if len(current_idx) > 0:
        keep_bool[current_idx] = True
    return keep_bool


# ---------------------------------------------------------------------------
# Endpoint-ROI segmentation (bsc_tractByEndpointROIs)
# ---------------------------------------------------------------------------

def tract_by_endpoint_rois(streamlines, roi1, roi2, min_dist=0.87):
    """Keep streamlines with one endpoint near roi1 and the other near roi2."""
    if len(roi1["coords"]) == 0 or len(roi2["coords"]) == 0:
        return np.zeros(len(streamlines), dtype=bool)

    ep1, ep2 = _endpoints(streamlines)
    tree1 = cKDTree(roi1["coords"])
    tree2 = cKDTree(roi2["coords"])

    d1e1, _ = tree1.query(ep1, k=1)
    d1e2, _ = tree1.query(ep2, k=1)
    d2e1, _ = tree2.query(ep1, k=1)
    d2e2, _ = tree2.query(ep2, k=1)

    keep = (
        ((d1e1 < min_dist) & (d2e2 < min_dist)) |
        ((d2e1 < min_dist) & (d1e2 < min_dist))
    )
    return keep


# ---------------------------------------------------------------------------
# Midpoint-ROI segment (bsc_midpointROISegment)
# ---------------------------------------------------------------------------

def midpoint_roi_segment(streamlines, roi, dist_threshold=0.5):
    """Return bool array: True where midpoint is within dist_threshold of roi."""
    mids = _midpoints(streamlines)
    if len(roi["coords"]) == 0:
        return np.zeros(len(streamlines), dtype=bool)
    tree = cKDTree(roi["coords"])
    dists, _ = tree.query(mids, k=1)
    return dists <= dist_threshold


# ---------------------------------------------------------------------------
# Endpoint atlas label criteria (bsc_endpointAtlasCriteria)
# ---------------------------------------------------------------------------

def endpoint_atlas_criteria(streamlines, atlas, roi_nums, criteria="either"):
    """Keep streamlines where endpoints land in specified atlas ROI IDs.

    criteria : 'one', 'both', 'neither', 'either'
    """
    inflated = inflate_labels(atlas, 2)
    ep1, ep2 = _endpoints(streamlines)
    zone1 = atlas_roi_nums_from_coords(inflated, ep1.T)  # (N,)
    zone2 = atlas_roi_nums_from_coords(inflated, ep2.T)  # (N,)

    roi_set = frozenset(np.atleast_1d(roi_nums).ravel().tolist())
    in1 = np.array([z in roi_set for z in zone1], dtype=bool)
    in2 = np.array([z in roi_set for z in zone2], dtype=bool)
    n_meet = in1.astype(int) + in2.astype(int)

    if criteria == "one":
        return n_meet == 1
    elif criteria == "both":
        return n_meet == 2
    elif criteria == "neither":
        return n_meet == 0
    elif criteria == "either":
        return n_meet >= 1
    raise ValueError(f"Unknown criteria: {criteria}")


# ---------------------------------------------------------------------------
# Streamline length
# ---------------------------------------------------------------------------

def streamline_lengths(streamlines):
    """Return (N,) array of arc-length for each streamline."""
    lengths = np.zeros(len(streamlines), dtype=np.float64)
    for i, s in enumerate(streamlines):
        diffs = np.diff(s, axis=1)
        lengths[i] = float(np.sqrt((diffs ** 2).sum(axis=0)).sum())
    return lengths
