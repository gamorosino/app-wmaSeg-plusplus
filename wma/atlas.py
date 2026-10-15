"""Atlas operations: deisland, inflate, ROI creation, planar ROIs."""

import numpy as np
from scipy.ndimage import label as ndi_label, uniform_filter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _xform_coords(xform, coords):
    """Apply 4×4 affine to coordinates.

    coords may be (3, N) or (N, 3).  Always returns (N, 3).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]
    if coords.shape[0] == 3 and (coords.ndim == 2 and coords.shape[1] != 3):
        coords = coords.T
    # (N, 3)
    return (xform[:3, :3] @ coords.T).T + xform[:3, 3]


def _apply_affine(aff, voxel_coords):
    """Transform 0-indexed integer voxel coords (N, 3) to mm space. Returns (N, 3)."""
    return _xform_coords(aff, voxel_coords)


def _new_roi(name, coords):
    """Create an ROI dict matching vistasoft ROI structure."""
    return {"name": name, "coords": np.asarray(coords, dtype=np.float64)}


# ---------------------------------------------------------------------------
# Connected-component filtering (bwareaopen equivalent)
# ---------------------------------------------------------------------------

def deisland_labels(atlas, max_isle_size=5, replace_val=0):
    """Remove small connected components (islands) from each label.

    Mirrors fnDeislandLabels_v3.

    Parameters
    ----------
    atlas : dict  (keys: data, qto_ijk, qto_xyz)
    max_isle_size : int  voxels; components smaller than this are replaced
    replace_val : int   value assigned to removed island voxels

    Returns
    -------
    dict  – copy of atlas with updated data array
    """
    data = atlas["data"].copy()
    out_data = np.zeros_like(data)
    unique_labels = np.unique(data)
    unique_labels = unique_labels[unique_labels > 0]

    n_lost = 0
    for lbl in unique_labels:
        mask = data == lbl
        labeled_arr, _ = ndi_label(mask)
        comp_sizes = np.bincount(labeled_arr.ravel())
        # keep components >= max_isle_size; islands are those < max_isle_size
        island_mask = np.zeros_like(mask)
        for comp_id in range(1, len(comp_sizes)):
            if comp_sizes[comp_id] < max_isle_size:
                island_mask |= labeled_arr == comp_id
        connected_mask = mask & ~island_mask
        out_data[connected_mask] = lbl
        out_data[mask & island_mask] = replace_val
        n_lost += int(island_mask.sum())

    print(f"{n_lost} total island voxels removed")
    return {**atlas, "data": out_data}


# ---------------------------------------------------------------------------
# Label inflation (bsc_inflateLabels)
# ---------------------------------------------------------------------------

_GREY_MATTER_ROIS = (
    list(range(101 + 11000, 175 + 11000 + 1))
    + list(range(101 + 12000, 175 + 12000 + 1))
)
_SUBCORTICAL_ROIS = [10, 11, 12, 13, 17, 18, 19, 20, 26, 27, 28,
                     49, 50, 51, 52, 53, 54, 55, 56, 58, 59]
_WM_ROIS = [41, 2]
_INFLATE_TARG_ROIS_BASE = _WM_ROIS + [2000, 1000, 999]

_INFLATE_SOURCE_SET = frozenset(_GREY_MATTER_ROIS + _SUBCORTICAL_ROIS)
_INFLATE_TARG_SET_BASE = frozenset(_INFLATE_TARG_ROIS_BASE)

_inflate_cache: dict = {}


def _assign_neighbor_label(data, x, y, z, targ_set, radius=1):
    """Pick the most-common non-target label in a (2r+1)^3 cube."""
    shape = data.shape
    for r in range(radius, 6):
        x0, x1 = max(0, x - r), min(shape[0], x + r + 1)
        y0, y1 = max(0, y - r), min(shape[1], y + r + 1)
        z0, z1 = max(0, z - r), min(shape[2], z + r + 1)
        cube = data[x0:x1, y0:y1, z0:z1].ravel()
        clean = cube[~np.isin(cube, list(targ_set))]
        if len(clean) == 0:
            continue
        vals, counts = np.unique(clean, return_counts=True)
        winners = vals[counts == counts.max()]
        if len(winners) == 1:
            return int(winners[0])
    return None


def inflate_labels(atlas, inflate_itr):
    """Iteratively inflate grey matter labels into adjacent WM/unknown voxels.

    Mirrors bsc_inflateLabels. Results are cached by atlas data identity so
    repeated calls with the same atlas are free.
    """
    key = (id(atlas["data"]), inflate_itr)
    if key in _inflate_cache:
        return _inflate_cache[key]
    result = _inflate_labels_impl(atlas, inflate_itr)
    _inflate_cache[key] = result
    return result


def _inflate_labels_impl(atlas, inflate_itr):
    # First deisland with replace_val=999
    atlas_iter = deisland_labels(atlas, max_isle_size=5, replace_val=999)
    data = atlas_iter["data"]

    targ_set = _INFLATE_TARG_SET_BASE

    THRESHOLD_VAL = 6.0 / 27.0

    for iIter in range(inflate_itr):
        data_cur = data.copy()
        data_next = data.copy()

        # Build binary mask of source (inflating) labels
        inflate_mask = np.isin(data_cur, list(_INFLATE_SOURCE_SET)).astype(float)
        # Smooth to find voxels adjacent to source labels
        smoothed = uniform_filter(inflate_mask, size=3)
        # Target voxels that are adjacent to source labels
        targ_mask = np.isin(data_cur, list(targ_set))
        border_mask = (smoothed >= THRESHOLD_VAL) & targ_mask

        shape = data_cur.shape
        coords = np.argwhere(border_mask)
        # Remove edge voxels
        edge = (
            (coords[:, 0] == 0) | (coords[:, 0] == shape[0] - 1) |
            (coords[:, 1] == 0) | (coords[:, 1] == shape[1] - 1) |
            (coords[:, 2] == 0) | (coords[:, 2] == shape[2] - 1)
        )
        coords = coords[~edge]

        print(f"(inflate_labels) {len(coords)} voxels for iteration {iIter + 1}")
        for c in coords:
            lbl = _assign_neighbor_label(data_cur, c[0], c[1], c[2], targ_set)
            if lbl is not None:
                data_next[c[0], c[1], c[2]] = lbl

        data = data_next

    # Final pass: replace any remaining 999s
    nines_coords = np.argwhere(data == 999)
    for c in nines_coords:
        lbl = _assign_neighbor_label(data, c[0], c[1], c[2], targ_set)
        if lbl is not None:
            data[c[0], c[1], c[2]] = lbl

    print("atlas inflation complete")
    return {**atlas, "data": data}


# ---------------------------------------------------------------------------
# Atlas ROI label lookup (bsc_atlasROINumsFromCoords_v3)
# ---------------------------------------------------------------------------

def atlas_roi_nums_from_coords(atlas, coords_acpc):
    """Return atlas label at each ACPC coordinate.

    coords_acpc : (3, N) or (N, 3) array in mm ACPC space.
    Returns     : (N,) integer array of atlas label IDs.
    """
    coords = np.asarray(coords_acpc, dtype=np.float64)
    if coords.ndim == 2 and coords.shape[0] == 3 and coords.shape[1] != 3:
        coords = coords.T  # → (N, 3)

    qto_ijk = atlas["qto_ijk"]
    data = atlas["data"]
    shape = np.array(data.shape)

    img_coords = np.floor(_xform_coords(qto_ijk, coords)).astype(int)
    img_coords = np.clip(img_coords, 0, shape - 1)

    # Vectorized direct lookup
    roi_nums = data[img_coords[:, 0], img_coords[:, 1], img_coords[:, 2]].astype(np.int32)

    # Fallback: neighborhood expansion only for zero-labeled voxels
    zero_mask = roi_nums == 0
    if zero_mask.any():
        for i in np.where(zero_mask)[0]:
            x, y, z = img_coords[i]
            for r in range(1, 5):
                x0, x1 = max(0, x - r), min(shape[0], x + r + 1)
                y0, y1 = max(0, y - r), min(shape[1], y + r + 1)
                z0, z1 = max(0, z - r), min(shape[2], z + r + 1)
                cube = data[x0:x1, y0:y1, z0:z1].ravel()
                nonzero = cube[cube > 0]
                if len(nonzero) > 0:
                    vals, counts = np.unique(nonzero, return_counts=True)
                    roi_nums[i] = vals[counts.argmax()]
                    break

    return roi_nums


# ---------------------------------------------------------------------------
# ROI creation from atlas label numbers (bsc_roiFromAtlasNums)
# ---------------------------------------------------------------------------

def roi_from_atlas_nums(atlas, roi_nums, smooth_kernel=0):
    """Create an ROI (dict with 'name' and 'coords') from atlas label IDs.

    smooth_kernel : int  box-filter half-width for inflation; 0 = no inflation.
    """
    roi_nums = np.atleast_1d(roi_nums).ravel()
    data = atlas["data"]
    roi_mask = np.zeros(data.shape, dtype=bool)
    for n in roi_nums:
        roi_mask |= data == n

    if smooth_kernel > 0:
        # Mirror MATLAB: smooth3(mask,'box',kernel)==0 → ~(smooth3==0)
        kernel_size = int(smooth_kernel)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = uniform_filter(roi_mask.astype(float), size=kernel_size)
        roi_mask = smoothed > 0

    vox_coords = np.argwhere(roi_mask)  # (N, 3) 0-indexed
    name_str = "_".join(str(n) for n in roi_nums)

    if len(vox_coords) == 0:
        return _new_roi(name_str, np.zeros((0, 3), dtype=np.float64))

    acpc_coords = _apply_affine(atlas["qto_xyz"], vox_coords)
    return _new_roi(name_str, acpc_coords)


# ---------------------------------------------------------------------------
# Planar ROI (bsc_makePlanarROI_v3)
# ---------------------------------------------------------------------------

def make_planar_roi(atlas, mm_plane, dimension):
    """Create a dense 1mm-grid planar ROI in ACPC space.

    dimension : 'x', 'y', or 'z'
    mm_plane  : scalar mm coordinate of the plane
    """
    data = atlas["data"]
    qto_xyz = atlas["qto_xyz"]
    # Bounding box: transform first and last voxel corners
    corners = np.array([[0, 0, 0], [s - 1 for s in data.shape]], dtype=float)
    bb = _apply_affine(qto_xyz, corners)
    bb_min = bb.min(axis=0)
    bb_max = bb.max(axis=0)

    dim_map = {"x": 0, "y": 1, "z": 2}
    d = dim_map[dimension.lower()]

    other_dims = [i for i in range(3) if i != d]
    ranges = [np.arange(bb_min[i], bb_max[i] + 1, 1.0) for i in other_dims]
    g0, g1 = np.meshgrid(ranges[0], ranges[1], indexing="ij")
    n = g0.size
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, d] = mm_plane
    coords[:, other_dims[0]] = g0.ravel()
    coords[:, other_dims[1]] = g1.ravel()

    name = f"{dimension}_{mm_plane:.2f}_plane"
    return _new_roi(name, coords)


# ---------------------------------------------------------------------------
# bsc_makePlanarROI (deprecated, used for interhemi plane at x=0)
# ---------------------------------------------------------------------------

def make_planar_roi_legacy(atlas, mm_plane, dimension):
    """Legacy version – same result as make_planar_roi for our purposes."""
    return make_planar_roi(atlas, mm_plane, dimension)


# ---------------------------------------------------------------------------
# Plane from ROI (bsc_planeFromROI_v2)
# ---------------------------------------------------------------------------

def plane_from_roi(roi_or_nums, location, atlas):
    """Create a planar ROI at the extreme coordinate of an ROI in a direction.

    roi_or_nums : ROI dict or integer / list of integers (atlas label IDs)
    location    : 'superior', 'inferior', 'anterior', 'posterior', 'medial', 'lateral'
    atlas       : atlas dict
    """
    if isinstance(roi_or_nums, (int, np.integer, list, np.ndarray)):
        roi = roi_from_atlas_nums(atlas, roi_or_nums, smooth_kernel=1)
    else:
        roi = roi_or_nums

    if len(roi["coords"]) == 0:
        # Return an empty plane at 0
        return make_planar_roi(atlas, 0.0, "x")

    coords = roi["coords"]
    lr_flag = float(np.mean(coords[:, 0])) < 0  # left hemisphere

    loc = location.lower()
    if loc in ("top", "superior"):
        mm_val = float(coords[:, 2].max())
        return make_planar_roi(atlas, mm_val, "z")
    elif loc in ("bottom", "inferior"):
        mm_val = float(coords[:, 2].min())
        return make_planar_roi(atlas, mm_val, "z")
    elif loc == "anterior":
        mm_val = float(coords[:, 1].max())
        return make_planar_roi(atlas, mm_val, "y")
    elif loc == "posterior":
        mm_val = float(coords[:, 1].min())
        return make_planar_roi(atlas, mm_val, "y")
    elif loc == "medial":
        mm_val = float(coords[:, 0].max()) if lr_flag else float(coords[:, 0].min())
        return make_planar_roi(atlas, mm_val, "x")
    elif loc == "lateral":
        mm_val = float(coords[:, 0].min()) if lr_flag else float(coords[:, 0].max())
        return make_planar_roi(atlas, mm_val, "x")
    else:
        raise ValueError(f"Unknown location: {location}")


# ---------------------------------------------------------------------------
# Modify ROI (bsc_modifyROI_v2)
# ---------------------------------------------------------------------------

def modify_roi(atlas, roi_in, ref_coord, location):
    """Clip an ROI to one side of a reference coordinate.

    ref_coord : ROI dict (its mean is used) or (3,) array
    location  : direction string
    """
    if isinstance(roi_in, (int, np.integer, list, np.ndarray)):
        roi_in = roi_from_atlas_nums(atlas, roi_in, smooth_kernel=0)

    if isinstance(ref_coord, dict):
        rc = ref_coord["coords"]
        ref = np.array([rc[:, 0].mean(), rc[:, 1].mean(), rc[:, 2].mean()])
    else:
        ref = np.asarray(ref_coord, dtype=float).ravel()

    lr_flag = float(ref[0]) < 0
    coords = roi_in["coords"].copy()

    loc = location.lower()
    if loc in ("top", "superior"):
        mask = coords[:, 2] > ref[2]
    elif loc in ("bottom", "inferior"):
        mask = coords[:, 2] < ref[2]
    elif loc == "anterior":
        mask = coords[:, 1] > ref[1]
    elif loc == "posterior":
        mask = coords[:, 1] < ref[1]
    elif loc == "medial":
        mask = coords[:, 0] > ref[0] if lr_flag else coords[:, 0] < ref[0]
    elif loc == "lateral":
        mask = coords[:, 0] < ref[0] if lr_flag else coords[:, 0] > ref[0]
    else:
        raise ValueError(f"Unknown location: {location}")

    new_coords = coords[mask]
    ref_val = ref[{"superior": 2, "inferior": 2, "top": 2, "bottom": 2,
                   "anterior": 1, "posterior": 1, "medial": 0, "lateral": 0}[loc]]
    name = f"{roi_in['name']}_{location}_to_{ref_val:.2f}"
    if len(new_coords) == 0:
        print(f"modify_roi: empty result for {roi_in['name']} {location}")
    return _new_roi(name, new_coords)


# ---------------------------------------------------------------------------
# Set operations on ROIs
# ---------------------------------------------------------------------------

def merge_rois(roi1, roi2):
    """Union of two ROIs (unique rows)."""
    if len(roi1["coords"]) == 0:
        return _new_roi(f"{roi1['name']}_and_{roi2['name']}", roi2["coords"].copy())
    if len(roi2["coords"]) == 0:
        return _new_roi(f"{roi1['name']}_and_{roi2['name']}", roi1["coords"].copy())
    merged = np.unique(np.vstack([roi1["coords"], roi2["coords"]]), axis=0)
    return _new_roi(f"{roi1['name']}_and_{roi2['name']}", merged)


def intersect_rois(roi1, roi2):
    """Intersection of two ROIs."""
    if len(roi1["coords"]) == 0 or len(roi2["coords"]) == 0:
        return _new_roi(f"{roi1['name']}_to_{roi2['name']}", np.zeros((0, 3)))
    # Round to avoid float noise before intersect
    r1 = np.round(roi1["coords"], 2)
    r2 = np.round(roi2["coords"], 2)
    s1 = set(map(tuple, r1))
    s2 = set(map(tuple, r2))
    common = np.array(sorted(s1 & s2)) if s1 & s2 else np.zeros((0, 3))
    return _new_roi(f"{roi1['name']}_to_{roi2['name']}", common)


def multi_intersect_rois(atlas, inflate, *roi_args):
    """Sequential intersection of multiple ROIs or atlas label IDs."""
    first = roi_args[0]
    result = roi_from_atlas_nums(atlas, first, inflate) if isinstance(first, (int, np.integer, list, np.ndarray)) else first
    for item in roi_args[1:]:
        cur = roi_from_atlas_nums(atlas, item, inflate) if isinstance(item, (int, np.integer, list, np.ndarray)) else item
        result = intersect_rois(result, cur)
    return result
