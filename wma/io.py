"""Tractography and atlas I/O."""

import json
import numpy as np
import nibabel as nib


def load_tractogram(tck_file):
    """Load a TCK/TRK file and return streamlines as a list of (3, N) float arrays."""
    tf = nib.streamlines.load(tck_file)
    return [s.T.astype(np.float64) for s in tf.streamlines]


def load_atlas(nifti_path):
    """Load a parcellation NIfTI and return an atlas dict.

    Keys:
        data    – (X, Y, Z) int32 array of label values
        qto_ijk – (4, 4) affine: mm RAS → voxel
        qto_xyz – (4, 4) affine: voxel → mm RAS
        img     – the nibabel image object
    """
    img = nib.load(nifti_path)
    data = np.round(np.asarray(img.dataobj)).astype(np.int32)
    qto_xyz = img.header.get_qform()
    qto_ijk = np.linalg.inv(qto_xyz)
    return {"data": data, "qto_ijk": qto_ijk, "qto_xyz": qto_xyz, "img": img}


def save_classification_mat(classification, path):
    """Save a classification dict to a .mat file (scipy-format)."""
    import scipy.io
    mat = {
        "classification": {
            "names": np.array(classification["names"], dtype=object),
            "index": classification["index"].astype(np.uint8),
        }
    }
    scipy.io.savemat(path, mat)


def save_tract_tck(fibers_3xN, path):
    """Save a list of (3, N) fiber arrays to a TCK file (RAS+ mm space)."""
    streamlines = [f.T.astype(np.float32) for f in fibers_3xN]
    tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(nib.streamlines.TckFile(tractogram), path)


def save_classification_json(classification, path):
    """Save classification as JSON (alternative to .mat)."""
    with open(path, "w") as f:
        json.dump(
            {
                "names": classification["names"],
                "index": classification["index"].tolist(),
            },
            f,
        )
