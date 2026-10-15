[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)

# wmaSeg++ (app-wmaSeg-plusplus)

**Accelerated, Python-native white matter tract segmentation for brainlife.io.**

`wmaSeg++` extends the white matter anatomy segmentation methodology of [Bullock et al. (2019)](https://doi.org/10.1007/s00429-019-01907-8) — originally implemented in MATLAB via the [wma_tools](https://github.com/DanNBullock/wma_tools) toolbox — with two major advances:

1. **Full Python port** — no MATLAB runtime or license required. All ROI construction, planar criteria, KDTree intersection, and classification logic reimplemented in pure Python/NumPy/SciPy.
2. **Accelerated computational architecture** — redesigned for speed:
   - Batched KDTree queries (all streamline nodes in one call instead of one call per streamline)
   - Cached label inflation (expensive voxel-level propagation runs once, reused across all bundle modules)
   - Vectorized endpoint lookups (direct array indexing replaces per-coordinate Python loops)
   - Parallelized bundle segmentation (11 independent tract groups run concurrently via `multiprocessing.Pool` with Linux copy-on-write fork — no serialization of large arrays)

## What it does

Given a whole-brain tractogram (TCK) and a FreeSurfer parcellation, this app:

1. Converts the FreeSurfer `aparc.a2009s+aseg` parcellation to NIfTI format.
2. Segments the tractogram into named white matter tracts using anatomical ROI criteria derived from the parcellation atlas.
3. Generates VTK surface files for each tract and the parcellation.

Segmented tracts include: Arcuate, SLF 1/2/3, Uncinate, IFOF, Cingulum, Forceps Minor/Major, Corpus Callosum subdivisions, Corticospinal Tract, Optic Radiation (Meyer/Baum loops), pArc, TPC, ILF, MDLFang, Aslant, VOF, thalamic projections, and cerebellar tracts.

## Inputs

| Input | Datatype | Description |
|-------|----------|-------------|
| `track` | `neuro/track/tck` | Whole-brain tractogram (TCK format) |
| `freesurfer` | `neuro/freesurfer` | FreeSurfer output directory (requires `aparc.a2009s+aseg`) |

## Outputs

| Output | Datatype | Description |
|--------|----------|-------------|
| `classification` | `neuro/wmc` | Classification structure: `.mat` file, per-tract JSON files, VTK surfaces |

## Running on Brainlife.io

This app is available on the [brainlife.io](https://brainlife.io) platform as **White Matter Anatomy Segmentation ++**.

## Running Locally

Requires Singularity, a FreeSurfer license, and Python ≥ 3.6.

```bash
export FREESURFER_LICENSE="<your license key>"
cp config.json.example config.json
# edit config.json to set the actual paths
./main
```

Python dependencies (installed inside the container — no local install needed):

- [nibabel](https://nipy.org/nibabel/) ≥ 2.3
- [numpy](https://numpy.org/) ≥ 1.11
- [scipy](https://scipy.org/) ≥ 0.18

## Authors

- Gabriele Amorosino (gabriele.amorosino@utexas.edu)

## Based on

- **Segmentation methodology**: [wma_tools](https://github.com/DanNBullock/wma_tools) — Daniel Bullock (Indiana University)
- **Original MATLAB app**: [app-wmaSeg](https://github.com/brainlife/app-wmaSeg) (brainlife.io [bl.app.188](https://doi.org/10.25663/brainlife.app.188)) — Daniel Bullock, Soichi Hayashi, Franco Pestilli

## References

Bullock, D., Takemura, H., Caiafa, C. F., Kitchell, L., McPherson, B., Caron, B., & Pestilli, F. (2019). Associative white matter connecting the dorsal and ventral posterior human cortex. *Brain Structure and Function*, 224(8), 2631–2660. https://doi.org/10.1007/s00429-019-01907-8

Avesani, P., McPherson, B., Hayashi, S. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Sci Data* 6, 69 (2019). https://doi.org/10.1038/s41597-019-0073-y
