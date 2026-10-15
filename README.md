[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)
[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-bl.app.6acfe9b83db9ceda5263760f-blue.svg)](https://brainlife.io/app/6acfe9b83db9ceda5263760f)

# wmaSeg++

`wmaSeg++` extends the white matter anatomy segmentation methodology of [Bullock et al. (2019)](https://doi.org/10.1007/s00429-019-01907-8) — originally implemented in MATLAB via the [wma_tools](https://github.com/DanNBullock/wma_tools) toolbox — with two major advances:

1. **Full Python port** — no MATLAB runtime or license required. All ROI construction, planar criteria, KDTree intersection, and classification logic reimplemented in pure Python/NumPy/SciPy.
2. **Accelerated computational architecture** — redesigned for speed:
   - Batched KDTree queries (all streamline nodes in one call instead of one call per streamline)
   - Cached label inflation (expensive voxel-level propagation runs once, reused across all bundle modules)
   - Vectorized endpoint lookups (direct array indexing replaces per-coordinate Python loops)
   - Parallelized bundle segmentation (11 independent tract groups run concurrently via `multiprocessing.Pool` with Linux copy-on-write fork — no serialization of large arrays)

### Authors
- Gabriele Amorosino (gabriele.amorosino@utexas.edu)

### Citations
We kindly ask that you cite the following articles when publishing papers and code using this code.

1. Bullock, D., Takemura, H., Caiafa, C. F., Kitchell, L., McPherson, B., Caron, B., & Pestilli, F. (2019). Associative white matter connecting the dorsal and ventral posterior human cortex. *Brain Structure and Function*, 224(8), 2631–2660. [https://doi.org/10.1007/s00429-019-01907-8](https://doi.org/10.1007/s00429-019-01907-8)

2. Avesani, P., McPherson, B., Hayashi, S. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Sci Data* 6, 69 (2019). [https://doi.org/10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

#### MIT Copyright (c) 2024 brainlife.io The University of Texas at Austin

## Running the App

### On Brainlife.io

You can submit this App online at [https://brainlife.io/app/6acfe9b83db9ceda5263760f](https://brainlife.io/app/6acfe9b83db9ceda5263760f) via the "Execute" tab.

### Running Locally (on your machine)

**Option 1 — CLI launcher** (generates `config.json` and runs the pipeline):

```bash
./main_cli.sh --track /path/to/track.tck --freesurfer /path/to/freesurfer/
```

**Option 2 — manual config**:

1. git clone this repo.
2. Create `config.json` with paths to your input files:

```json
{
    "track":      "/path/to/track.tck",
    "freesurfer": "/path/to/freesurfer/"
}
```

3. Launch the App:

```bash
export FREESURFER_LICENSE="<your license key>"
./main
```

### Sample Datasets

If you don't have your own input file, you can download sample datasets from Brainlife.io, or you can use [Brainlife CLI](https://github.com/brain-life/cli).

```
npm install -g brainlife
bl login
mkdir input
bl dataset download <dataset-id> && mv <dataset-id> input/track
bl dataset download <dataset-id> && mv <dataset-id> input/freesurfer
```

## Inputs

| Input | Datatype | Description |
|-------|----------|-------------|
| `track` | `neuro/track/tck` | Whole-brain tractogram (TCK format) |
| `freesurfer` | `neuro/freesurfer` | FreeSurfer output directory (requires `aparc.a2009s+aseg`) |

## Output

All output files will be generated under the current working directory. The main output of this App is a `classification/` directory containing:

- `classification.mat` — classification structure (streamline-to-tract assignment)
- `tracts/` — one JSON file per named tract (up to 1000 streamlines each) plus a `tracts.json` index

Segmented tracts include: Arcuate, SLF 1/2/3, Uncinate, IFOF, Cingulum, Forceps Minor/Major, Corpus Callosum subdivisions, Corticospinal Tract, Optic Radiation (Meyer/Baum loops), pArc, TPC, ILF, MDLFang, Aslant, VOF, thalamic projections, and cerebellar tracts.

| Output | Datatype | Description |
|--------|----------|-------------|
| `classification` | `neuro/wmc` | Classification structure: `.mat` file, per-tract JSON files, VTK surfaces |

### Dependencies

This App only requires [Singularity](https://www.sylabs.io/singularity/) to run. Python dependencies are bundled in the container:

- [nibabel](https://nipy.org/nibabel/) ≥ 2.3
- [numpy](https://numpy.org/) ≥ 1.11
- [scipy](https://scipy.org/) ≥ 0.18

#### MIT Copyright (c) 2024 brainlife.io The University of Texas at Austin
