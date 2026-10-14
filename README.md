[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)
[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-bl.app.6acfe9b83db9ceda5263760f-blue.svg)](https://brainlife.io/app/6acfe9b83db9ceda5263760f)

# app-wmaSeg-python

Automatically segment a tractogram into major white matter tracts.

This app is a **Python port** of [app-wmaSeg](https://github.com/brainlife/app-wmaSeg) (brainlife.io app [bl.app.188](https://doi.org/10.25663/brainlife.app.188)), which was originally implemented using compiled MATLAB code from the [wma_tools](https://github.com/DanNBullock/wma_tools) toolbox by Daniel Bullock. The MATLAB segmentation logic has been refactored and recast entirely in Python — no MATLAB runtime or license required.

## What it does

Given a whole-brain tractogram (TCK) and a FreeSurfer parcellation, this app:

1. Converts the FreeSurfer `aparc.a2009s+aseg` parcellation to NIfTI format.
2. Segments the tractogram into named white matter tracts using anatomical ROI criteria derived from the parcellation atlas.
3. Generates VTK surface files for each tract and the parcellation.

Segmented tracts include: Arcuate, SLF 1/2/3, Uncinate, IFOF, Cingulum, Forceps Minor/Major, Corpus Callosum subdivisions, Corticospinal Tract, Optic Radiation (Meyer/Baum loops), pArc, TPC, ILF, MDLFang, MDLFspl, Aslant, VOF, thalamic projections, and cerebellar tracts.

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

Visit https://brainlife.io/app/6acfe9b83db9ceda5263760f to run this app on the brainlife.io platform.

## Running Locally

Requires Singularity, a FreeSurfer license, and Python ≥ 3.6.

```bash
# set your FreeSurfer license key
export FREESURFER_LICENSE="<your license key>"

# create config.json pointing to your inputs
cp config.json.example config.json
# edit config.json to set the actual paths

./main
```

Dependencies (installed inside the container — no local install needed):

- [nibabel](https://nipy.org/nibabel/) ≥ 2.3
- [numpy](https://numpy.org/) ≥ 1.11
- [scipy](https://scipy.org/) ≥ 0.18

## Authors

- Gabriele Amorosino (gabriele.amorosino@utexas.edu)

## Based on

- **Original app**: [app-wmaSeg](https://github.com/brainlife/app-wmaSeg) — Daniel Bullock, Soichi Hayashi, Franco Pestilli (Indiana University)
- **Segmentation toolbox**: [wma_tools](https://github.com/DanNBullock/wma_tools) — Daniel Bullock (Indiana University)

## References

[Bullock, D., Takemura, H., Caiafa, C. F., Kitchell, L., McPherson, B., Caron, B., & Pestilli, F. (2019). Associative white matter connecting the dorsal and ventral posterior human cortex. *Brain Structure and Function*, 224(8), 2631–2660.](https://doi.org/10.1007/s00429-019-01907-8)

[Avesani, P., McPherson, B., Hayashi, S. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Sci Data* 6, 69 (2019).](https://doi.org/10.1038/s41597-019-0073-y)
