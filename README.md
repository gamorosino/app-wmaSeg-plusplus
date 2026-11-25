<h1 align="center">White Matter Anatomy Segmentation Plus (wmaSeg++)</h1>

<p align="center">
  Segments major white matter tracts from a whole-brain tractogram with wmaSeg++, a fast Python port of the wma_tools anatomical segmentation (Bullock et al. 2019) using the FreeSurfer Destrieux parcellation.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.987"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.987-blue.svg" alt="Run on brainlife.io"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Segments major white matter tracts from a whole-brain tractogram with wmaSeg++, a fast Python port of the wma_tools anatomical segmentation (Bullock et al. 2019) using the FreeSurfer Destrieux parcellation.**

Converts the FreeSurfer aparc.a2009s+aseg to NIfTI, then applies anatomically defined ROI, planar and endpoint criteria (ported from the MATLAB wma_tools) in parallel to classify streamlines into named bundles such as the arcuate, IFOF, ILF, MdLF, VOF, cingulum, CST and callosal tracts, and builds VTK surfaces for visualisation. Use it for atlas-free anatomical tract segmentation without a MATLAB runtime. The tractogram and FreeSurfer directory must be from the same subject and in the same space; needs a FreeSurfer license.

| | |
|---|---|
| **Inputs** | `neuro/track/tck` · `neuro/freesurfer` |
| **Outputs** | `neuro/wmc` · `neuro/tcks` |
| **Container** | `brainlife/freesurfer-mini:6.0.1` |
| **Resources** | 8 CPUs · 32 GB memory · 3 h walltime |
| **Requires** | `FREESURFER_LICENSE` |
| **Run it** | [doi.org/10.25663/brainlife.app.987](https://doi.org/10.25663/brainlife.app.987) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `track` | `neuro/track/tck` | yes | Whole-brain tractogram (TCK format) |
| `freesurfer` | `neuro/freesurfer` | yes | FreeSurfer output directory (aparc.a2009s+aseg parcellation required) |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `classification` | `neuro/wmc` | White matter tract classification: classification.mat, per-tract JSON files, and VTK surfaces. |
| `tracts_tck` | `neuro/tcks` | Per-bundle TCK files (one per named tract), as alternative to WMC. Only produced when save_tck is enabled. |

<details>
<summary><b>Parameters</b> (1)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `save_tck` | boolean | `true` | Save per-tract TCK files to tracts_tck/ (alternative to WMC) |

</details>

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.987](https://doi.org/10.25663/brainlife.app.987), choose `neuro/track/tck`, `neuro/freesurfer` from your project, set the parameters and press **Execute**. The results are saved to your project as `neuro/wmc`, `neuro/tcks`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `brainlife/freesurfer-mini:6.0.1`), [`jq`](https://jqlang.github.io/jq/), `FREESURFER_LICENSE`.

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "track": "/path/to/track/track",
       "freesurfer": "/path/to/freesurfer/freesurfer",
       "save_tck": true
   }
   ```

3. Run `./main`.

## Troubleshooting

- **ERROR: classification/classification.mat is missing or empty**: Check the log of the segmentation step; the FreeSurfer directory must contain aparc.a2009s+aseg and match the tractogram space.

## 2026-11

- **Tract renaming:** Meyer's loop is now named **Inferior optic radiation** (`Inferior_OR`) and Baum's loop is now named **Superior optic radiation** (`Superior_OR`), following current anatomical nomenclature. Output classification names change from `left/rightmeyer`, `left/rightbaum` to `left/rightInferior_OR`, `left/rightSuperior_OR`.

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `save_tck` | boolean | `false` | If `true`, save each segmented tract as an individual TCK file in `tracts_tck/` |

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- White Matter Anatomy Segmentation Plus (wmaSeg++). brainlife.io App. [doi:10.25663/brainlife.app.987](https://doi.org/10.25663/brainlife.app.987)
- Bullock D, Takemura H, Caiafa CF, et al. Associative white matter connecting the dorsal and ventral posterior human cortex. Brain Struct Funct 224(8):2631-2660 (2019). [doi:10.1007/s00429-019-01907-8](https://doi.org/10.1007/s00429-019-01907-8)
- Fischl. FreeSurfer. *NeuroImage* 62:774-781 (2012). [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)
- Garyfallidis et al. Dipy, a library for the analysis of diffusion MRI data. *Frontiers in Neuroinformatics* 8 (2014). [doi:10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

- Gabriele Amorosino
