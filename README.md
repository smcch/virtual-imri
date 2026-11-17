# virtual-imri

Virtual intraoperative MRI synthesized from navigated ultrasound in brain tumor surgery.

This repository contains the final method of the article *Synthesis of virtual intraoperative MRI from navigated
ultrasound in brain tumor surgery* (Cepeda S, Esteban-Sinovas O, Arrese I, Sarabia R; submitted): the trained model and
the code needed to apply it to new patients. Given the **post-resection navigated 3D ultrasound** and the
**preoperative T2-weighted MRI** of a patient, it computes

- the **virtual intraoperative MRI within the ultrasound field of view**, and
- a **whole-brain virtual intraoperative MRI**, in which the preoperative MRI, realigned to the ultrasound, is
  completed with the virtual image inside the field of view.

The real intraoperative MRI is never needed.

> For research use only. This software is not a medical device and must not be used for clinical decisions.

## Installation

```bash
git clone https://github.com/smcch/virtual-imri.git
cd virtual-imri
conda create -n virtual-imri python=3.10 -y && conda activate virtual-imri
conda install -c conda-forge niftyreg -y          # rigid registration (reg_aladin)
pip install -e .                                  # numpy, scipy, nibabel, torch
pip install matplotlib                            # optional: preview.png
```

Install the PyTorch build that matches your system first if needed (<https://pytorch.org>). A GPU is recommended but
not required: one patient takes about 15 s, most of it for the rigid registration (the networks take less than 1 s on a
GPU and a few seconds on a CPU).

The weights of the model are distributed with the package (`virtual_imri/weights/virtual_imri_v1.pth`, 52 MB; SHA-256
`881c196eefe8c11fbcdec53af9c573269fd0373b0958007023c0dc47ccb131cd`).

## Usage

```bash
virtual-imri --us us_post.nii.gz --mr mr_pre_t2.nii.gz --out results/patient01
```

With the tumor segmented on the preoperative MRI, the planned tumor is kept on the image except where the model finds
a resection cavity (step 3 of the method):

```bash
virtual-imri --us us_post.nii.gz --mr mr_pre_t2.nii.gz --tumor tumor_pre.nii.gz --out results/patient01
```

From Python:

```python
import virtual_imri
result = virtual_imri.run("us_post.nii.gz", "mr_pre_t2.nii.gz", "results/patient01", tumor_path=None)
print(result["files"]["virtual_imri"], result["files"]["virtual_imri_whole_brain"])
```

Options: `--tumor-label N` (label of the tumor in a multi-label mask), `--no-registration` (the preoperative MRI is
already aligned with the ultrasound), `--reg-aladin PATH`, `--threads N`, `--device cuda|cpu`, `--no-whole-brain`,
`--no-preview`. Run `virtual-imri -h` for the full list.

## Inputs

| Input | Requirements |
|---|---|
| `--us` | Navigated 3D ultrasound acquired **after resection**, as one NIfTI volume reconstructed from a tracked sweep. B-mode intensities (8-bit scale); voxels outside the sweep must be zero. |
| `--mr` | Preoperative **T2-weighted** MRI (NIfTI). |
| `--tumor` | Optional. Tumor mask of the preoperative MRI (NIfTI, any grid in the world frame of the MRI). |

The ultrasound and the MRI must be in the **same world coordinate frame**, that of the navigation system, as in the
ReMIND database: the NIfTI headers must place the ultrasound inside the head of the MRI. The method refines this
alignment with a rigid registration, but it cannot find it from scratch.

## Outputs

All files are written to the output directory.

| File | Content |
|---|---|
| `virtual_imri.nii.gz` | Virtual intraoperative MRI within the ultrasound field of view (working grid, 0.86 × 0.86 × 2 mm; intensities in [0, 1], 0 outside the field of view). |
| `virtual_imri_whole_brain.nii.gz` | Whole-brain virtual intraoperative MRI on the grid of the preoperative MRI, placed in the frame of the ultrasound. |
| `fov_mask.nii.gz`, `fov_mask_whole_brain.nii.gz` | Ultrasound field of view on each of the two grids. |
| `cavity_map.nii.gz` | Cavity map of the model (probability). |
| `rewrite_map.nii.gz` | Region rewritten by the model; elsewhere the image is the patient's own preoperative MRI. |
| `ultrasound.nii.gz`, `mr_pre_rigid.nii.gz`, `mr_pre_warped.nii.gz` | Inputs on the working grid: ultrasound, and preoperative MRI after the rigid realignment and after the deformation. |
| `tumor_planned.nii.gz`, `tumor_kept.nii.gz` | With `--tumor`: planned tumor on the working grid, and the part of it that is still displayed. |
| `rigid_us_to_mr.txt` | Rigid transform (4 × 4) from the world of the ultrasound to the world of the preoperative MRI. |
| `preview.png`, `info.json` | Quick-look figure, and a summary (volumes, registration, timing). |

## Method

1. **Update the geometry.** The preoperative MRI is aligned with the ultrasound by an automatic rigid registration
   restricted to the ultrasound field of view (NiftyReg `reg_aladin`) and then deformed by a first network (G1).
2. **Synthesize what has changed.** A second network (G2) receives the ultrasound and the warped preoperative MRI. It
   predicts a rewrite map, new image content, a cavity map, and a smooth intensity correction; the output copies the
   warped MRI wherever nothing is rewritten.
3. **Keep the planned tumor** (only with `--tumor`). The planned tumor is restored from the preoperative MRI except
   where the cavity map indicates that tissue has been removed.

Outside the ultrasound field of view the ultrasound provides no information. The whole-brain image therefore shows
there the preoperative MRI after the rigid realignment (its voxels are not resampled), with its intensities mapped to
those of the virtual image, and blends the two over 4 mm at the inner border of the field of view.

## Model

The networks were trained on the public [ReMIND](https://doi.org/10.7937/3RAG-D070) database (84 patients; navigated
3D ultrasound acquired with a BK5000 system, 5–13 MHz, and T2-weighted MRI at 3 T). The model distributed here was
trained with all of them; its performance was estimated by fivefold cross-validation and is reported in the article.
Run from the original ultrasound volumes on its own working grid, the package gave on those cross-validation patients
results equivalent to those of the article.

Data from other centers, ultrasound systems, MRI sequences, or stages of surgery were not part of the training data.

## Citation

If you use this software, please cite the article (see `CITATION.cff`).

## License and acknowledgments

Code and weights are released under the Apache License 2.0 (see `LICENSE`). The model was trained with the ReMIND
database (Juvekar et al., *Sci Data* 2024; The Cancer Imaging Archive), distributed under CC BY 4.0. The rigid
registration uses [NiftyReg](https://github.com/KCL-BMEIS/niftyreg).
