"""Grids, resampling, and the rigid realignment of the preoperative MRI to the ultrasound.

World coordinates are those of the NIfTI headers (as read by nibabel). The ultrasound and the preoperative MRI must be
in the same world frame (that of the navigation system).
"""
import os
import shutil
import subprocess

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, map_coordinates

VOXEL = (0.859375, 0.859375, 2.0)          # working grid of the networks, mm
CUBE_SIZE, CUBE_SPACING = 256, 0.5         # grid on which the rigid registration is computed


def resample(src, src_affine, dst_affine, dst_shape, transform=None, order=1, cval=0.0):
    """Resamples `src` on the destination grid. `transform` maps destination world to source world (identity if None)."""
    M = np.linalg.inv(src_affine) @ (np.eye(4) if transform is None else transform) @ dst_affine
    src = np.asarray(src, dtype=np.float32)
    out = np.empty(dst_shape, np.float32)
    jj, kk = np.meshgrid(np.arange(dst_shape[1], dtype=np.float64), np.arange(dst_shape[2], dtype=np.float64), indexing="ij")
    for i in range(dst_shape[0]):                       # slab by slab, to bound the memory
        v = np.stack([np.full(jj.shape, float(i)), jj, kk], -1).reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        out[i] = map_coordinates(src, v.T, order=order, mode="constant", cval=cval).reshape(dst_shape[1:])
    return out


def save(data, affine, path, dtype=np.float32):
    nib.save(nib.Nifti1Image(np.asarray(data).astype(dtype), affine), path)


def load_ultrasound(path):
    """Ultrasound volume as float32 in [0, 255] (the networks were trained with 8-bit B-mode intensities)."""
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    data = np.nan_to_num(data)
    top = float(data.max())
    if top <= 0:
        raise ValueError("the ultrasound volume is empty")
    if top <= 1.0:                                      # intensities in [0, 1]
        data = data * 255.0
    elif top > 255.0:                                   # more than 8 bits
        data = data * (255.0 / float(np.percentile(data[data > 0], 99.9)))
    return np.clip(data, 0, 255), img.affine.astype(np.float64)


def working_grid(us, us_affine, voxel=VOXEL):
    """Axis-aligned working grid (axis codes R, A, I) over the bounding box of the ultrasound field of view.

    Returns the ultrasound resampled on this grid (uint8), the mask of its field of view, and the affine of the grid.
    The field of view is the region swept by the ultrasound, including anechoic regions such as the fluid of a
    resection cavity (their intensity is low but not zero; holes of exactly zero intensity are filled).
    """
    idx = np.argwhere(us > 0)
    if len(idx) < 1000:
        raise ValueError("the ultrasound volume has almost no signal")
    lo, hi = idx.min(0), idx.max(0)
    box = np.array([[i, j, k] for i in (lo[0], hi[0]) for j in (lo[1], hi[1]) for k in (lo[2], hi[2])], float)
    world = box @ us_affine[:3, :3].T + us_affine[:3, 3]
    wlo, whi = world.min(0), world.max(0)
    voxel = np.asarray(voxel, float)
    shape = tuple(int(v) for v in np.maximum(np.ceil((whi - wlo) / voxel).astype(int) + 1, 8))
    affine = np.eye(4)
    affine[:3, :3] = np.diag([voxel[0], voxel[1], -voxel[2]])
    affine[:3, 3] = [wlo[0], wlo[1], whi[2]]
    # trilinear interpolation without low-pass filtering, as in the training data
    usw = resample(us, us_affine, affine, shape, order=1)
    return np.clip(usw, 0, 255).astype(np.uint8), fill_field_of_view(usw > 0), affine


def fill_field_of_view(fov):
    """Field of view without holes (closing and hole filling, slice by slice)."""
    out = np.zeros(fov.shape, bool)
    for k in range(fov.shape[2]):
        out[:, :, k] = binary_fill_holes(binary_closing(fov[:, :, k], iterations=2) | fov[:, :, k])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# rigid realignment (NiftyReg reg_aladin, block matching)
def find_reg_aladin(exe=None):
    exe = exe or os.environ.get("REG_ALADIN") or shutil.which("reg_aladin")
    if not exe or not (os.path.isfile(exe) or shutil.which(exe)):
        raise FileNotFoundError(
            "reg_aladin (NiftyReg) was not found. Install NiftyReg (e.g. `conda install -c conda-forge niftyreg`) or give "
            "its path with --reg-aladin or the REG_ALADIN environment variable; use --no-registration only if the "
            "preoperative MRI is already aligned with the ultrasound.")
    return exe


def registration_cube(us, us_affine):
    """Isotropic cube (256^3 voxels of 0.5 mm) centered on the ultrasound, and the mask of its field of view."""
    R = us_affine[:3, :3]
    R = R / np.linalg.norm(R, axis=0) * CUBE_SPACING
    fg = us > 0
    center = np.argwhere(fg).mean(0) if fg.any() else np.array(us.shape, float) / 2
    center = us_affine[:3, :3] @ center + us_affine[:3, 3]
    affine = np.eye(4)
    affine[:3, :3] = R
    affine[:3, 3] = center - R @ np.array([CUBE_SIZE / 2] * 3)
    cube = resample(us, us_affine, affine, (CUBE_SIZE,) * 3, order=1)
    mask = binary_fill_holes(binary_closing(cube > 0, iterations=2))
    return cube, mask.astype(np.uint8), affine


def rigid_realignment(us, us_affine, mr, mr_affine, workdir, exe=None, threads=8):
    """Rigid registration of the preoperative MRI to the ultrasound, restricted to the ultrasound field of view.

    Returns the 4 x 4 matrix that maps world coordinates of the ultrasound to world coordinates of the MRI.
    """
    exe = find_reg_aladin(exe)
    cube, mask, affine = registration_cube(us, us_affine)
    ref, msk, flo = (os.path.join(workdir, n) for n in ("us_cube.nii.gz", "us_cube_mask.nii.gz", "mr_pre.nii.gz"))
    out, res = os.path.join(workdir, "rigid_us_to_mr.txt"), os.path.join(workdir, "mr_pre_rigid_cube.nii.gz")
    save(cube, affine, ref)
    save(mask, affine, msk, np.uint8)
    save(mr, mr_affine, flo)
    cmd = [exe, "-ref", ref, "-flo", flo, "-rmask", msk, "-rigOnly", "-noSym", "-nac", "-ln", "2", "-lp", "10",
           "-aff", out, "-res", res, "-omp", str(threads)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode or not os.path.exists(out):
        raise RuntimeError("reg_aladin failed:\n" + " ".join(cmd) + "\n" + (r.stdout or "")[-2000:] + (r.stderr or "")[-2000:])
    T = np.loadtxt(out)
    for f in (ref, msk, flo, res):
        if os.path.exists(f):
            os.remove(f)
    return T


def rigid_magnitude(T):
    """Translation (mm) and rotation (degrees) of a rigid transform."""
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(T[:3, 3])), float(ang)
