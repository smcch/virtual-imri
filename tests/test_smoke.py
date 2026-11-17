"""Smoke test with synthetic volumes (no patient data): the pipeline runs end to end and writes its outputs.

    python tests/test_smoke.py          (or: pytest tests)
It skips the rigid registration, so NiftyReg is not needed. The images are noise: only the mechanics are checked.
"""
import os
import sys
import tempfile

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import virtual_imri  # noqa: E402


def phantom(d):
    rng = np.random.default_rng(0)
    # "preoperative MRI": 1 mm isotropic head-like ellipsoid
    g = np.stack(np.meshgrid(*[np.arange(n) for n in (96, 112, 80)], indexing="ij"), -1)
    r = np.sqrt((((g - np.array([48, 56, 40])) / np.array([40, 48, 34])) ** 2).sum(-1))
    mr = np.where(r < 1, 300 + 80 * rng.standard_normal(r.shape), 0).clip(0).astype(np.float32)
    mr_aff = np.diag([1.0, 1.0, 1.0, 1.0])
    # "ultrasound": 0.5 mm wedge inside the head, with an anechoic region; tumor mask around it
    us = np.zeros((90, 90, 70), np.uint8)
    us_aff = np.diag([0.5, 0.5, 0.5, 1.0]); us_aff[:3, 3] = [26, 30, 22]
    u = np.stack(np.meshgrid(*[np.arange(n) for n in us.shape], indexing="ij"), -1)
    wedge = (np.abs(u[..., 0] - 45) < 8 + 0.45 * u[..., 2]) & (np.abs(u[..., 1] - 45) < 8 + 0.45 * u[..., 2])
    us[wedge] = rng.integers(20, 200, int(wedge.sum()))
    us[(((u - np.array([45, 45, 40])) / 12.0) ** 2).sum(-1) < 1] = 1
    tumor = ((((g - np.array([48, 52, 42])) / 9.0) ** 2).sum(-1) < 1).astype(np.uint8)
    p = {n: os.path.join(d, n + ".nii.gz") for n in ("us", "mr", "tumor")}
    nib.save(nib.Nifti1Image(us, us_aff), p["us"])
    nib.save(nib.Nifti1Image(mr, mr_aff), p["mr"])
    nib.save(nib.Nifti1Image(tumor, mr_aff), p["tumor"])
    return p


def test_pipeline_runs():
    with tempfile.TemporaryDirectory() as d:
        p = phantom(d)
        r = virtual_imri.run(p["us"], p["mr"], os.path.join(d, "out"), tumor_path=p["tumor"], registration=False,
                             device="cpu", verbose=False)
        v = nib.load(r["files"]["virtual_imri"])
        x, f = np.asarray(v.dataobj), np.asarray(nib.load(r["files"]["fov_mask"]).dataobj) > 0
        assert x.shape == tuple(r["info"]["grid_shape"]) and np.isfinite(x).all()
        assert 0 <= x.min() and x.max() <= 1 and f.sum() > 1000 and float(np.abs(x[~f]).max()) == 0
        wb = nib.load(r["files"]["virtual_imri_whole_brain"])
        assert wb.shape == nib.load(p["mr"]).shape and np.isfinite(np.asarray(wb.dataobj)).all()
        assert r["info"]["display_rule"] and os.path.exists(r["files"]["tumor_kept"])


if __name__ == "__main__":
    test_pipeline_runs()
    print("ok")
