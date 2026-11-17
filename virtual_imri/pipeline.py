"""Virtual intraoperative MRI from the post-resection navigated ultrasound and the preoperative MRI.

    1. Update the geometry: the preoperative MRI is rigidly realigned to the ultrasound (NiftyReg) and deformed by G1.
    2. Synthesize what has changed: G2 rewrites the warped MRI only inside its rewrite map and outputs a cavity map.
    3. Keep the planned tumor (optional; needs the tumor mask of the preoperative MRI): the tumor of the preoperative
       MRI is restored except where the cavity map indicates that tissue has been removed.

Two images are written: the virtual intraoperative MRI within the ultrasound field of view, and a whole-brain image in
which the preoperative MRI, realigned to the ultrasound, is completed with the virtual image inside the field of view.
"""
import contextlib
import json
import math
import os
import tempfile
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from . import geometry as G
from . import model as M

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "virtual_imri_v1.pth")

# display rule
FAR_TUMOR, FAR_CAVITY = (9, 9, 5), (9, 9, 3)                    # voxels; "surrounding tissue" lies beyond these dilations
SCALES = ((4.0, 1.0), (12.0, 0.03), (36.0, 0.001))               # (sigma in mm, weight) of the extrapolation kernel
# whole-brain image
FEATHER_MM = 4.0


def _norm(v, lo, hi):
    return ((v - lo) / (hi - lo) * 2 - 1).clamp(-1, 1)


def _gauss(x, sigma):
    """Separable Gaussian smoothing of a volume (X, Y, Z); sigma in voxels per axis; zero outside."""
    x = x[None, None]
    for ax, sg in enumerate(sigma):
        r = max(int(3 * sg + 0.5), 1)
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) / sg) ** 2)
        x = F.conv3d(x, (k / k.sum()).view([1, 1] + [-1 if i == ax else 1 for i in range(3)]),
                     padding=[r if i == ax else 0 for i in range(3)])
    return x[0, 0]


def _t(a, device):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)


@torch.no_grad()
def synthesize_on_grid(us, fov, affine, mr, mr_affine, T, g1, g2, tumor=None, device="cuda"):
    """Runs the networks and the display rule on the working grid.

    us        uint8 array (X, Y, Z): ultrasound on the working grid
    fov       boolean array (X, Y, Z): field of view of the ultrasound (it includes anechoic regions)
    affine    4 x 4 affine of the working grid (voxel -> world of the ultrasound)
    mr        preoperative MRI (array) and mr_affine, its affine
    T         4 x 4 rigid transform, world of the ultrasound -> world of the MRI
    tumor     None, or (binary array, affine): tumor mask in the world of the MRI
    Returns a dict of float32 arrays on the working grid; images in [0, 1], 0 outside the field of view.
    """
    device = torch.device(device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    S = tuple(us.shape)
    P = tuple(int(math.ceil(s / m) * m) for s, m in zip(S, M.MULT))
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(0))
    sp = _t(spacing, device)
    ust = torch.from_numpy(np.ascontiguousarray(us)).to(device)
    fov = torch.from_numpy(np.ascontiguousarray(fov).astype(bool)).to(device)
    inside = ust > 0                                                            # voxels with echo: intensity window of the MRI
    up = torch.zeros(P, dtype=torch.float32, device=device)
    up[:S[0], :S[1], :S[2]] = ust.float()
    up = up[None, None] / 127.5 - 1
    vol = torch.from_numpy(np.ascontiguousarray(mr, dtype=np.float32)).to(device)
    Mc = _t(np.linalg.inv(mr_affine) @ T @ affine, device)
    idx, idx_s = M.index_grid(P, device), M.index_grid(S, device)

    # intensity window of the MRI: 1st and 99th percentiles within the ultrasound field of view
    x = M.sample_volume(vol, Mc, idx_s)[inside]
    x = x[::max(1, x.numel() // 200000)]
    lo, hi = torch.quantile(x, torch.tensor([0.01, 0.99], device=device)).tolist()
    hi = max(hi, lo + 1e-3)

    # step 1: deformation
    rig = _norm(M.sample_volume(vol, Mc, idx), lo, hi)[None, None]
    with amp:
        phi_low = g1(torch.cat([up, rig], 1))
    phi = M.upsample(phi_low, P)                                               # mm, along the axes of the grid
    w = _norm(M.sample_volume(vol, Mc, idx + phi[0].permute(1, 2, 3, 0) / sp), lo, hi)[None, None]

    # step 2: synthesis
    with amp:
        o = g2(torch.cat([up, w], 1), w)
    c = lambda v: v[0, 0, :S[0], :S[1], :S[2]].float()  # noqa: E731
    out, m, cav, w, rig = torch.where(fov, c(o["out"]), -1.0), c(o["m"]), c(o["cavity"]), c(w), c(rig)
    phi = phi[0, :, :S[0], :S[1], :S[2]]
    r = dict(networks=out)

    # step 3: display rule
    if tumor is not None and np.any(tumor[0]):
        tm, ta = tumor
        ii = np.argwhere(tm)
        b0, b1 = np.maximum(ii.min(0) - 2, 0), np.minimum(ii.max(0) + 3, tm.shape)
        ta = np.array(ta, dtype=np.float64)
        ta[:3, 3] = ta[:3, :3] @ b0 + ta[:3, 3]
        tv = torch.from_numpy(np.ascontiguousarray(tm[b0[0]:b1[0], b0[1]:b1[1], b0[2]:b1[2]]).astype(np.float32)).to(device)
        tw = M.sample_volume(tv, _t(np.linalg.inv(ta) @ T @ affine, device), idx_s + phi.permute(1, 2, 3, 0) / sp)
        val = (fov & (M.dilate((tw > 0.5)[None, None], FAR_TUMOR)[0, 0] < 0.5)
               & (M.dilate((cav > 0.5)[None, None], FAR_CAVITY)[0, 0] < 0.5)).float()
        num, den = torch.zeros_like(out), torch.zeros_like(out)
        for mm, weight in SCALES:
            sg = [mm / float(v) for v in spacing]
            num, den = num + weight * _gauss((out - w) * val, sg), den + weight * _gauss(val, sg)
        delta = num / den.clamp_min(1e-6) if val.sum() > 1000 else torch.zeros_like(out)
        k = _gauss(tw, (1.0, 1.0, 0.5)) * (1 - M.dilate(cav[None, None], (3, 3, 3))[0, 0])
        m = m * (1 - k)
        out = torch.where(fov, ((1 - k) * out + k * (w + delta)).clamp(-1, 1), -1.0)
        r.update(tumor_planned=tw, tumor_kept=k)

    img = lambda v: (torch.where(fov, v, -1.0) + 1) / 2  # noqa: E731
    r.update(virtual=img(out), networks=img(r["networks"]), rigid=img(rig), warped=img(w), rewrite=m * fov, cavity=cav * fov,
             fov=fov.float(), displacement=phi.norm(dim=0) * fov)
    return {k: v.float().cpu().numpy() for k, v in r.items()}


def whole_brain(virtual, rewrite, fov, affine, mr, mr_affine, T, feather_mm=FEATHER_MM):
    """Whole-brain virtual intraoperative MRI on the grid of the preoperative MRI.

    Inside the ultrasound field of view: the virtual image. Outside it, where the ultrasound provides no information:
    the preoperative MRI after the rigid realignment (its voxels are not resampled), with its intensities mapped to
    those of the virtual image. The two are blended over `feather_mm` at the inner border of the field of view.
    Returns the image, its affine in the world of the ultrasound, and the field of view on that grid.
    """
    A = np.linalg.inv(T) @ mr_affine                                 # voxel of the MRI -> world of the ultrasound
    sh = mr.shape
    zo = np.sqrt((A[:3, :3] ** 2).sum(0))
    # block of the MRI grid that contains the field of view
    c = np.array([[i, j, k] for i in (0, fov.shape[0] - 1) for j in (0, fov.shape[1] - 1) for k in (0, fov.shape[2] - 1)], float)
    v = (c @ affine[:3, :3].T + affine[:3, 3] - A[:3, 3]) @ np.linalg.inv(A[:3, :3]).T
    b0 = np.clip(np.floor(v.min(0)).astype(int) - 3, 0, np.array(sh))
    b1 = np.clip(np.ceil(v.max(0)).astype(int) + 4, 0, np.array(sh))
    P = np.asarray(mr, dtype=np.float32)
    Fv = np.zeros(sh, bool)
    if np.any(b1 - b0 < 2):                                          # the field of view lies outside the MRI
        return P / max(float(np.percentile(P[P > 0], 99)), 1e-6) if (P > 0).any() else P, A, Fv
    Ab = A.copy()
    Ab[:3, 3] = A[:3, :3] @ b0 + A[:3, 3]
    bs = tuple(int(x) for x in b1 - b0)
    blk = tuple(slice(int(a), int(b)) for a, b in zip(b0, b1))
    V = G.resample(virtual, affine, Ab, bs)
    Fb = G.resample((fov > 0.5).astype(np.float32), affine, Ab, bs) > 0.5
    Mb = G.resample(rewrite, affine, Ab, bs)
    Pb = P[blk]
    # intensities of the preoperative MRI -> those of the virtual image, from the tissue that the model copied
    ok = ndi.binary_erosion(Fb, iterations=2) & (Mb < 0.1) & (Pb > 0)
    if ok.sum() < 1000:
        ok = Fb & (Pb > 0)
    if ok.sum() < 100:
        return P / max(float(np.percentile(P[P > 0], 99)), 1e-6) if (P > 0).any() else P, A, Fv
    qs = np.linspace(1, 99, 50)
    qx, qy = np.percentile(Pb[ok], qs), np.percentile(V[ok], qs)
    slope = (qy[-1] - qy[-6]) / max(qx[-1] - qx[-6], 1e-6)           # above the last quantile: line with the slope of the top range
    Pn = np.where(P > qx[-1], qy[-1] + slope * (P - qx[-1]), np.interp(P, qx, qy))
    Pn = (np.clip(Pn, 0, 1.25) * (P > 0)).astype(np.float32)
    wgt = np.clip(ndi.distance_transform_edt(Fb, sampling=zo) / feather_mm, 0, 1).astype(np.float32)
    Pn[blk] = wgt * V + (1 - wgt) * Pn[blk]
    Fv[blk] = Fb
    return Pn, A, Fv


def _preview(path, us, rigid, virtual, cavity, fov, spacing):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False
    ref = cavity > 0.5 if (cavity > 0.5).sum() > 50 else fov > 0.5
    cm = [int(round(x)) for x in ndi.center_of_mass(ref)]
    rows = (("Ultrasound", us / 255.0), ("Preoperative MRI (realigned)", rigid), ("Virtual intraoperative MRI", virtual))
    fig, axs = plt.subplots(3, 3, figsize=(9, 9.4))
    for i, (name, vol) in enumerate(rows):
        for j, axis in enumerate((2, 1, 0)):
            o = [k for k in range(3) if k != axis]
            ax = axs[i, j]
            ax.imshow(np.take(vol, cm[axis], axis=axis).T, cmap="gray", vmin=0, vmax=1, aspect=spacing[o[1]] / spacing[o[0]],
                      interpolation="bilinear")
            if i == 2 and (cavity > 0.5).any():
                ax.contour(np.take(cavity, cm[axis], axis=axis).T, levels=[0.5], colors=["#FFD23F"], linewidths=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle("Three orthogonal planes of the working grid; yellow outline: cavity map of the model", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def run(us_path, mr_path, out_dir, tumor_path=None, tumor_label=None, weights=None, device=None, registration=True,
        reg_aladin=None, threads=8, wholebrain=True, preview=True, verbose=True):
    """Computes the virtual intraoperative MRI of one patient and writes the results to `out_dir`.

    us_path      post-resection navigated 3D ultrasound (NIfTI)
    mr_path      preoperative T2-weighted MRI (NIfTI), in the same world frame as the ultrasound
    tumor_path   optional tumor mask of the preoperative MRI (NIfTI, world frame of the MRI); enables the display rule
    Returns a dict with the paths of the files written and a short summary.
    """
    say = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(out_dir, exist_ok=True)

    us, us_affine = G.load_ultrasound(us_path)
    mri = nib.load(mr_path)
    mr = np.asarray(mri.dataobj, dtype=np.float32)
    if mr.ndim == 4:
        mr = mr[..., 0]
    mr, mr_affine = np.nan_to_num(mr), mri.affine.astype(np.float64)
    c = us_affine[:3, :3] @ np.argwhere(us > 0).mean(0) + us_affine[:3, 3]
    v = np.linalg.inv(mr_affine) @ np.r_[c, 1]
    if np.any(v[:3] < 0) or np.any(v[:3] > np.array(mr.shape) - 1):
        raise ValueError("the center of the ultrasound lies outside the preoperative MRI: the two images must be in the "
                         "same world coordinate frame (that of the navigation system)")

    # step 1a: rigid realignment of the preoperative MRI to the ultrasound
    if registration:
        say("rigid realignment of the preoperative MRI to the ultrasound (reg_aladin) ...")
        with tempfile.TemporaryDirectory() as tmp:
            T = G.rigid_realignment(us, us_affine, mr, mr_affine, tmp, exe=reg_aladin, threads=threads)
    else:
        T = np.eye(4)
    t_reg = time.time() - t0
    usw, fov, affine = G.working_grid(us, us_affine)
    del us
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(0))
    pts = np.argwhere(fov)[::97] @ affine[:3, :3].T + affine[:3, 3]
    shift = float(np.linalg.norm(pts @ T[:3, :3].T + T[:3, 3] - pts, axis=1).mean())
    say("  mean displacement within the field of view: %.1f mm; working grid %s" % (shift, "x".join(map(str, usw.shape))))

    tumor = None
    if tumor_path:
        ti = nib.load(tumor_path)
        td = np.rint(np.asarray(ti.dataobj, dtype=np.float32))
        if td.ndim == 4:
            td = td[..., 0]
        tumor = ((td == tumor_label) if tumor_label is not None else (td > 0), ti.affine.astype(np.float64))
        if not tumor[0].any():
            say("  the tumor mask is empty: the display rule is not applied")
            tumor = None

    # steps 1b-3: networks and display rule
    say("synthesis on %s ..." % device)
    g1, g2, _ = M.load_networks(weights or os.environ.get("VIRTUAL_IMRI_WEIGHTS") or DEFAULT_WEIGHTS, device)
    t1 = time.time()
    r = synthesize_on_grid(usw, fov, affine, mr, mr_affine, T, g1, g2, tumor=tumor, device=device)
    t_net = time.time() - t1

    files = {}

    def write(name, data, aff, dtype=np.float32):
        files[name] = os.path.join(out_dir, name + ".nii.gz")
        G.save(data, aff, files[name], dtype)

    write("virtual_imri", r["virtual"], affine)
    write("ultrasound", usw, affine, np.uint8)
    write("mr_pre_rigid", r["rigid"], affine)
    write("mr_pre_warped", r["warped"], affine)
    write("cavity_map", r["cavity"], affine)
    write("rewrite_map", r["rewrite"], affine)
    write("fov_mask", r["fov"], affine, np.uint8)
    if "tumor_kept" in r:
        write("tumor_planned", r["tumor_planned"], affine)
        write("tumor_kept", r["tumor_kept"], affine)
    np.savetxt(os.path.join(out_dir, "rigid_us_to_mr.txt"), T, fmt="%.8f")
    files["rigid_us_to_mr"] = os.path.join(out_dir, "rigid_us_to_mr.txt")

    if wholebrain:
        say("whole-brain image ...")
        wb, wa, wf = whole_brain(r["virtual"], r["rewrite"], r["fov"], affine, mr, mr_affine, T)
        write("virtual_imri_whole_brain", wb, wa)
        write("fov_mask_whole_brain", wf, wa, np.uint8)
    if preview and _preview(os.path.join(out_dir, "preview.png"), usw, r["rigid"], r["virtual"], r["cavity"], r["fov"], spacing):
        files["preview"] = os.path.join(out_dir, "preview.png")

    ml = float(np.prod(spacing)) / 1000
    t_mm, t_deg = G.rigid_magnitude(T)
    info = dict(ultrasound=os.path.abspath(us_path), mr_pre=os.path.abspath(mr_path),
                tumor_mask=os.path.abspath(tumor_path) if tumor_path else None, display_rule="tumor_kept" in r,
                device=str(device), registration=bool(registration), rigid_translation_mm=round(t_mm, 2),
                rigid_rotation_deg=round(t_deg, 2), rigid_mean_displacement_in_fov_mm=round(shift, 2),
                grid_shape=list(usw.shape), grid_voxel_mm=[round(float(s), 4) for s in spacing],
                fov_volume_ml=round(float((r["fov"] > 0.5).sum() * ml), 1),
                cavity_map_volume_ml=round(float((r["cavity"] > 0.5).sum() * ml), 1),
                rewritten_volume_ml=round(float((r["rewrite"] > 0.5).sum() * ml), 1),
                tumor_kept_volume_ml=round(float((r["tumor_kept"] > 0.5).sum() * ml), 1) if "tumor_kept" in r else None,
                seconds=dict(registration=round(t_reg, 1), networks=round(t_net, 2), total=round(time.time() - t0, 1)))
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    files["info"] = os.path.join(out_dir, "info.json")
    say("done in %.0f s: cavity map %.1f mL -> %s" % (time.time() - t0, info["cavity_map_volume_ml"], out_dir))
    return dict(files=files, info=info)
