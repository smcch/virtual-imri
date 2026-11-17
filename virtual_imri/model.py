"""Networks of the virtual intraoperative MRI.

G1 (deformation network): (ultrasound, rigidly realigned preoperative MRI) -> smooth displacement field phi, in mm.
G2 (synthesis network):   (ultrasound, warped preoperative MRI w) -> rewrite map m, new content c, cavity map s,
                          and a smooth intensity correction (gain a, offset b). The output image is

                              out = (1 - m) * (a * w + b) + m * c

                          so synthesized content can appear only where m > 0.

Both are 3D residual U-Nets with three anisotropic downsampling stages. The working grid has a voxel size of about
0.86 x 0.86 x 2 mm, with the third array axis as the slice axis.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MULT = (8, 8, 4)                                   # the grid size must be a multiple of this along each axis
STRIDES = [(2, 2, 1), (2, 2, 2), (2, 2, 2)]
PHI_MAX = 20.0                                     # bound of the displacement field, in mm

_GRIDS = {}


def index_grid(shape, device):
    """Voxel indices of a grid, shape (X, Y, Z, 3), cached."""
    key = (tuple(shape), str(device))
    if key not in _GRIDS:
        axes = [torch.arange(n, device=device, dtype=torch.float32) for n in shape]
        _GRIDS[key] = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
    return _GRIDS[key]


def sample_volume(vol, matrix, idx):
    """Trilinear sampling of `vol` (X, Y, Z) at the positions `idx` (..., 3) of another grid.

    `matrix` (4 x 4) maps voxel indices of that grid to voxel indices of `vol`. Zero outside the volume.
    """
    n = torch.tensor(vol.shape, device=vol.device, dtype=torch.float32) - 1
    vox = idx @ matrix[:3, :3].T + matrix[:3, 3]
    g = (2 * vox / n - 1).flip(-1)[None]
    return F.grid_sample(vol.float()[None, None], g, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]


def upsample(field, size):
    return F.interpolate(field.float(), size=tuple(size), mode="trilinear", align_corners=False)


def dilate(mask, kernel):
    return F.max_pool3d(mask.float(), kernel, 1, tuple(k // 2 for k in kernel))


class Res(nn.Module):
    """Residual block: two 3 x 3 x 3 convolutions with instance normalization and leaky ReLU."""

    def __init__(self, i, o, stride=(1, 1, 1)):
        super().__init__()
        self.c1, self.n1 = nn.Conv3d(i, o, 3, stride, 1), nn.InstanceNorm3d(o, affine=True)
        self.c2, self.n2 = nn.Conv3d(o, o, 3, 1, 1), nn.InstanceNorm3d(o, affine=True)
        self.sk = None if (i == o and tuple(stride) == (1, 1, 1)) else nn.Conv3d(i, o, 1, stride)

    def forward(self, x):
        h = F.leaky_relu(self.n1(self.c1(x)), 0.2)
        h = self.n2(self.c2(h))
        return F.leaky_relu(h + (x if self.sk is None else self.sk(x)), 0.2)


class UNet(nn.Module):
    """Encoder with three anisotropic downsampling stages; decoder up to level `niv` (0 = full resolution)."""

    def __init__(self, ic, ch, niv=0):
        super().__init__()
        self.niv = niv
        self.inc = Res(ic, ch[0])
        self.down = nn.ModuleList(nn.Sequential(Res(ch[k], ch[k + 1], STRIDES[k]), Res(ch[k + 1], ch[k + 1])) for k in range(3))
        self.up = nn.ModuleDict({str(k): Res(ch[k + 1] + ch[k], ch[k]) for k in (2, 1, 0) if k >= niv})

    def forward(self, x):
        f = [self.inc(x)]
        for d in self.down:
            f.append(d(f[-1]))
        h, dec = f[3], {}
        for k in (2, 1, 0):
            if k < self.niv:
                break
            h = F.interpolate(h, size=f[k].shape[2:], mode="trilinear", align_corners=False)
            h = self.up[str(k)](torch.cat([h, f[k]], 1))
            dec[k] = h
        return dec


class G1(nn.Module):
    """Deformation network. Input (B, 2, X, Y, Z) in [-1, 1]; output: low-resolution displacement field in mm."""

    def __init__(self, ch=(16, 32, 64, 128)):
        super().__init__()
        self.u = UNet(2, ch, niv=2)
        self.head = nn.Conv3d(ch[2], 3, 3, 1, 1)

    def forward(self, x):
        return PHI_MAX * torch.tanh(self.head(self.u(x)[2]).float() / 10.0)


class G2(nn.Module):
    """Synthesis network. Input (B, 2, X, Y, Z) in [-1, 1] and the warped MRI w (B, 1, X, Y, Z) in [-1, 1]."""

    def __init__(self, ch=(32, 64, 128, 256)):
        super().__init__()
        self.u = UNet(2, ch, niv=0)
        self.head = nn.Conv3d(ch[0], 3, 3, 1, 1)           # rewrite map m, content c, cavity map s
        self.low = nn.Conv3d(ch[2], 2, 3, 1, 1)            # smooth gain and offset

    def forward(self, x, w):
        dec = self.u(x)
        h = self.head(dec[0]).float()
        lo = upsample(self.low(dec[2]), x.shape[2:])
        a = torch.exp(math.log(2.0) * torch.tanh(lo[:, :1]))          # gain in [0.5, 2]
        b = 0.5 * torch.tanh(lo[:, 1:2])                              # offset in [-0.5, 0.5]
        base = 2 * (a * (w + 1) / 2 + b) - 1                          # preoperative MRI with adapted intensities
        m = torch.sigmoid(h[:, :1])
        c = torch.tanh(h[:, 1:2])
        out = ((1 - m) * base + m * c).clamp(-1, 1)
        return dict(out=out, m=m, c=c, cavity=torch.sigmoid(h[:, 2:3]), base=base)


def load_networks(weights, device):
    """Loads the trained networks (exponential moving average of the weights) in evaluation mode."""
    ck = torch.load(weights, map_location=device, weights_only=True)
    g1, g2 = G1().to(device).eval(), G2().to(device).eval()
    g1.load_state_dict(ck["g1"])
    g2.load_state_dict(ck["g2"])
    return g1, g2, ck.get("meta", {})
