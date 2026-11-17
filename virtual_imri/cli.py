"""Command line:  virtual-imri --us us_post.nii.gz --mr mr_pre_t2.nii.gz --out results/patient01 [--tumor tumor.nii.gz]"""
import argparse

from .pipeline import run


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="virtual-imri",
        description="Virtual intraoperative MRI from the post-resection navigated ultrasound and the preoperative MRI. "
                    "For research use only.")
    ap.add_argument("--us", required=True, help="post-resection navigated 3D ultrasound (NIfTI)")
    ap.add_argument("--mr", required=True, help="preoperative T2-weighted MRI (NIfTI), same world frame as the ultrasound")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--tumor", default=None,
                    help="optional tumor mask of the preoperative MRI (NIfTI): the planned tumor is kept on the image "
                         "except where the model finds a resection cavity")
    ap.add_argument("--tumor-label", type=int, default=None, help="label of the tumor in the mask (default: any nonzero voxel)")
    ap.add_argument("--no-registration", action="store_true",
                    help="skip the rigid realignment (the preoperative MRI is already aligned with the ultrasound)")
    ap.add_argument("--reg-aladin", default=None, help="path of NiftyReg's reg_aladin (default: found in PATH)")
    ap.add_argument("--threads", type=int, default=8, help="threads for the rigid registration")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--weights", default=None, help="weights file (default: the model distributed with the package)")
    ap.add_argument("--no-whole-brain", action="store_true", help="do not write the whole-brain image")
    ap.add_argument("--no-preview", action="store_true", help="do not write preview.png")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    run(a.us, a.mr, a.out, tumor_path=a.tumor, tumor_label=a.tumor_label, weights=a.weights, device=a.device,
        registration=not a.no_registration, reg_aladin=a.reg_aladin, threads=a.threads, wholebrain=not a.no_whole_brain,
        preview=not a.no_preview, verbose=not a.quiet)


if __name__ == "__main__":
    main()
