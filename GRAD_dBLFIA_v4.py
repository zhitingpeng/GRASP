#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  GRASP dB-LFIA  v8.0.0
#  Standard-addition quantification for digital bead lateral-flow immunoassay
# =============================================================================
# Core rules (v2):
#  1) S0 is the absolute PBS-blank signal.
#  2) Initial threshold theta0 = blank background mean + 3 SD (sigma-clipped).
#  3) Bright-field (BF) defines occupied capture-trap ROIs. Fluorescence images
#     are ECC-registered to (inverted) BF; one trap contributes at most one
#     event. Registration QC = plausible warp (shift, rotation); the ECC
#     coefficient is bounded by ~sqrt(f) and is therefore only a warning.
#  4) Fluorescence objects outside occupied-trap ROIs are never counted.
#  5) ONE common threshold is used for the complete standard-addition series.
#     Default policy is FIXED theta = theta0 (data-independent). The legacy
#     outward search ("adjust theta until the gate passes") is kept only as an
#     explicitly labelled diagnostic policy and is recorded as such in the audit.
#  6) Digital mode uses f = n_positive/N_valid and lambda = -ln(1-f); the fit is
#     weighted by the binomial/Poisson variance of lambda. Digital mode is used
#     only when max_i f_i <= F_MAX_DIGITAL over ALL loaded stages (not just C0);
#     otherwise analog. Fluorescence-only input forces analog mode.
#  7) T+C0 and T+C1 support a two-point estimate. The full quantitative-window
#     gate requires T+C2. The gate compares g1 with the last supported gradient
#     using a stated relative tolerance (GATE_REL_TOL) plus, in digital mode, a
#     one-sided counting-noise allowance (GATE_Z * SE), and with >= 3 points
#     R2 >= R2_MIN. Failure -> OUT OF QUANTITATIVE WINDOW - DILUTE AND RE-MEASURE.
#  8) The processing panel exposes threshold controls only. Registration and
#     trap-QC parameters are fixed, versioned constants exported in the audit.
#  9) A threshold-stability scan (theta0 +/- SCAN_HALF_WIDTH) is always
#     reported so the sensitivity of T to theta is visible, not hidden.
# =============================================================================

import csv
import json
import hashlib
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)

# ----------------------------------------------------------------------------
# Versioned constants (all exported in the audit log)
# ----------------------------------------------------------------------------
APP_NAME = "GRASP dB-LFIA"
APP_VERSION = "8.0.0"
METHOD_ID = "GRASP-gate-endpoint-v2"

F_MAX_DIGITAL = 0.70          # digital mode only if max f over all stages <= this
BG_SD_FACTOR = 3.0            # theta0 = BG + 3 SD
GATE_REL_TOL = 0.15           # endpoint gradient may fall below g1 by at most 15 %
GATE_Z = 1.645                # digital mode: one-sided 95 % counting-noise allowance on the gate
R2_MIN = 0.95                 # linearity requirement when >= 3 points
SCAN_HALF_WIDTH = 20          # threshold-stability scan: theta0 +/- 20
THRESHOLD_SEARCH_RADIUS = 255 # legacy search only
THRESHOLD_SEARCH_STEP = 1
CHAIN_REL_TOL = 1e-6          # numerical tolerance of the inequality
CHAIN_ABS_TOL = 1e-12
ECC_WARN_CC = 0.70            # ECC correlation below this -> warning (cc is bounded by ~sqrt(f))
ECC_MAX_ITER = 250
ECC_EPS = 1e-6
REG_MAX_SHIFT_FRAC = 0.05     # registration QC: |translation| <= 5 % of the short image side
REG_MAX_ROT_DEG = 3.0         # registration QC: |rotation| <= 3 degrees
OBJ_MIN_AREA_PX = 4           # containment diagnostic: minimum area of a positive object
CONTAIN_WARN_RATIO = 0.50     # warn if < 50 % of positive objects fall inside trap ROIs
CONTAIN_MIN_OBJECTS = 30      # ... but only when at least this many objects exist
ROI_DILATION_PX = 2
TRAP_MIN_AREA_PX = 12
TRAP_MAX_AREA_PX = 5000
TRAP_MAX_ASPECT = 2.5
TRAP_BORDER_MARGIN_PX = 3
MIN_VALID_TRAPS = 20
MIN_ROI_OVERLAP = 0.25
MIN_POSITIVE_PIXELS = 1
DEBOUNCE_MS = 250
EPS = 1e-12

# two-sided 97.5 % Student-t quantiles (df 1..10); z used beyond
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
Z975 = 1.960

STAGE_NAMES = ["S0  blank (PBS)", "St0  T + C0", "St1  T + C1",
               "St2  T + C2", "St3  T + C3"]

STATUS_IN = "IN QUANTITATIVE WINDOW"
STATUS_PARTIAL = "PROVISIONAL TWO-POINT ESTIMATE - FULL WINDOW GATE NOT EVALUABLE"
STATUS_OUT = "OUT OF QUANTITATIVE WINDOW - DILUTE AND RE-MEASURE"
STATUS_FAIL = "ANALYSIS FAILED"

AUDIT_CONFIG = {
    "app": APP_NAME, "version": APP_VERSION, "method_id": METHOD_ID,
    "f_max_digital": F_MAX_DIGITAL, "bg_sd_factor": BG_SD_FACTOR,
    "gate_rel_tol": GATE_REL_TOL, "gate_z": GATE_Z, "r2_min": R2_MIN,
    "scan_half_width": SCAN_HALF_WIDTH,
    "threshold_search_radius": THRESHOLD_SEARCH_RADIUS,
    "threshold_search_step": THRESHOLD_SEARCH_STEP,
    "chain_rel_tol": CHAIN_REL_TOL, "chain_abs_tol": CHAIN_ABS_TOL,
    "ecc_warn_cc": ECC_WARN_CC, "ecc_max_iter": ECC_MAX_ITER, "ecc_eps": ECC_EPS,
    "reg_max_shift_frac": REG_MAX_SHIFT_FRAC, "reg_max_rot_deg": REG_MAX_ROT_DEG,
    "obj_min_area_px": OBJ_MIN_AREA_PX, "contain_warn_ratio": CONTAIN_WARN_RATIO,
    "contain_min_objects": CONTAIN_MIN_OBJECTS,
    "roi_dilation_px": ROI_DILATION_PX,
    "trap_min_area_px": TRAP_MIN_AREA_PX, "trap_max_area_px": TRAP_MAX_AREA_PX,
    "trap_max_aspect": TRAP_MAX_ASPECT, "trap_border_margin_px": TRAP_BORDER_MARGIN_PX,
    "min_valid_traps": MIN_VALID_TRAPS, "min_roi_overlap": MIN_ROI_OVERLAP,
    "min_positive_pixels": MIN_POSITIVE_PIXELS,
}

RULES_TEXT = (
    "1) S0 is the absolute PBS-blank signal.\n"
    "2) theta0 = blank background mean + 3 SD (sigma-clipped).\n"
    "3) Bright-field defines occupied trap ROIs; fluorescence is ECC-registered to it; "
    "one trap = at most one event. Registration QC = plausible warp; a low ECC cc "
    "(bounded by ~sqrt(f)) is a warning only.\n"
    "4) Fluorescence outside occupied-trap ROIs is never counted.\n"
    "5) One common threshold for the whole series. Default: FIXED theta = theta0. "
    "Legacy 'search until gate passes' is a labelled diagnostic policy only.\n"
    "6) Digital: f = n_pos/N, lambda = -ln(1-f), Poisson-weighted fit; digital only if "
    f"max f <= {F_MAX_DIGITAL} over ALL stages, else analog. No BF -> analog.\n"
    "7) Two stages -> provisional two-point estimate. Full gate needs T+C2: "
    f"g1 > 0, g_end >= (1-{GATE_REL_TOL}) g1 (minus a {GATE_Z}*SE counting-noise "
    f"allowance in digital mode), and R2 >= {R2_MIN} (>= 3 points).\n"
    "8) Only threshold controls are exposed; registration / trap-QC constants are "
    "fixed, versioned, and exported in the audit log.\n"
    f"9) A threshold-stability scan (theta0 +/- {SCAN_HALF_WIDTH}) is always reported."
)

_IMG_CACHE = {}
_BF_CACHE = {}
_PROFILE_CACHE = {}


def invalidate_cache(path):
    """Drop every cached product of `path` (re-loading a changed file must not
    reuse stale pixels, trap masks or registrations)."""
    _IMG_CACHE.pop(path, None)
    _BF_CACHE.pop(path, None)
    for key in [k for k in _PROFILE_CACHE if path in k]:
        _PROFILE_CACHE.pop(key, None)


# ----------------------------------------------------------------------------
# Image primitives
# ----------------------------------------------------------------------------
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cached_gray(path):
    """8-bit grayscale view of the image (16-bit input is scaled by 1/256)."""
    if path not in _IMG_CACHE:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Cannot read image: {path}")
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)
        depth = int(raw.dtype.itemsize * 8)
        if raw.dtype == np.uint8:
            g = raw
        elif raw.dtype == np.uint16:
            g = (raw >> 8).astype(np.uint8)
        else:
            g = cv2.normalize(raw.astype(np.float32), None, 0, 255,
                              cv2.NORM_MINMAX).astype(np.uint8)
        _IMG_CACHE[path] = (np.ascontiguousarray(g), depth)
    return _IMG_CACHE[path][0]


def image_depth(path):
    cached_gray(path)
    return _IMG_CACHE[path][1]


def robust_blank_stats(gray):
    """Sigma-clipped (3 SD, <= 8 passes) mean / SD of all blank pixels."""
    v = gray.astype(np.float64).ravel()
    keep = np.ones(v.shape, dtype=bool)
    for _ in range(8):
        x = v[keep]
        mu = float(x.mean())
        sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
        w = 3.0 * max(sd, 1e-6)
        new = (v >= mu - w) & (v <= mu + w)
        if np.array_equal(new, keep):
            break
        keep = new
    x = v[keep]
    return float(x.mean()), (float(x.std(ddof=1)) if x.size > 1 else 0.0)


def _gt_table(hist):
    """hist[..., 256] -> gt[..., t] = number of pixels with value > t."""
    ge = np.cumsum(hist[..., ::-1], axis=-1)[..., ::-1]           # >= t
    return np.concatenate([ge[..., 1:], np.zeros(hist.shape[:-1] + (1,),
                                                 dtype=ge.dtype)], axis=-1)


def blank_metrics(blank_path):
    g = cached_gray(blank_path)
    mu, sd = robust_blank_stats(g)
    theta0 = int(np.clip(round(mu + BG_SD_FACTOR * sd), 0, 255))
    hist = np.bincount(g.ravel(), minlength=256).astype(np.int64)
    return {"mu": mu, "sd": sd, "theta0": theta0, "mean": float(g.mean()),
            "sum": float(g.sum()), "gt": _gt_table(hist),
            "depth": image_depth(blank_path), "shape": list(g.shape)}


def global_metrics(fluor_path):
    g = cached_gray(fluor_path)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.int64)
    return {"mean": float(g.mean()), "sum": float(g.sum()), "gt": _gt_table(hist),
            "depth": image_depth(fluor_path), "shape": list(g.shape)}


# ----------------------------------------------------------------------------
# Trap segmentation (bright-field) and registration
# ----------------------------------------------------------------------------
def segment_traps(bf_path):
    if bf_path in _BF_CACHE:
        return _BF_CACHE[bf_path]
    gray = cached_gray(bf_path)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    block = max(15, (min(gray.shape) // 20) | 1)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, 3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    d = ROI_DILATION_PX
    kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
              if d > 0 else None)
    rois, rejected = [], {"area": 0, "border": 0, "shape": 0}
    union = np.zeros((h, w), dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < TRAP_MIN_AREA_PX or area > TRAP_MAX_AREA_PX:
            rejected["area"] += 1
            continue
        if (x < TRAP_BORDER_MARGIN_PX or y < TRAP_BORDER_MARGIN_PX or
                x + bw >= w - TRAP_BORDER_MARGIN_PX or y + bh >= h - TRAP_BORDER_MARGIN_PX):
            rejected["border"] += 1
            continue
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > TRAP_MAX_ASPECT:
            rejected["shape"] += 1
            continue
        x0, y0 = max(0, x - d), max(0, y - d)
        x1, y1 = min(w, x + bw + d), min(h, y + bh + d)
        crop = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        if kernel is not None:
            crop = cv2.dilate(crop, kernel)
        mask = crop > 0
        union[y0:y1, x0:x1] |= mask
        rois.append({"id": len(rois) + 1, "bbox": [x0, y0, x1 - x0, y1 - y0],
                     "mask": mask, "area": area, "npx": int(mask.sum()),
                     "centroid": [float(centroids[i][0]), float(centroids[i][1])]})
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for r in rois:
        x, y, bw, bh = r["bbox"]
        cv2.rectangle(preview, (x, y), (x + bw - 1, y + bh - 1), (0, 200, 0), 1)
    out = {"gray": gray, "rois": rois, "N": len(rois), "union": union,
           "rejected": rejected, "preview": preview}
    _BF_CACHE[bf_path] = out
    return out


def register_fluorescence(fluor_path, bf_path):
    """Euclidean ECC registration of fluorescence onto bright-field.
    Bright-field is inverted first (traps dark in BF, bright in fluorescence) so
    that the two modalities are positively correlated."""
    moving = cached_gray(fluor_path)
    fixed = cached_gray(bf_path)
    if moving.shape != fixed.shape:
        moving = cv2.resize(moving, (fixed.shape[1], fixed.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    f = cv2.normalize(cv2.GaussianBlur(fixed, (5, 5), 0).astype(np.float32),
                      None, 0, 1, cv2.NORM_MINMAX)
    f = 1.0 - f
    m = cv2.normalize(cv2.GaussianBlur(moving, (5, 5), 0).astype(np.float32),
                      None, 0, 1, cv2.NORM_MINMAX)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITER, ECC_EPS)
    try:
        try:
            cc, warp = cv2.findTransformECC(f, m, warp, cv2.MOTION_EUCLIDEAN,
                                            criteria, None, 3)
        except TypeError:   # OpenCV < 4.1 signature
            cc, warp = cv2.findTransformECC(f, m, warp, cv2.MOTION_EUCLIDEAN, criteria)
        registered = cv2.warpAffine(moving, warp, (fixed.shape[1], fixed.shape[0]),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        shift = float(np.hypot(warp[0, 2], warp[1, 2]))
        rot = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
        max_shift = REG_MAX_SHIFT_FRAC * min(fixed.shape)
        problems = []
        if not np.isfinite(cc):
            problems.append("ECC did not converge")
        if shift > max_shift:
            problems.append(f"translation {shift:.1f} px > {max_shift:.1f} px")
        if abs(rot) > REG_MAX_ROT_DEG:
            problems.append(f"rotation {rot:.2f} deg > {REG_MAX_ROT_DEG:g} deg")
        ok, error = (not problems), "; ".join(problems)
        if not ok:                      # an implausible warp is worse than none
            registered = moving
            error += " (identity transform used)"
    except cv2.error as e:
        cc, ok, error = np.nan, False, "ECC failed: " + str(e).split("\n")[0]
        shift, rot = np.nan, np.nan
        registered = moving
    # The ECC coefficient is intrinsically bounded by ~sqrt(f) (only a fraction f of
    # the traps is lit), so a low cc is reported as a warning, not a failure.
    low_cc = bool(np.isfinite(cc) and cc < ECC_WARN_CC)
    return {"image": registered, "warp": np.asarray(warp).tolist(), "cc": float(cc),
            "shift_px": shift, "rotation_deg": rot, "ok": ok, "error": error,
            "low_cc": low_cc}


def containment_qc(profile, threshold):
    """Diagnostic: share of positive objects (connected components after a 3x3
    opening, area >= OBJ_MIN_AREA_PX) whose centroid lies inside a trap ROI.
    Low containment at a plausible threshold indicates misregistration or
    heavy debris; it is reported, and only warns when enough objects exist."""
    t = int(np.clip(threshold, 0, 255))
    binary = (profile["image"] > t).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, _, stats, cent = cv2.connectedComponentsWithStats(binary, 8)
    union = profile["bf"]["union"]
    inside = outside = 0
    for i in range(1, n):
        if stats[i][cv2.CC_STAT_AREA] < OBJ_MIN_AREA_PX:
            continue
        cx, cy = int(round(cent[i][0])), int(round(cent[i][1]))
        cy = min(max(cy, 0), union.shape[0] - 1)
        cx = min(max(cx, 0), union.shape[1] - 1)
        if union[cy, cx]:
            inside += 1
        else:
            outside += 1
    total = inside + outside
    return {"n_inside": inside, "n_outside": outside,
            "ratio": (inside / total) if total else np.nan,
            "warn": bool(total >= CONTAIN_MIN_OBJECTS and total and
                         inside / total < CONTAIN_WARN_RATIO)}


def stage_profile(fluor_path, bf_path):
    """Threshold-independent per-trap profile. Computed once per (fluor, BF) pair;
    afterwards any threshold is evaluated in O(N) with array look-ups."""
    key = (fluor_path, bf_path)
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]
    bf = segment_traps(bf_path)
    reg = register_fluorescence(fluor_path, bf_path)
    img = reg["image"]
    rois = bf["rois"]
    N = len(rois)
    hist = np.zeros((N, 256), dtype=np.int64)
    isum = np.zeros(N, dtype=np.float64)
    npx = np.zeros(N, dtype=np.int64)
    for k, r in enumerate(rois):
        x, y, bw, bh = r["bbox"]
        vals = img[y:y + bh, x:x + bw][r["mask"]]
        hist[k] = np.bincount(vals, minlength=256)
        isum[k] = float(vals.sum())
        npx[k] = vals.size
    outside = img[~bf["union"]]
    hist_out = np.bincount(outside, minlength=256).astype(np.int64)
    union_vals = img[bf["union"]]
    out = {"bf": bf, "registration": reg, "image": img, "N": N,
           "gt": _gt_table(hist) if N else np.zeros((0, 256), np.int64),
           "gt_out": _gt_table(hist_out), "npx": npx, "isum": isum,
           "mean_px": isum / np.maximum(npx, 1),
           "union_mean": float(union_vals.mean()) if union_vals.size else np.nan,
           "fluor_path": fluor_path, "bf_path": bf_path, "stage_label": ""}
    _PROFILE_CACHE[key] = out
    return out


def trap_metrics(profile, threshold):
    t = int(np.clip(threshold, 0, 255))
    N = profile["N"]
    if N == 0:
        return {"N": 0, "n_positive": 0, "f": np.nan, "positive_mask": np.zeros(0, bool),
                "assigned_positive_pixels": 0,
                "unassigned_positive_pixels": int(profile["gt_out"][t]),
                "mean_intensity": np.nan}
    pos_n = profile["gt"][:, t]
    overlap = pos_n / np.maximum(profile["npx"], 1)
    is_pos = (pos_n >= MIN_POSITIVE_PIXELS) & (overlap >= MIN_ROI_OVERLAP)
    n_pos = int(is_pos.sum())
    return {"N": N, "n_positive": n_pos, "f": n_pos / N, "positive_mask": is_pos,
            "assigned_positive_pixels": int(pos_n.sum()),
            "unassigned_positive_pixels": int(profile["gt_out"][t]),
            "mean_intensity": profile["union_mean"]}


def trap_rows(profile, threshold):
    t = int(np.clip(threshold, 0, 255))
    m = trap_metrics(profile, t)
    rows = []
    for k, r in enumerate(profile["bf"]["rois"]):
        pos_n = int(profile["gt"][k, t])
        rows.append({"trap_id": r["id"], "positive": bool(m["positive_mask"][k]),
                     "overlap": float(pos_n / max(profile["npx"][k], 1)),
                     "positive_pixels": pos_n,
                     "integrated_intensity": float(profile["isum"][k]),
                     "mean_intensity": float(profile["mean_px"][k])})
    return rows


def stage_overlay(profile, positive_mask):
    rgb = cv2.cvtColor(profile["image"], cv2.COLOR_GRAY2RGB)
    for k, r in enumerate(profile["bf"]["rois"]):
        x, y, bw, bh = r["bbox"]
        color = (255, 40, 40) if positive_mask[k] else (0, 200, 0)
        cv2.rectangle(rgb, (x, y), (x + bw - 1, y + bh - 1), color, 1)
    return rgb


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def safe_lambda(f):
    if not np.isfinite(f):
        return np.nan
    return float(-np.log(1.0 - np.clip(f, 1e-9, 1.0 - 1e-9)))


def lambda_variance(f, N):
    """Var(lambda) from binomial counting: f / (N (1-f)); f clipped to [0.5/N, 1-0.5/N]."""
    if not np.isfinite(f) or N <= 0:
        return np.nan
    fc = float(np.clip(f, 0.5 / N, 1.0 - 0.5 / N))
    return fc / (N * (1.0 - fc))


def fit_standard_addition(C, y, var=None):
    """Weighted linear fit y = a C + b.  x_intercept = -b/a, T = |x_intercept|.
    var: per-point variance of y (digital mode, Poisson) -> weights 1/var and a
    dispersion-corrected covariance; None -> ordinary least squares."""
    C = np.asarray(C, float)
    y = np.asarray(y, float)
    var = None if var is None else np.asarray(var, float)
    m = np.isfinite(C) & np.isfinite(y)
    if var is not None:
        m &= np.isfinite(var) & (var > 0)
    C, y = C[m], y[m]
    var = None if var is None else var[m]
    empty = {"a": np.nan, "b": np.nan, "x_intercept": np.nan, "T": np.nan,
             "r2": np.nan, "intercept_sign_ok": False, "n": int(len(C)), "df": 0,
             "se_x": np.nan, "T_ci": [np.nan, np.nan], "ci_k": np.nan,
             "ci_basis": "none", "dispersion": np.nan, "weighted": var is not None,
             "residuals": [], "sd_y": []}
    if len(C) < 2 or len(np.unique(C)) < 2:
        return empty
    w = np.ones_like(C) if var is None else 1.0 / var
    X = np.column_stack([C, np.ones_like(C)])
    XtW = X.T * w
    try:
        Ainv = np.linalg.inv(XtW @ X)
    except np.linalg.LinAlgError:
        return empty
    beta = Ainv @ (XtW @ y)
    a, b = float(beta[0]), float(beta[1])
    if not np.isfinite(a) or abs(a) <= EPS:
        return empty
    resid = y - X @ beta
    n, df = len(C), len(C) - 2
    chi2 = float(np.sum(w * resid ** 2))
    ybar = float(np.sum(w * y) / np.sum(w))
    sst = float(np.sum(w * (y - ybar) ** 2))
    r2 = np.nan if df < 1 else (1.0 if sst < EPS else 1.0 - chi2 / sst)
    xi = -b / a
    T = abs(xi)
    if var is not None:
        phi = max(1.0, chi2 / df) if df >= 1 else 1.0
        cov = phi * Ainv
        k = T975.get(df, Z975) if df >= 1 else Z975
        basis = (f"Poisson weights, dispersion phi={phi:.2f}, "
                 + (f"t(df={df})" if df >= 1 else "z, counting error only (2-point)"))
    elif df >= 1:
        phi = chi2 / df
        cov = phi * Ainv
        k = T975.get(df, Z975)
        basis = f"OLS residual variance, t(df={df})"
    else:
        phi, cov, k, basis = np.nan, None, np.nan, "none (2-point analog)"
    if cov is not None:
        va, vb, cab = cov[0, 0], cov[1, 1], cov[0, 1]
        var_x = (vb + xi * xi * va + 2.0 * xi * cab) / (a * a)   # delta method
        se_x = float(np.sqrt(max(var_x, 0.0)))
        ci = [max(0.0, T - k * se_x), T + k * se_x]
    else:
        se_x, ci = np.nan, [np.nan, np.nan]
    return {"a": a, "b": b, "x_intercept": float(xi), "T": float(T), "r2": float(r2),
            "intercept_sign_ok": bool(xi < 0), "n": int(n), "df": int(df),
            "se_x": se_x, "T_ci": ci, "ci_k": float(k), "ci_basis": basis,
            "dispersion": float(phi), "weighted": var is not None,
            "residuals": resid.tolist(),
            "sd_y": (np.sqrt(var).tolist() if var is not None else [])}


def quant_window_gate(S0, S, C, fit, var_stage=None, var_blank=0.0):
    """Endpoint rule.  g1 = (S_t0 - S0)/T is the left-hand side; the right-hand
    side is the LAST gradient the loaded data supports:
        2 stages : g2 = (S_t1 - S0)/(T + C1)
        3 stages : g3 = (S_t2 - S_t1)/(C2 - C1)
        4 stages : g4 = (S_t3 - S_t2)/(C3 - C2)
    Checks:  (i) g1 > 0;  (ii) g_end >= (1 - GATE_REL_TOL) * g1;
             (iii) R2 >= R2_MIN when >= 3 points.
    In digital mode (var_stage given) check (ii) receives a one-sided counting-
    noise allowance of GATE_Z * SE(g_end - (1-tol) g1), so that a series is not
    declared saturated when the shortfall is within Poisson noise."""
    out = {"gradients": {}, "gradient_labels": {}, "checks": [], "pass": None,
           "coverage": "none", "required_rule": "not evaluable", "rule_expanded": "",
           "message": "", "n_evaluable": 0, "endpoint": "", "violation": np.inf,
           "ratio": np.nan}
    T = fit["T"]
    if not np.isfinite(T) or T <= 0:
        out["message"] = "T not positive; gate undefined"
        return out
    g, lab = {}, {}
    if len(S) >= 1:
        g["g1"] = (S[0] - S0) / T
        lab["g1"] = "(S_t0-S_0)/T"
    if len(S) >= 2 and len(C) >= 2 and (T + C[1]) > EPS:
        g["g2"] = (S[1] - S0) / (T + C[1])
        lab["g2"] = "(S_t1-S_0)/(T+C_1)"
    for j in range(2, len(S)):
        if j < len(C) and (C[j] - C[j - 1]) > EPS:
            g[f"g{j + 1}"] = (S[j] - S[j - 1]) / (C[j] - C[j - 1])
            lab[f"g{j + 1}"] = f"(S_t{j}-S_t{j - 1})/(C_{j}-C_{j - 1})"
    out["gradients"] = {k: float(v) for k, v in g.items()}
    out["gradient_labels"] = lab
    order = sorted(g, key=lambda s: int(s[1:]))
    if len(order) < 2:
        out["message"] = "no inequality evaluable"
        return out
    endpoint = order[-1]
    g1, ge = g["g1"], g[endpoint]
    checks = []
    c1 = bool(g1 > EPS)
    checks.append({"name": "response", "rule": "g1 > 0", "pass": c1,
                   "detail": f"g1 = {g1:.5g}"})
    ratio = ge / g1 if abs(g1) > EPS else np.nan
    scale = max(abs(g1), abs(ge), 1.0)
    tol_num = CHAIN_ABS_TOL + CHAIN_REL_TOL * scale
    margin = 0.0
    if var_stage is not None and len(var_stage) == len(S):
        v = np.asarray(var_stage, float)
        if endpoint == "g2":
            vg = v[1] / (T + C[1]) ** 2
        else:
            j = int(endpoint[1:]) - 1
            vg = (v[j] + v[j - 1]) / (C[j] - C[j - 1]) ** 2
        vg1 = (v[0] + float(var_blank)) / T ** 2
        margin = float(GATE_Z * np.sqrt(vg + (1.0 - GATE_REL_TOL) ** 2 * vg1))
    c2 = bool(c1 and ge + margin >= (1.0 - GATE_REL_TOL) * g1 - tol_num)
    detail = f"{endpoint}/g1 = {ratio:.3f}"
    if margin > 0 and abs(g1) > EPS:
        detail += f"; counting-noise allowance {margin / abs(g1):.3f}"
    checks.append({"name": "endpoint",
                   "rule": f"{endpoint}/g1 >= {1 - GATE_REL_TOL:g}" + (" - z*SE" if margin > 0 else ""),
                   "pass": c2, "detail": detail, "noise_margin_rel": (margin / abs(g1)) if abs(g1) > EPS else np.nan})
    viol = (0.0 if c1 else 10.0)
    viol += 0.0 if c2 else (max(0.0, (1 - GATE_REL_TOL) - ratio) if np.isfinite(ratio) else 10.0)
    if len(order) >= 3 and fit["n"] >= 3:
        c3 = bool(np.isfinite(fit["r2"]) and fit["r2"] >= R2_MIN)
        checks.append({"name": "linearity", "rule": f"R2 >= {R2_MIN:g}", "pass": c3,
                       "detail": f"R2 = {fit['r2']:.4f}"})
        viol += 0.0 if c3 else (max(0.0, R2_MIN - fit["r2"]) if np.isfinite(fit["r2"]) else 1.0)
    passed = all(c["pass"] for c in checks)
    out.update({"checks": checks, "pass": passed, "n_evaluable": len(checks),
                "endpoint": endpoint, "ratio": float(ratio),
                "violation": float(viol),
                "required_rule": "; ".join(c["rule"] for c in checks),
                "rule_expanded": f"{lab['g1']} vs {lab[endpoint]}",
                "coverage": "full" if len(order) >= 3 else "two_point"})
    failed = [f"{c['name']} ({c['detail']})" for c in checks if not c["pass"]]
    out["message"] = ("all checks satisfied" if passed else "failed: " + "; ".join(failed))
    return out


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def build_context(blank_path, fluor_paths, bf_paths, C, blank_bf_path=None, progress=None):
    def report(msg):
        if progress:
            progress(msg)
    report("Blank statistics ...")
    blank = blank_metrics(blank_path)
    n = len(fluor_paths)
    paired = n > 0 and all(bf_paths[i] for i in range(n))
    profiles, globals_ = [], []
    if paired:
        for k, (fp, bp) in enumerate(zip(fluor_paths, bf_paths)):
            report(f"Registering / profiling stage St{k} ...")
            p = stage_profile(fp, bp)
            p["stage_label"] = f"St{k}"
            profiles.append(p)
    else:
        globals_ = [global_metrics(fp) for fp in fluor_paths]
    blank_profile = None
    if paired and blank_bf_path:
        report("Profiling blank traps ...")
        blank_profile = stage_profile(blank_path, blank_bf_path)
        blank_profile["stage_label"] = "S0"
    return {"blank": blank, "paired": paired, "profiles": profiles, "globals": globals_,
            "blank_profile": blank_profile, "C": [float(c) for c in C],
            "blank_path": blank_path, "fluor_paths": list(fluor_paths),
            "bf_paths": list(bf_paths), "blank_bf_path": blank_bf_path}


def evaluate_threshold(ctx, threshold, mode_setting="auto"):
    t = int(np.clip(threshold, 0, 255))
    blank, C = ctx["blank"], ctx["C"]
    qc_errors, warns, stages = [], [], []
    var, var_stage, var0 = None, None, 0.0
    if ctx["paired"]:
        for p in ctx["profiles"]:
            m = trap_metrics(p, t)
            m["registration_cc"] = p["registration"]["cc"]
            stages.append(m)
            if not p["registration"]["ok"]:
                qc_errors.append(f"{p['stage_label']}: registration {p['registration']['error']}")
            if p["N"] < MIN_VALID_TRAPS:
                qc_errors.append(f"{p['stage_label']}: N={p['N']} valid traps "
                                 f"(minimum {MIN_VALID_TRAPS})")
        low = [f"{p['stage_label']} ({p['registration']['cc']:.2f})"
               for p in ctx["profiles"] if p["registration"]["ok"] and p["registration"]["low_cc"]]
        if low:
            warns.append(f"ECC cc < {ECC_WARN_CC:g} on " + ", ".join(low) +
                         " - expected when only a fraction of traps is lit; verify in the overlay tab")
        f = np.array([m["f"] for m in stages], float)
        f_max = float(np.nanmax(f)) if f.size else np.nan
        if mode_setting == "auto":
            mode = "digital" if (np.isfinite(f_max) and f_max <= F_MAX_DIGITAL) else "analog"
            mode_reason = (f"auto: max f = {f_max:.3f} "
                           f"{'<=' if mode == 'digital' else '>'} {F_MAX_DIGITAL:g}")
        else:
            mode, mode_reason = mode_setting, "manual selection"
            if mode == "digital" and np.isfinite(f_max) and f_max > F_MAX_DIGITAL:
                warns.append(f"manual digital mode with max f = {f_max:.3f} > "
                             f"{F_MAX_DIGITAL:g}: Poisson correction unreliable")
        if mode == "digital":
            lam = np.array([safe_lambda(x) for x in f])
            var_stage = np.array([lambda_variance(m["f"], m["N"]) for m in stages])
            var0 = 0.0
            if ctx["blank_profile"] is not None:
                mb = trap_metrics(ctx["blank_profile"], t)
                lam0 = safe_lambda(mb["f"])
                var0 = lambda_variance(mb["f"], mb["N"])
                blank_desc = (f"lambda0 = {lam0:.4g} from blank traps "
                              f"({mb['n_positive']}/{mb['N']})")
                if not ctx["blank_profile"]["registration"]["ok"]:
                    warns.append("blank: registration QC failed "
                                 f"({ctx['blank_profile']['registration']['error']}); "
                                 "lambda0 computed on the unregistered image")
            else:
                lam0, blank_desc = 0.0, "lambda0 = 0 (no blank bright-field loaded)"
            var = var_stage + var0
            S_abs, S0, y = lam, float(lam0), lam - lam0
            unit = "lambda"
        else:
            S_abs = np.array([m["mean_intensity"] for m in stages], float)
            S0 = float(blank["mu"])
            y = S_abs - S0
            blank_desc = f"S0 = blank robust mean {S0:.3f} a.u./px"
            unit = "mean trap intensity (a.u./px)"
    else:
        mode, mode_reason, f_max = "analog", "fluorescence-only input", np.nan
        for gm in ctx["globals"]:
            stages.append({"N": 0, "n_positive": 0, "f": np.nan,
                           "mean_intensity": gm["mean"],
                           "positive_pixels": int(gm["gt"][t])})
        S_abs = np.array([m["mean_intensity"] for m in stages], float)
        S0 = float(blank["mean"])
        y = S_abs - S0
        blank_desc = f"S0 = blank global mean {S0:.3f} a.u./px"
        unit = "mean image intensity (a.u./px)"
    fit = fit_standard_addition(C, y, var)
    gate = quant_window_gate(S0, list(S_abs), C, fit,
                             var_stage if mode == "digital" else None, var0)
    return {"threshold": t, "mode": mode, "mode_reason": mode_reason,
            "paired_registration": bool(ctx["paired"]), "f_max": float(f_max),
            "blank_signal": float(S0), "blank_desc": blank_desc, "unit": unit,
            "stage_signals": [float(v) for v in S_abs], "y": [float(v) for v in y],
            "var_y": (None if var is None else [float(v) for v in var]),
            "fit": fit, "gate": gate, "stages": stages,
            "qc_errors": qc_errors, "warnings": warns}


def _feasible(ev):
    fit, gate = ev["fit"], ev["gate"]
    return (not ev["qc_errors"] and np.isfinite(fit["T"]) and fit["T"] > 0
            and fit["intercept_sign_ok"] and gate["n_evaluable"] >= 1)


def threshold_candidates(theta0):
    lo = max(0, theta0 - THRESHOLD_SEARCH_RADIUS)
    hi = min(255, theta0 + THRESHOLD_SEARCH_RADIUS)
    vals = list(range(lo, hi + 1, THRESHOLD_SEARCH_STEP))
    return sorted(vals, key=lambda t: (abs(t - theta0), t))


def select_threshold(ctx, policy="fixed", mode_setting="auto", manual_theta=None):
    """policy = 'fixed'  : theta = theta0 = BG + 3 SD  (recommended, data-independent)
                'manual' : theta = manual_theta
                'search' : LEGACY outward sweep from theta0 until the gate passes."""
    theta0 = ctx["blank"]["theta0"]
    if policy == "manual":
        thetas = [int(np.clip(manual_theta if manual_theta is not None else theta0, 0, 255))]
    elif policy == "search":
        thetas = threshold_candidates(theta0)
    else:
        policy, thetas = "fixed", [theta0]
    evaluations, solved = [], None
    for theta in thetas:
        try:
            ev = evaluate_threshold(ctx, theta, mode_setting)
        except Exception as exc:
            evaluations.append({"threshold": theta, "error": str(exc),
                                "theta_distance": abs(theta - theta0),
                                "qc_errors": [str(exc)]})
            continue
        ev["theta_distance"] = abs(theta - theta0)
        evaluations.append(ev)
        if _feasible(ev) and ev["gate"]["pass"] is True:
            solved = ev
            break
    if policy != "search":
        if not evaluations or "fit" not in evaluations[0]:
            raise RuntimeError(evaluations[0]["error"] if evaluations else "no evaluation")
        chosen = evaluations[0]
        chosen["gate_solved"] = bool(_feasible(chosen) and chosen["gate"]["pass"] is True)
        chosen["selection_reason"] = ("fixed theta0 = BG + 3 SD (data-independent)"
                                      if policy == "fixed" else "manual threshold")
    elif solved is not None:
        chosen = solved
        chosen["gate_solved"] = True
        chosen["selection_reason"] = ("LEGACY SEARCH: nearest theta to theta0 at which "
                                      f"'{chosen['gate']['required_rule']}' holds")
    else:
        scored = [e for e in evaluations if "fit" in e and e["gate"]["n_evaluable"] >= 1]
        if not scored:
            scored = [e for e in evaluations if "fit" in e]
        if not scored:
            raise RuntimeError("No threshold candidate could be evaluated")

        def rank(e):
            v = e["gate"].get("violation", np.inf)
            return (1 if e.get("qc_errors") else 0,
                    0 if e["fit"].get("intercept_sign_ok") else 1,
                    v if np.isfinite(v) else 1e12, e["theta_distance"], e["threshold"])
        chosen = min(scored, key=rank)
        chosen["gate_solved"] = False
        chosen["selection_reason"] = ("LEGACY SEARCH: no theta in 0-255 satisfied the gate; "
                                      "minimum-violation candidate shown for diagnosis only")
    chosen.update({"theta0": int(theta0), "policy": policy,
                   "n_candidates": len(evaluations)})
    if policy == "search":
        chosen["candidate_summary"] = [{
            "threshold": e["threshold"], "distance": e["theta_distance"],
            "T": e.get("fit", {}).get("T"), "x_intercept": e.get("fit", {}).get("x_intercept"),
            "gate_pass": e.get("gate", {}).get("pass"),
            "violation": e.get("gate", {}).get("violation"),
            "qc_errors": e.get("qc_errors", []), "error": e.get("error", "")}
            for e in evaluations]
    return chosen


def stability_scan(ctx, theta_center, mode_setting="auto", half_width=SCAN_HALF_WIDTH,
                   mode_ref=None):
    """T as a function of theta around theta_center. Reports the contiguous
    plateau of thresholds (containing theta_center) on which the gate passes
    in the SAME mode as the reported result (a threshold that flips the mode
    is shown but never counted as part of the plateau)."""
    lo, hi = max(0, theta_center - half_width), min(255, theta_center + half_width)
    rows = []
    for t in range(lo, hi + 1):
        try:
            ev = evaluate_threshold(ctx, t, mode_setting)
        except Exception as exc:
            rows.append({"theta": t, "T": np.nan, "pass": False, "feasible": False,
                         "f_max": np.nan, "mode": None, "error": str(exc)})
            continue
        feas = _feasible(ev)
        same_mode = (mode_ref is None) or (ev["mode"] == mode_ref)
        rows.append({"theta": t, "T": float(ev["fit"]["T"]) if feas else np.nan,
                     "pass": bool(feas and same_mode and ev["gate"]["pass"]),
                     "feasible": feas, "same_mode": same_mode,
                     "f_max": ev["f_max"], "mode": ev["mode"]})
    passing = {r["theta"] for r in rows if r["pass"]}
    plateau = None
    if theta_center in passing:
        a = b = theta_center
        while (a - 1) in passing:
            a -= 1
        while (b + 1) in passing:
            b += 1
        plateau = [a, b]
    Ts = np.array([r["T"] for r in rows if r["pass"]], float)
    Tp = (np.array([r["T"] for r in rows if plateau and plateau[0] <= r["theta"] <= plateau[1]], float)
          if plateau else np.array([]))
    return {"theta_range": [lo, hi], "rows": rows, "n_pass": int(len(Ts)),
            "plateau": plateau, "plateau_width": (plateau[1] - plateau[0] + 1) if plateau else 0,
            "T_median_pass": float(np.median(Ts)) if Ts.size else np.nan,
            "T_cv_plateau": (float(Tp.std(ddof=1) / Tp.mean()) if Tp.size > 1 and Tp.mean() > 0
                             else np.nan),
            "T_min_plateau": float(Tp.min()) if Tp.size else np.nan,
            "T_max_plateau": float(Tp.max()) if Tp.size else np.nan}


def status_from_result(res):
    fit, gate = res["fit"], res["gate"]
    if res["qc_errors"] or not np.isfinite(fit["T"]) or not fit["intercept_sign_ok"]:
        return STATUS_FAIL
    if gate["pass"] is not True:
        return STATUS_OUT
    return STATUS_IN if gate["coverage"] == "full" else STATUS_PARTIAL


def stage_summaries(res):
    out = []
    for k, m in enumerate(res["stages"]):
        out.append({"stage": f"St{k}", "C_fM": res["C"][k], "N": int(m.get("N", 0)),
                    "n_positive": int(m.get("n_positive", 0)), "f": float(m.get("f", np.nan)),
                    "S_abs": res["stage_signals"][k], "y": res["y"][k],
                    "sd_y": (res["var_y"][k] ** 0.5 if res["var_y"] else None),
                    "registration_cc": m.get("registration_cc"),
                    "unassigned_positive_pixels": m.get("unassigned_positive_pixels")})
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


# ----------------------------------------------------------------------------
# GUI helpers
# ----------------------------------------------------------------------------
class Tooltip:
    def __init__(self, widget, text=""):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def set_text(self, text):
        self.text = text

    def _show(self, _=None):
        if not self.text or self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, bg="#FFFBE6", relief="solid", bd=1,
                 padx=6, pady=3, justify="left", wraplength=440).pack()

    def _hide(self, _=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class ScrollFrame(ttk.Frame):
    """Vertically scrollable container (used for the overlay gallery)."""
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def to_photo(arr, box):
    img = Image.fromarray(arr if arr.ndim == 2 else arr.astype(np.uint8))
    img.thumbnail(box, Image.LANCZOS)
    return ImageTk.PhotoImage(img)


# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------
class App:
    THUMB = (300, 220)

    def __init__(self, root):
        self.root = root
        root.title(f"{APP_NAME}  v{APP_VERSION}")
        root.minsize(1180, 740)
        self.fluor_paths = [None] * 5          # 0 = blank, 1..4 = T+C0..T+C3
        self.bf_paths = [None] * 5             # 0 = optional blank BF, 1..4 = stages
        self.last_result = None
        self.last_scan = None
        self.overlay_arrays = {}
        self._photos = {}
        self._debounce = None
        self._busy = False

        self.c_vars = [tk.StringVar(value="0"), tk.StringVar(), tk.StringVar(), tk.StringVar()]
        self.mode_var = tk.StringVar(value="auto")
        self.policy_var = tk.StringVar(value="fixed")
        self.theta_var = tk.IntVar(value=20)
        self.scan_var = tk.BooleanVar(value=True)
        self.autorun_var = tk.BooleanVar(value=True)

        self._theme()
        self._build_menu()
        self._build_body()
        self._build_statusbar()
        self._refresh_controls()
        self.root.bind("<Control-r>", lambda e: self.calculate())
        self.root.bind("<Control-e>", lambda e: self.export_audit())
        self.root.bind("<F5>", lambda e: self.calculate())

    # ---------------- theme / fonts ----------------
    def _theme(self):
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(size=10)
            except tk.TclError:
                pass
        try:
            tkfont.nametofont("TkFixedFont").configure(size=9)
        except tk.TclError:
            pass
        fam = tkfont.nametofont("TkDefaultFont").actual("family")
        self.font_bold = (fam, 10, "bold")
        self.font_small = (fam, 9)
        self.font_big = (fam, 20, "bold")
        self.font_mono = (tkfont.nametofont("TkFixedFont").actual("family"), 9)
        style = ttk.Style(self.root)
        style.configure("Primary.TButton", font=self.font_bold, padding=(14, 5))
        style.configure("Small.TButton", padding=(4, 1))
        style.configure("Section.TLabelframe.Label", font=self.font_bold)
        style.configure("Hint.TLabel", foreground="#6B7280", font=self.font_small)
        style.configure("Big.TLabel", font=self.font_big, foreground="#1D4ED8")

    # ---------------- menu ----------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Run analysis", accelerator="F5 / Ctrl+R", command=self.calculate)
        filem.add_command(label="Export audit folder...", accelerator="Ctrl+E", command=self.export_audit)
        filem.add_separator()
        filem.add_command(label="Clear all inputs", command=self.clear_all)
        filem.add_separator()
        filem.add_command(label="Quit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=filem)
        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="Method rules", command=self.show_rules)
        helpm.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", f"{APP_NAME} v{APP_VERSION}\nMethod: {METHOD_ID}\n"
                     f"Python {platform.python_version()}, OpenCV {cv2.__version__}, "
                     f"NumPy {np.__version__}"))
        menubar.add_cascade(label="Help", menu=helpm)
        self.root.config(menu=menubar)

    # ---------------- body ----------------
    def _build_body(self):
        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        left = ttk.Frame(paned, padding=(4, 2))
        right = ttk.Frame(paned, padding=(4, 2))
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        self._build_inputs(left)
        self._build_settings(left)
        self._build_run(left)
        self._build_result(left)
        self._build_tabs(right)

    def _build_inputs(self, parent):
        sec = ttk.LabelFrame(parent, text="1 · Images and spike concentrations",
                             style="Section.TLabelframe", padding=(8, 4))
        sec.pack(fill="x", pady=(0, 6))
        heads = ("Stage", "Fluorescence", "Bright-field", "Spike C (fM)", "Readout")
        for c, h in enumerate(heads):
            ttk.Label(sec, text=h, font=self.font_bold).grid(row=0, column=c, sticky="w",
                                                               padx=(0, 8), pady=(0, 2))
        self.fluor_name, self.bf_name, self.readout, self.bf_readout = [], [], [], []
        self.fluor_tip, self.bf_tip = [], []
        self.c_entries = []
        for i in range(5):
            r = i + 1
            ttk.Label(sec, text=STAGE_NAMES[i], font=self.font_bold).grid(
                row=r, column=0, sticky="w", padx=(0, 8), pady=2)
            fc = ttk.Frame(sec)
            fc.grid(row=r, column=1, sticky="w", padx=(0, 8))
            ttk.Button(fc, text="Load...", width=7, style="Small.TButton",
                       command=lambda i=i: self.load_image("fluor", i)).pack(side="left")
            lab = ttk.Label(fc, text="—", width=16, anchor="w", style="Hint.TLabel")
            lab.pack(side="left", padx=(4, 0))
            self.fluor_name.append(lab)
            self.fluor_tip.append(Tooltip(lab, ""))
            bc = ttk.Frame(sec)
            bc.grid(row=r, column=2, sticky="w", padx=(0, 8))
            ttk.Button(bc, text="Load...", width=7, style="Small.TButton",
                       command=lambda i=i: self.load_image("bf", i)).pack(side="left")
            ttk.Button(bc, text="×", width=2, style="Small.TButton",
                       command=lambda i=i: self.clear_bf(i)).pack(side="left", padx=(2, 0))
            blab = ttk.Label(bc, text="—", width=14, anchor="w", style="Hint.TLabel")
            blab.pack(side="left", padx=(4, 0))
            self.bf_name.append(blab)
            self.bf_tip.append(Tooltip(blab, ""))
            if i == 0:
                ttk.Label(sec, text="—", style="Hint.TLabel").grid(row=r, column=3, sticky="w")
                Tooltip(bc, "Optional for the blank. If loaded, the blank's lambda0 is "
                            "measured on its traps and subtracted in digital mode.")
            elif i == 1:
                ttk.Label(sec, text="0 (fixed)", style="Hint.TLabel").grid(row=r, column=3, sticky="w")
            else:
                e = ttk.Entry(sec, textvariable=self.c_vars[i - 1], width=9, justify="right")
                e.grid(row=r, column=3, sticky="w")
                e.bind("<KeyRelease>", lambda ev: self.schedule_recalc())
                e.bind("<Return>", lambda ev: self.calculate())
                self.c_entries.append(e)
            rd = ttk.Label(sec, text="", width=24, anchor="w", font=self.font_small)
            rd.grid(row=r, column=4, sticky="w")
            self.readout.append(rd)
        ttk.Label(sec, style="Hint.TLabel", wraplength=560, justify="left",
                  text="Minimum: blank + T+C0 + T+C1 (two-point, provisional). The full "
                       "quantitative-window gate needs T+C2. Require 0 = C0 < C1 < C2 < C3. "
                       "Load stages in order; bright-field is needed on every loaded stage "
                       "for digital counting.").grid(row=6, column=0, columnspan=5,
                                                      sticky="w", pady=(4, 0))

    def _build_settings(self, parent):
        sec = ttk.LabelFrame(parent, text="2 · Analysis settings",
                             style="Section.TLabelframe", padding=(8, 4))
        sec.pack(fill="x", pady=(0, 6))
        # mode
        ttk.Label(sec, text="Mode:", font=self.font_bold).grid(row=0, column=0, sticky="w")
        mf = ttk.Frame(sec)
        mf.grid(row=0, column=1, sticky="w")
        self.mode_radios = []
        for txt, val in (("Auto", "auto"), ("Digital (lambda)", "digital"), ("Analog", "analog")):
            rb = ttk.Radiobutton(mf, text=txt, value=val, variable=self.mode_var,
                                 command=self._on_setting_change)
            rb.pack(side="left", padx=(0, 10))
            self.mode_radios.append(rb)
        self.mode_note = ttk.Label(sec, style="Hint.TLabel", wraplength=540, justify="left",
                                   text=f"Auto: digital when max f over all loaded stages "
                                        f"<= {F_MAX_DIGITAL:g}; fluorescence-only input forces analog.")
        self.mode_note.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        # threshold policy
        ttk.Label(sec, text="Threshold θ:", font=self.font_bold).grid(row=2, column=0, sticky="nw")
        pf = ttk.Frame(sec)
        pf.grid(row=2, column=1, sticky="w")
        for txt, val in (("Fixed θ0 = BG + 3 SD  (recommended)", "fixed"),
                         ("Manual", "manual"),
                         ("Legacy search until gate passes  (diagnostic only)", "search")):
            ttk.Radiobutton(pf, text=txt, value=val, variable=self.policy_var,
                            command=self._on_policy_change).pack(anchor="w")
        tf = ttk.Frame(sec)
        tf.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        self.theta_scale = tk.Scale(tf, from_=0, to=255, orient="horizontal", length=360,
                                    variable=self.theta_var, showvalue=False,
                                    command=self._on_scale)
        self.theta_scale.pack(side="left")
        self.theta_spin = ttk.Spinbox(tf, from_=0, to=255, width=5, textvariable=self.theta_var,
                                      command=self._on_spin)
        self.theta_spin.pack(side="left", padx=(6, 0))
        self.theta_spin.bind("<Return>", lambda e: self._on_spin())
        self.theta_note = ttk.Label(sec, style="Hint.TLabel", wraplength=540, justify="left",
                                    text="Load the blank to obtain BG, SD and θ0.")
        self.theta_note.grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(sec, text=f"Threshold-stability scan (θ ± {SCAN_HALF_WIDTH}; "
                                  "see 'Threshold scan' tab)",
                        variable=self.scan_var, command=self._on_setting_change).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def _build_run(self, parent):
        sec = ttk.LabelFrame(parent, text="3 · Run", style="Section.TLabelframe", padding=(8, 4))
        sec.pack(fill="x", pady=(0, 6))
        ttk.Button(sec, text="▶  Run analysis", style="Primary.TButton",
                   command=self.calculate).pack(side="left")
        ttk.Button(sec, text="Export audit...", command=self.export_audit).pack(side="left", padx=6)
        ttk.Button(sec, text="Clear all", command=self.clear_all).pack(side="left")
        ttk.Checkbutton(sec, text="Auto-run on change", variable=self.autorun_var).pack(
            side="right")

    def _build_result(self, parent):
        sec = ttk.LabelFrame(parent, text="Result:  T = |x-intercept|",
                             style="Section.TLabelframe", padding=(8, 4))
        sec.pack(fill="both", expand=True)
        self.T_label = ttk.Label(sec, text="T = —", style="Big.TLabel")
        self.T_label.pack(anchor="w")
        self.ci_label = ttk.Label(sec, text="", style="Hint.TLabel", wraplength=560, justify="left")
        self.ci_label.pack(anchor="w")
        self.banner = tk.Label(sec, text="Load images to begin.", bg="#E5E7EB", fg="#111827",
                               anchor="w", justify="left", padx=10, pady=6,
                               font=self.font_bold, wraplength=540)
        self.banner.pack(fill="x", pady=(4, 4))
        self.info_label = ttk.Label(sec, text="", justify="left", wraplength=560)
        self.info_label.pack(anchor="w")
        self.gate_label = ttk.Label(sec, text="", justify="left", font=self.font_mono, wraplength=560)
        self.gate_label.pack(anchor="w", pady=(4, 0))
        self.qc_label = ttk.Label(sec, text="", justify="left", wraplength=560, style="Hint.TLabel")
        self.qc_label.pack(anchor="w", pady=(4, 0))

    def _build_tabs(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        self.nb = nb
        # fit
        tab_fit = ttk.Frame(nb)
        nb.add(tab_fit, text="Standard-addition fit")
        self.fig = Figure(figsize=(5.6, 4.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab_fit)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, tab_fit).pack(side="bottom", fill="x")
        # scan
        tab_scan = ttk.Frame(nb)
        nb.add(tab_scan, text="Threshold scan")
        self.fig2 = Figure(figsize=(5.6, 4.4), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=tab_scan)
        self.canvas2.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas2, tab_scan).pack(side="bottom", fill="x")
        # overlays
        tab_img = ScrollFrame(nb)
        nb.add(tab_img, text="Trap overlays")
        g = tab_img.inner
        ttk.Label(g, text="Fluorescence (registered)  —  red = positive trap, green = negative",
                  font=self.font_bold).grid(row=0, column=1, padx=6, pady=4, sticky="w")
        ttk.Label(g, text="Bright-field  —  accepted trap ROIs", font=self.font_bold).grid(
            row=0, column=2, padx=6, pady=4, sticky="w")
        self.thumb_fluor, self.thumb_bf, self.thumb_cap = [], [], []
        for i in range(5):
            cap = ttk.Label(g, text=STAGE_NAMES[i], font=self.font_bold, width=16, anchor="nw")
            cap.grid(row=i + 1, column=0, sticky="nw", padx=6, pady=6)
            self.thumb_cap.append(cap)
            lf = tk.Label(g, relief="sunken", bd=1, bg="#F3F4F6", width=self.THUMB[0] // 7,
                          height=self.THUMB[1] // 14, cursor="hand2")
            lf.grid(row=i + 1, column=1, padx=6, pady=6)
            lf.bind("<Double-Button-1>", lambda e, k=("fluor", i): self.enlarge(k))
            self.thumb_fluor.append(lf)
            lb = tk.Label(g, relief="sunken", bd=1, bg="#F3F4F6", width=self.THUMB[0] // 7,
                          height=self.THUMB[1] // 14, cursor="hand2")
            lb.grid(row=i + 1, column=2, padx=6, pady=6)
            lb.bind("<Double-Button-1>", lambda e, k=("bf", i): self.enlarge(k))
            self.thumb_bf.append(lb)
        ttk.Label(g, text="Double-click a thumbnail to open it at full size.",
                  style="Hint.TLabel").grid(row=6, column=0, columnspan=3, sticky="w", padx=6)
        # details
        tab_det = ttk.Frame(nb)
        nb.add(tab_det, text="Details / audit")
        self.details = ScrolledText(tab_det, wrap="none", font=self.font_mono, state="disabled")
        self.details.pack(fill="both", expand=True)

    def _build_statusbar(self):
        self.status = ttk.Label(self.root, text="Ready.", anchor="w", relief="sunken", padding=(6, 2))
        self.status.pack(fill="x", side="bottom")

    # ---------------- small helpers ----------------
    def set_status(self, text):
        self.status.config(text=text)
        self.root.update_idletasks()

    def loaded_stage_count(self):
        n = 0
        for i in range(1, 5):
            if self.fluor_paths[i] is None:
                break
            n += 1
        return n

    def paired(self, n):
        return n > 0 and all(self.bf_paths[i] is not None for i in range(1, n + 1))

    def read_concentrations(self, n):
        vals = [0.0]
        for var in self.c_vars[1:n]:
            s = var.get().strip()
            if not s:
                raise ValueError("Enter C values for all loaded stages")
            vals.append(float(s))
        if any(vals[i] <= vals[i - 1] for i in range(1, len(vals))):
            raise ValueError("Require 0 = C0 < C1 < C2 < C3")
        return vals

    def _refresh_controls(self):
        n = self.loaded_stage_count()
        paired = self.paired(n)
        state = "normal" if paired else "disabled"
        self.mode_radios[1].config(state=state)
        if not paired:
            self.mode_note.config(text="Analog only: bright-field is missing for at least one "
                                       "loaded stage. Load BF on every stage for digital counting.")
        else:
            self.mode_note.config(text=f"Auto: digital when max f over all loaded stages "
                                       f"<= {F_MAX_DIGITAL:g}.")
        manual = self.policy_var.get() == "manual"
        self.theta_scale.config(state="normal" if manual else "disabled",
                                troughcolor="#DBEAFE" if manual else "#E5E7EB")
        self.theta_spin.config(state="normal" if manual else "disabled")

    # ---------------- event handlers ----------------
    def load_image(self, kind, idx):
        path = filedialog.askopenfilename(
            title=f"{'Fluorescence' if kind == 'fluor' else 'Bright-field'} — {STAGE_NAMES[idx]}",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")])
        if not path:
            return
        name = Path(path).name
        invalidate_cache(path)
        if kind == "fluor":
            self.fluor_paths[idx] = path
            self.fluor_name[idx].config(text=name[:16], foreground="#111827")
            self.fluor_tip[idx].set_text(path)
            try:
                set_preview_arr = cached_gray(path)
                self._set_thumb(self.thumb_fluor[idx], set_preview_arr, ("fluor", idx))
            except Exception as e:
                messagebox.showerror("Image error", str(e))
                self.fluor_paths[idx] = None
                return
            if idx == 0:
                try:
                    b = blank_metrics(path)
                    self.theta_var.set(b["theta0"])
                    self.theta_note.config(text=f"Blank: BG = {b['mu']:.2f}, SD = {b['sd']:.2f}  →  "
                                                f"θ0 = {b['theta0']}   ({b['depth']}-bit input)")
                except Exception as e:
                    self.theta_note.config(text=f"Blank statistics failed: {e}")
        else:
            self.bf_paths[idx] = path
            self.bf_name[idx].config(text=name[:14], foreground="#111827")
            self.bf_tip[idx].set_text(path)
            try:
                bf = segment_traps(path)
                self._set_thumb(self.thumb_bf[idx], bf["preview"], ("bf", idx))
                self.readout[idx].config(text=f"BF: N={bf['N']}  rejected={sum(bf['rejected'].values())}")
            except Exception as e:
                messagebox.showerror("Image error", str(e))
                self.bf_paths[idx] = None
                return
        self._refresh_controls()
        self.schedule_recalc()

    def clear_bf(self, idx):
        self.bf_paths[idx] = None
        self.bf_name[idx].config(text="—", foreground="#6B7280")
        self.bf_tip[idx].set_text("")
        self.thumb_bf[idx].config(image="")
        self.thumb_bf[idx].image = None
        self.overlay_arrays.pop(("bf", idx), None)
        self._refresh_controls()
        self.schedule_recalc()

    def _on_setting_change(self):
        self.schedule_recalc()

    def _on_policy_change(self):
        self._refresh_controls()
        self.schedule_recalc()

    def _on_scale(self, _value):
        if self.policy_var.get() == "manual":
            self.schedule_recalc()

    def _on_spin(self):
        try:
            v = int(self.theta_var.get())
        except (tk.TclError, ValueError):
            return
        self.theta_var.set(int(np.clip(v, 0, 255)))
        if self.policy_var.get() == "manual":
            self.schedule_recalc()

    def schedule_recalc(self):
        if not self.autorun_var.get():
            return
        if self._debounce is not None:
            try:
                self.root.after_cancel(self._debounce)
            except Exception:
                pass
        self._debounce = self.root.after(DEBOUNCE_MS, self.calculate)

    # ---------------- analysis ----------------
    def calculate(self):
        if self._busy:
            return
        n = self.loaded_stage_count()
        if self.fluor_paths[0] is None or n < 2:
            self.set_status("Waiting for input: PBS blank + T+C0 + T+C1 (minimum).")
            return
        try:
            C = self.read_concentrations(n)
        except ValueError as e:
            self.set_status(f"Waiting for input: {e}")
            return
        self._busy = True
        try:
            self.set_status("Computing ...")
            ctx = build_context(self.fluor_paths[0], self.fluor_paths[1:1 + n],
                                self.bf_paths[1:1 + n], C,
                                blank_bf_path=self.bf_paths[0], progress=self.set_status)
            policy = self.policy_var.get()
            mode_setting = self.mode_var.get() if ctx["paired"] else "analog"
            manual_theta = int(self.theta_var.get()) if policy == "manual" else None
            self.set_status("Selecting threshold / fitting ...")
            res = select_threshold(ctx, policy, mode_setting, manual_theta)
            res.update({"C": C, "blank_path": self.fluor_paths[0], "blank_bf_path": self.bf_paths[0],
                        "fluor_paths": self.fluor_paths[1:1 + n], "bf_paths": self.bf_paths[1:1 + n],
                        "blank_mu": ctx["blank"]["mu"], "blank_sd": ctx["blank"]["sd"],
                        "input_depth_bits": ctx["blank"]["depth"],
                        "timestamp_utc": datetime.now(timezone.utc).isoformat()})
            if self.bf_paths[0] and not ctx["paired"]:
                res["warnings"].append("blank bright-field loaded but stages are not all paired: ignored")
            res["containment"] = []
            if ctx["paired"]:
                for p in ctx["profiles"]:
                    cq = containment_qc(p, res["threshold"])
                    cq["stage"] = p["stage_label"]
                    res["containment"].append(cq)
                    if cq["warn"]:
                        res["warnings"].append(
                            f"{p['stage_label']}: only {cq['ratio']:.0%} of positive objects "
                            f"({cq['n_inside']}/{cq['n_inside'] + cq['n_outside']}) lie inside "
                            "trap ROIs - check registration / debris")
            res["status"] = status_from_result(res)
            scan = None
            if self.scan_var.get():
                self.set_status("Threshold-stability scan ...")
                scan = stability_scan(ctx, res["threshold"], mode_setting, mode_ref=res["mode"])
            res["stability_scan"] = ({k: v for k, v in scan.items() if k != "rows"} if scan else None)
            self.last_result, self.last_scan, self._ctx = res, scan, ctx
        except Exception as e:
            self._busy = False
            self.banner.config(text=f"{STATUS_FAIL}: {e}", bg="#B91C1C", fg="white")
            self.set_status("Analysis error.")
            return
        self._busy = False
        if res["policy"] != "manual":
            self.theta_var.set(res["threshold"])
        try:
            self._update_result_panel(res)
            self._update_readouts(res)
            self._update_overlays(res)
            self.plot_fit(res)
            self.plot_scan(res, scan)
            self._update_details(res, scan)
        except Exception as e:
            self.set_status(f"Display error: {e}")
            return
        self.set_status(f"Done {datetime.now().strftime('%H:%M:%S')}  —  "
                        f"θ = {res['threshold']} ({res['policy']}), mode = {res['mode']}, "
                        f"T = {res['fit']['T']:.4g} fM")

    # ---------------- UI updates ----------------
    def _update_result_panel(self, res):
        fit, gate = res["fit"], res["gate"]
        warn = "" if fit["intercept_sign_ok"] else "   [invalid: +x intercept]"
        self.T_label.config(text=f"T = {fit['T']:.4g} fM{warn}" if np.isfinite(fit["T"])
                            else "T = —")
        lo, hi = fit["T_ci"]
        if np.isfinite(lo) and np.isfinite(hi):
            self.ci_label.config(text=f"95 % CI [{lo:.4g}, {hi:.4g}] fM   "
                                      f"(SE = {fit['se_x']:.3g}; {fit['ci_basis']})")
        else:
            self.ci_label.config(text=f"95 % CI: not available ({fit['ci_basis']})")
        status = res["status"]
        bg = {"IN": "#15803D", "PARTIAL": "#B45309"}.get(
            "IN" if status == STATUS_IN else "PARTIAL" if status == STATUS_PARTIAL else "X", "#B91C1C")
        extra = gate["message"] if gate["message"] and gate["message"] != "all checks satisfied" else ""
        self.banner.config(text=status + (f"\n{extra}" if extra else ""), bg=bg, fg="white")
        r2 = "n/a (2-point)" if not np.isfinite(fit["r2"]) else f"{fit['r2']:.4f}"
        self.info_label.config(
            text=f"mode = {res['mode']} ({res['mode_reason']})\n"
                 f"θ = {res['threshold']}  [θ0 = {res['theta0']}, policy = {res['policy']}]"
                 f"   R² = {r2}   points = {fit['n']}   x-int = {fit['x_intercept']:.5g}\n"
                 f"{res['blank_desc']}")
        grads = "  ".join(f"{k}={v:.4g}" for k, v in
                          sorted(gate["gradients"].items(), key=lambda kv: int(kv[0][1:])))
        checks = "\n".join(f"  {'OK  ' if c['pass'] else 'FAIL'} {c['rule']:<22} {c['detail']}"
                           for c in gate["checks"])
        self.gate_label.config(text=f"gate ({gate['coverage']}): {gate['rule_expanded']}\n"
                                    f"{grads}\n{checks}" if gate["checks"] else
                                    f"gate: {gate['message']}")
        qc = "QC: PASS" if not res["qc_errors"] else "QC: " + "; ".join(res["qc_errors"])
        if res.get("warnings"):
            qc += "\nWarnings: " + "; ".join(res["warnings"])
        if res.get("stability_scan"):
            s = res["stability_scan"]
            if s["plateau"]:
                qc += (f"\nStability: gate holds for θ ∈ [{s['plateau'][0]}, {s['plateau'][1]}] "
                       f"(width {s['plateau_width']}), T range "
                       f"[{s['T_min_plateau']:.4g}, {s['T_max_plateau']:.4g}], CV = {s['T_cv_plateau']:.1%}"
                       if np.isfinite(s["T_cv_plateau"]) else
                       f"\nStability: plateau width {s['plateau_width']}")
            else:
                qc += f"\nStability: gate does not hold at θ = {res['threshold']} (no plateau)"
        self.qc_label.config(text=qc)

    def _update_readouts(self, res):
        self.readout[0].config(text=f"S0 = {res['blank_signal']:.4g}  ({res['unit']})"
                               if res["mode"] != "digital" else
                               f"λ0 = {res['blank_signal']:.4g}")
        for k, m in enumerate(res["stages"]):
            i = k + 1
            if res["paired_registration"]:
                txt = f"f = {m['f']:.3f}  ({m['n_positive']}/{m['N']})"
                if res["mode"] == "digital":
                    txt += f"  λ = {res['stage_signals'][k]:.3g}"
                if not np.isfinite(m.get("registration_cc", np.nan)) or \
                        m["registration_cc"] < ECC_WARN_CC:
                    txt += "  ⚠reg"
            else:
                txt = f"mean = {m['mean_intensity']:.3g} a.u./px"
            self.readout[i].config(text=txt)

    def _set_thumb(self, label, arr, key):
        self.overlay_arrays[key] = arr
        photo = to_photo(arr, self.THUMB)
        self._photos[key] = photo
        label.config(image=photo, width=photo.width(), height=photo.height())
        label.image = photo

    def _update_overlays(self, res):
        if res["paired_registration"]:
            for k, (m, p) in enumerate(zip(res["stages"], self._ctx["profiles"])):
                self._set_thumb(self.thumb_fluor[k + 1], stage_overlay(p, m["positive_mask"]), ("fluor", k + 1))
                self._set_thumb(self.thumb_bf[k + 1], p["bf"]["preview"], ("bf", k + 1))
            bp = self._ctx["blank_profile"]
            if bp is not None:
                mb = trap_metrics(bp, res["threshold"])
                self._set_thumb(self.thumb_fluor[0], stage_overlay(bp, mb["positive_mask"]), ("fluor", 0))
                self._set_thumb(self.thumb_bf[0], bp["bf"]["preview"], ("bf", 0))

    def enlarge(self, key):
        arr = self.overlay_arrays.get(key)
        if arr is None:
            return
        kind, i = key
        top = tk.Toplevel(self.root)
        top.title(f"{STAGE_NAMES[i]} — {'fluorescence overlay' if kind == 'fluor' else 'bright-field ROIs'}")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        photo = to_photo(arr, (int(sw * 0.85), int(sh * 0.85)))
        lab = tk.Label(top, image=photo)
        lab.image = photo
        lab.pack()

    def plot_fit(self, res):
        ax = self.ax
        ax.clear()
        C, y, fit = np.asarray(res["C"]), np.asarray(res["y"]), res["fit"]
        if res["var_y"]:
            ax.errorbar(C, y, yerr=np.sqrt(res["var_y"]), fmt="o", color="#2563EB",
                        ecolor="#93C5FD", capsize=3, label="stages ± SD (Poisson)")
        else:
            ax.scatter(C, y, color="#2563EB", s=55, label="stages")
        if np.isfinite(fit["a"]):
            lo = min(-1.3 * abs(fit["T"]), -1.0)
            hi = max(C) * 1.15 + EPS
            x = np.linspace(lo, hi, 200)
            ax.plot(x, fit["a"] * x + fit["b"], color="#2563EB", lw=1.3)
        if np.isfinite(fit["x_intercept"]):
            ax.scatter([fit["x_intercept"]], [0], marker="*", s=150, color="#16A34A",
                       zorder=6, label=f"x-int = {fit['x_intercept']:.4g} → T = {fit['T']:.4g} fM")
            lo_ci, hi_ci = fit["T_ci"]
            if np.isfinite(lo_ci) and np.isfinite(hi_ci) and fit["intercept_sign_ok"]:
                ax.axvspan(-hi_ci, -lo_ci, color="#86EFAC", alpha=0.35, label="95 % CI of T")
        ax.axhline(0, color="black", lw=.7, ls="--")
        ax.axvline(0, color="grey", lw=.7, ls="--")
        ax.set_xlabel("Added concentration Ci (fM)")
        ax.set_ylabel("λ − λ0" if res["mode"] == "digital" else "S_ti − S0 (a.u./px)")
        ax.set_title(f"{res['mode']} | θ = {res['threshold']} | {res['status']}", fontsize=9,
                     color="#B91C1C" if res["status"] in (STATUS_OUT, STATUS_FAIL) else "black")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def plot_scan(self, res, scan):
        ax = self.ax2
        ax.clear()
        if scan is None:
            ax.text(0.5, 0.5, "Stability scan disabled", ha="center", va="center",
                    transform=ax.transAxes)
            self.canvas2.draw_idle()
            return
        th = np.array([r["theta"] for r in scan["rows"]])
        T = np.array([r["T"] for r in scan["rows"]], float)
        ps = np.array([r["pass"] for r in scan["rows"]], bool)
        om = np.array([not r.get("same_mode", True) for r in scan["rows"]], bool)
        ax.plot(th, T, color="#9CA3AF", lw=1)
        ax.scatter(th[ps], T[ps], color="#16A34A", s=28, label=f"gate passes ({res['mode']})", zorder=4)
        ax.scatter(th[~ps & ~om], T[~ps & ~om], color="#DC2626", s=18, marker="x",
                   label="gate fails / infeasible", zorder=4)
        if om.any():
            ax.scatter(th[om], T[om], color="#F59E0B", s=18, marker="^",
                       label="other mode (auto switched)", zorder=4)
        if scan["plateau"]:
            ax.axvspan(scan["plateau"][0] - 0.5, scan["plateau"][1] + 0.5, color="#BBF7D0", alpha=0.4,
                       label=f"plateau (CV {scan['T_cv_plateau']:.1%})"
                       if np.isfinite(scan["T_cv_plateau"]) else "plateau")
        ax.axvline(res["threshold"], color="#2563EB", ls="--", lw=1, label=f"θ used = {res['threshold']}")
        ax.axvline(res["theta0"], color="#111827", ls=":", lw=1, label=f"θ0 = {res['theta0']}")
        ax.set_xlabel("threshold θ (8-bit gray level)")
        ax.set_ylabel("T estimate (fM)")
        ax.set_title("Sensitivity of T to threshold", fontsize=9)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        self.fig2.tight_layout()
        self.canvas2.draw_idle()

    def _update_details(self, res, scan):
        fit, gate = res["fit"], res["gate"]
        L = [f"{APP_NAME} v{APP_VERSION}   method {METHOD_ID}   {res['timestamp_utc']}",
             f"policy = {res['policy']}   θ0 = {res['theta0']}   θ = {res['threshold']}   "
             f"candidates tried = {res['n_candidates']}",
             f"selection: {res['selection_reason']}",
             f"mode = {res['mode']} ({res['mode_reason']});  max f = {res['f_max']:.4g}",
             f"blank: BG = {res['blank_mu']:.3f}, SD = {res['blank_sd']:.3f};  {res['blank_desc']}",
             "",
             f"{'stage':<6}{'C (fM)':>10}{'N':>7}{'n_pos':>7}{'f':>9}{'S_abs':>12}{'y':>12}{'sd_y':>10}{'reg cc':>9}"]
        for s in stage_summaries(res):
            sd = f"{s['sd_y']:.4g}" if s["sd_y"] is not None else "-"
            cc = f"{s['registration_cc']:.3f}" if s["registration_cc"] is not None and \
                np.isfinite(s["registration_cc"]) else "-"
            L.append(f"{s['stage']:<6}{s['C_fM']:>10.4g}{s['N']:>7}{s['n_positive']:>7}"
                     f"{s['f']:>9.4f}{s['S_abs']:>12.5g}{s['y']:>12.5g}{sd:>10}{cc:>9}")
        if res.get("containment"):
            L.append("registration diagnostics (shift px, rotation deg, ECC cc, positive-object containment):")
            for p, cq in zip(self._ctx["profiles"], res["containment"]):
                r = p["registration"]
                L.append(f"   {p['stage_label']}: shift = {r['shift_px']:.2f}, rot = {r['rotation_deg']:.3f}, "
                         f"cc = {r['cc']:.3f}, containment = {cq['ratio']:.3f} "
                         f"({cq['n_inside']} in / {cq['n_outside']} out)")
        L += ["",
              f"fit: a = {fit['a']:.5g}, b = {fit['b']:.5g}, x-int = {fit['x_intercept']:.5g}, "
              f"T = {fit['T']:.5g} fM, R2 = {fit['r2']:.4f}, n = {fit['n']}, df = {fit['df']}",
              f"     95% CI = [{fit['T_ci'][0]:.5g}, {fit['T_ci'][1]:.5g}], SE = {fit['se_x']:.4g}, "
              f"k = {fit['ci_k']:.3g}, dispersion = {fit['dispersion']:.3g}, basis: {fit['ci_basis']}",
              "",
              f"gate: coverage = {gate['coverage']}; endpoint = {gate['endpoint']}; "
              f"rule = {gate['required_rule']}"]
        for k, v in sorted(gate["gradients"].items(), key=lambda kv: int(kv[0][1:])):
            L.append(f"   {k} = {v:.6g}    {gate['gradient_labels'].get(k, '')}")
        for c in gate["checks"]:
            L.append(f"   {'OK  ' if c['pass'] else 'FAIL'} {c['name']:<10} {c['rule']:<24} {c['detail']}")
        L.append(f"   -> {gate['message']}")
        L.append(f"status: {res['status']}")
        if scan:
            L += ["", f"stability scan θ ∈ [{scan['theta_range'][0]}, {scan['theta_range'][1]}]: "
                      f"gate passes at {scan['n_pass']} thresholds; plateau = {scan['plateau']} "
                      f"(width {scan['plateau_width']}); T median (pass) = {scan['T_median_pass']:.5g}; "
                      f"T on plateau = [{scan['T_min_plateau']:.5g}, {scan['T_max_plateau']:.5g}], "
                      f"CV = {scan['T_cv_plateau']:.3%}" if np.isfinite(scan["T_cv_plateau"]) else
                      f"stability scan: plateau = {scan['plateau']} (width {scan['plateau_width']})"]
        if res.get("candidate_summary"):
            L += ["", "legacy search candidates (θ, T, pass, violation):"]
            for c in res["candidate_summary"][:60]:
                Tc = c["T"]
                Ts = "-" if Tc is None or not np.isfinite(Tc) else f"{Tc:.5g}"
                L.append(f"   {c['threshold']:>4}  {Ts:>10}  {str(c['gate_pass']):<6} {c['violation']}")
            if len(res["candidate_summary"]) > 60:
                L.append(f"   ... {len(res['candidate_summary']) - 60} more in audit.json")
        L += ["", "QC: " + ("PASS" if not res["qc_errors"] else "; ".join(res["qc_errors"]))]
        if res.get("warnings"):
            L.append("warnings: " + "; ".join(res["warnings"]))
        L += ["", "fixed constants: " + json.dumps(AUDIT_CONFIG)]
        self.details.config(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", "\n".join(L))
        self.details.config(state="disabled")

    # ---------------- export / clear / help ----------------
    def export_audit(self):
        if self.last_result is None:
            messagebox.showwarning("No result", "Run the analysis first.")
            return
        folder = filedialog.askdirectory(title="Select audit-export folder")
        if not folder:
            return
        res = self.last_result
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(folder) / f"GRASP_audit_{stamp}"
        base.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for p in ([res["blank_path"]] + res["fluor_paths"] +
                  [x for x in res["bf_paths"] if x] + ([res["blank_bf_path"]] if res["blank_bf_path"] else [])):
            hashes[str(p)] = file_sha256(p)
        clean = {k: v for k, v in res.items() if k not in ("stages",)}
        clean["stages"] = stage_summaries(res)
        audit = {"software": {**AUDIT_CONFIG, "python": platform.python_version(),
                              "opencv": cv2.__version__, "numpy": np.__version__},
                 "input_sha256": hashes, "result": clean,
                 "stability_scan_rows": (self.last_scan["rows"] if self.last_scan else None)}
        with open(base / "audit.json", "w", encoding="utf-8") as fh:
            json.dump(audit, fh, indent=2, ensure_ascii=False, default=_json_default)
        if res["paired_registration"]:
            fields = ["trap_id", "positive", "overlap", "positive_pixels",
                      "integrated_intensity", "mean_intensity"]
            for k, p in enumerate(self._ctx["profiles"]):
                with open(base / f"stage_{k}_trap_assignments.csv", "w", newline="",
                          encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=fields)
                    w.writeheader()
                    w.writerows(trap_rows(p, res["threshold"]))
            for (kind, i), arr in self.overlay_arrays.items():
                Image.fromarray(arr).save(base / f"stage{i}_{kind}_overlay.png")
        self.fig.savefig(base / "standard_addition_fit.png", dpi=150)
        if self.last_scan:
            self.fig2.savefig(base / "threshold_scan.png", dpi=150)
        messagebox.showinfo("Export complete", f"Audit exported to:\n{base}")
        self.set_status(f"Audit exported to {base}")

    def clear_all(self):
        self.last_result = self.last_scan = None
        _IMG_CACHE.clear(); _BF_CACHE.clear(); _PROFILE_CACHE.clear()
        self.overlay_arrays.clear(); self._photos.clear()
        for i in range(5):
            self.fluor_paths[i] = self.bf_paths[i] = None
            self.fluor_name[i].config(text="—", foreground="#6B7280")
            self.bf_name[i].config(text="—", foreground="#6B7280")
            self.fluor_tip[i].set_text(""); self.bf_tip[i].set_text("")
            self.readout[i].config(text="")
            for lab in (self.thumb_fluor[i], self.thumb_bf[i]):
                lab.config(image="", width=self.THUMB[0] // 7, height=self.THUMB[1] // 14)
                lab.image = None
        for v in self.c_vars[1:]:
            v.set("")
        self.T_label.config(text="T = —")
        self.ci_label.config(text="")
        self.banner.config(text="Load images to begin.", bg="#E5E7EB", fg="#111827")
        for w in (self.info_label, self.gate_label, self.qc_label):
            w.config(text="")
        self.theta_note.config(text="Load the blank to obtain BG, SD and θ0.")
        self.details.config(state="normal"); self.details.delete("1.0", "end")
        self.details.config(state="disabled")
        self.ax.clear(); self.canvas.draw_idle()
        self.ax2.clear(); self.canvas2.draw_idle()
        self._refresh_controls()
        self.set_status("Cleared.")

    def show_rules(self):
        messagebox.showinfo("Method rules", f"{APP_NAME} v{APP_VERSION} — {METHOD_ID}\n\n{RULES_TEXT}")


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", max(float(root.tk.call("tk", "scaling")), 1.0))
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
