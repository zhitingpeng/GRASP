# ===========================================================================
# =============================================================================
# Core rules:
# 1) S0 is the absolute PBS-blank signal.
# 2) Initial threshold theta0 = blank background mean + 3 SD.
# 3) Bright-field defines occupied capture-trap ROIs. Fluorescence images are
#    ECC-registered to bright-field; one trap contributes at most one event.
# 4) Fluorescence objects outside occupied-trap ROIs are never counted.
# 5) One common threshold is selected for the complete standard-addition series
#    by a deterministic constrained search, not by sample-specific tuning.
# 6) Digital mode uses f=n_positive/N_valid and lambda=-ln(1-f).
#    Fluorescence-only input forces analog mode.
# 7) T+C0 and T+C1 support a two-point estimate. The full quantitative-window
#    gate requires T+C2. If the available gradient chain fails, T is marked
#    non-reportable: OUT OF QUANTITATIVE WINDOW - DILUTE AND RE-MEASURE.
# 8) Processing panel exposes threshold controls only. Registration/trap-QC
#    parameters are fixed, versioned constants and exported in the audit log.
# =============================================================================

import csv
import json
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path
import warnings
import cv2
import numpy as np
from tkinter import Tk, Label, Scale, StringVar, BooleanVar, IntVar
from tkinter import HORIZONTAL, N, S, E, W, LEFT
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

warnings.simplefilter("ignore", np.RankWarning)

APP_NAME = "GRASP dB-LFIA"
APP_VERSION = "7.7.0-compact"
METHOD_ID = "GRASP-gate-endpoint-v1"
F_GATE = 0.70
BG_SD_FACTOR = 3.0
THRESHOLD_SEARCH_RADIUS = 255
THRESHOLD_SEARCH_STEP = 1
CHAIN_REL_TOL = 1e-6
CHAIN_ABS_TOL = 1e-12
ECC_MIN_CC = 0.70
ECC_MAX_ITER = 250
ECC_EPS = 1e-6
ROI_DILATION_PX = 2
TRAP_MIN_AREA_PX = 12
TRAP_MAX_AREA_PX = 5000
TRAP_BORDER_MARGIN_PX = 3
MIN_VALID_TRAPS = 20
MIN_ROI_OVERLAP = 0.25
MIN_POSITIVE_PIXELS = 1
DEBOUNCE_MS = 180
EPS = 1e-12

STATUS_IN = "IN QUANTITATIVE WINDOW"
STATUS_PARTIAL = "PROVISIONAL TWO-POINT ESTIMATE - FULL WINDOW GATE NOT EVALUABLE"
STATUS_OUT = "OUT OF QUANTITATIVE WINDOW - DILUTE AND RE-MEASURE"
STATUS_FAIL = "ANALYSIS FAILED"

AUDIT_CONFIG = {
    "app": APP_NAME, "version": APP_VERSION, "method_id": METHOD_ID,
    "f_gate": F_GATE, "bg_sd_factor": BG_SD_FACTOR,
    "threshold_search_radius": THRESHOLD_SEARCH_RADIUS,
    "threshold_search_step": THRESHOLD_SEARCH_STEP,
    "chain_rel_tol": CHAIN_REL_TOL, "chain_abs_tol": CHAIN_ABS_TOL,
    "ecc_min_cc": ECC_MIN_CC, "ecc_max_iter": ECC_MAX_ITER,
    "ecc_eps": ECC_EPS, "roi_dilation_px": ROI_DILATION_PX,
    "trap_min_area_px": TRAP_MIN_AREA_PX, "trap_max_area_px": TRAP_MAX_AREA_PX,
    "trap_border_margin_px": TRAP_BORDER_MARGIN_PX,
    "min_valid_traps": MIN_VALID_TRAPS, "min_roi_overlap": MIN_ROI_OVERLAP,
    "min_positive_pixels": MIN_POSITIVE_PIXELS,
}

_IMG_CACHE = {}
_BF_CACHE = {}
_REG_CACHE = {}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cached_gray(path):
    if path not in _IMG_CACHE:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        _IMG_CACHE[path] = img
    return _IMG_CACHE[path]


def robust_blank_stats(gray):
    v = gray.astype(np.float64).ravel()
    keep = np.ones(v.shape, dtype=bool)
    for _ in range(8):
        x = v[keep]
        mu, sd = float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else 0.0
        new = (v >= mu - 3 * max(sd, 1e-6)) & (v <= mu + 3 * max(sd, 1e-6))
        if np.array_equal(new, keep):
            break
        keep = new
    x = v[keep]
    return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else 0.0


def initial_threshold(blank_path):
    mu, sd = robust_blank_stats(cached_gray(blank_path))
    return int(np.clip(round(mu + BG_SD_FACTOR * sd), 0, 255)), mu, sd


def segment_traps(bf_path):
    if bf_path in _BF_CACHE:
        return _BF_CACHE[bf_path]
    gray = cached_gray(bf_path)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    block = max(15, (min(gray.shape) // 20) | 1)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, 3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    h, w = gray.shape
    rois, rejected = [], {"area": 0, "border": 0, "shape": 0}
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < TRAP_MIN_AREA_PX or area > TRAP_MAX_AREA_PX:
            rejected["area"] += 1; continue
        if x < TRAP_BORDER_MARGIN_PX or y < TRAP_BORDER_MARGIN_PX or \
           x + bw >= w - TRAP_BORDER_MARGIN_PX or y + bh >= h - TRAP_BORDER_MARGIN_PX:
            rejected["border"] += 1; continue
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > 2.5:
            rejected["shape"] += 1; continue
        mask = labels == i
        if ROI_DILATION_PX > 0:
            mask = cv2.dilate(mask.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (2 * ROI_DILATION_PX + 1,) * 2)) > 0
        rois.append({"id": len(rois) + 1, "mask": mask,
                     "centroid": [float(centroids[i][0]), float(centroids[i][1])],
                     "area": int(area), "bbox": [int(x), int(y), int(bw), int(bh)]})
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for r in rois:
        x, y, bw, bh = r["bbox"]
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
    out = {"gray": gray, "rois": rois, "N": len(rois),
           "rejected": rejected, "preview": overlay}
    _BF_CACHE[bf_path] = out
    return out


def register_fluorescence(fluor_path, bf_path):
    key = (fluor_path, bf_path)
    if key in _REG_CACHE:
        return _REG_CACHE[key]
    moving = cached_gray(fluor_path)
    fixed = cached_gray(bf_path)
    if moving.shape != fixed.shape:
        moving = cv2.resize(moving, (fixed.shape[1], fixed.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    f = cv2.normalize(cv2.GaussianBlur(fixed, (5, 5), 0).astype(np.float32),
                      None, 0, 1, cv2.NORM_MINMAX)
    m = cv2.normalize(cv2.GaussianBlur(moving, (5, 5), 0).astype(np.float32),
                      None, 0, 1, cv2.NORM_MINMAX)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                ECC_MAX_ITER, ECC_EPS)
    try:
        cc, warp = cv2.findTransformECC(f, m, warp, cv2.MOTION_EUCLIDEAN,
                                        criteria, None, 3)
        registered = cv2.warpAffine(moving, warp, (fixed.shape[1], fixed.shape[0]),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ok = bool(np.isfinite(cc) and cc >= ECC_MIN_CC)
        error = "" if ok else f"ECC correlation {cc:.4f} below {ECC_MIN_CC:.2f}"
    except cv2.error as e:
        cc, ok, error = np.nan, False, str(e).split("\n")[0]
        registered = moving
    out = {"image": registered, "warp": warp.tolist(), "cc": float(cc),
           "ok": ok, "error": error}
    _REG_CACHE[key] = out
    return out


def global_fluor_metrics(fluor_path, threshold):
    gray = cached_gray(fluor_path)
    pos = gray > int(threshold)
    return {"signal": float(gray.sum()), "positive_pixels": int(pos.sum()),
            "positive_area": pos}


def trap_metrics(fluor_path, bf_path, threshold):
    bf = segment_traps(bf_path)
    reg = register_fluorescence(fluor_path, bf_path)
    img = reg["image"]
    positive = img > int(threshold)
    rows, n_pos, assigned_pixels = [], 0, 0
    for roi in bf["rois"]:
        mask = roi["mask"]
        roi_n = int(mask.sum())
        pos_n = int(np.count_nonzero(positive & mask))
        overlap = pos_n / max(roi_n, 1)
        is_pos = pos_n >= MIN_POSITIVE_PIXELS and overlap >= MIN_ROI_OVERLAP
        n_pos += int(is_pos)
        assigned_pixels += pos_n
        vals = img[mask]
        rows.append({"trap_id": roi["id"], "positive": bool(is_pos),
                     "overlap": float(overlap), "positive_pixels": pos_n,
                     "integrated_intensity": float(vals.sum()),
                     "mean_intensity": float(vals.mean()) if vals.size else 0.0})
    union = np.zeros_like(img, dtype=bool)
    for roi in bf["rois"]:
        union |= roi["mask"]
    unassigned = int(np.count_nonzero(positive & ~union))
    preview = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for row, roi in zip(rows, bf["rois"]):
        x, y, bw, bh = roi["bbox"]
        color = (255, 0, 0) if row["positive"] else (0, 255, 0)
        cv2.rectangle(preview, (x, y), (x + bw, y + bh), color, 1)
    N = bf["N"]
    return {"N": N, "n_positive": n_pos,
            "f": n_pos / N if N else np.nan,
            "trap_rows": rows, "unassigned_positive_pixels": unassigned,
            "assigned_positive_pixels": assigned_pixels,
            "signal": float(img[union].sum()) if N else np.nan,
            "registration": reg, "trap_qc": bf["rejected"], "preview": preview}


def safe_lambda(f):
    if not np.isfinite(f):
        return np.nan
    return float(-np.log(1 - np.clip(f, 1e-9, 1 - 1e-9)))


def fit_standard_addition(C, y):
    """y = a*C + b; x_intercept = -b/a; T = |x_intercept|."""
    C, y = np.asarray(C, float), np.asarray(y, float)
    m = np.isfinite(C) & np.isfinite(y)
    C, y = C[m], y[m]
    empty = {"a": np.nan, "b": np.nan, "x_intercept": np.nan,
             "T": np.nan, "r2": np.nan, "intercept_sign_ok": False}
    if len(C) < 2 or len(np.unique(C)) < 2:
        return empty
    if len(C) == 2:
        a = (y[1] - y[0]) / (C[1] - C[0]); b = y[0] - a * C[0]; r2 = np.nan
    else:
        a, b = np.polyfit(C, y, 1)
        ssr = float(np.sum((y - (a * C + b)) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 if sst < EPS else 1.0 - ssr / sst
    if not np.isfinite(a) or abs(a) <= EPS:
        return empty
    xi = float(-b / a)
    return {"a": float(a), "b": float(b), "x_intercept": xi,
            "T": abs(xi), "r2": float(r2), "intercept_sign_ok": bool(xi < 0)}


def gradient_gate(S0, S, C, T):
    """
    Endpoint rule. g1 is always the left-hand side; the right-hand side is the
    LAST gradient the loaded data can support, and it extends as stages are added:

      blank + T+C0 + T+C1                  -> g1 <= g2
        g1 = (S_t0 - S0)/T,  g2 = (S_t1 - S0)/(T + C1)
      blank + T+C0 + T+C1 + T+C2           -> g1 <= g3
        g3 = (S_t2 - S_t1)/(C2 - C1)
      blank + T+C0 + T+C1 + T+C2 + T+C3    -> g1 <= g4
        g4 = (S_t3 - S_t2)/(C3 - C2)

    All intermediate gradients are still reported for inspection, but the
    pass/fail decision uses g1 versus the final endpoint gradient.
    """
    out = {"gradients": {}, "checks": [], "pass": None, "coverage": "none",
           "required_rule": "not evaluable", "message": "",
           "n_evaluable": 0, "endpoint": "", "violation": np.inf}
    if not np.isfinite(T) or T <= 0:
        out["message"] = "T not positive; gate undefined"
        return out

    g, labels = {}, {}
    if len(S) >= 1 and T > EPS:
        g["g1"] = (S[0] - S0) / T
        labels["g1"] = "(S_t0-S_0)/T"
    if len(S) >= 2 and len(C) >= 2 and (T + C[1]) > EPS:
        g["g2"] = (S[1] - S0) / (T + C[1])
        labels["g2"] = "(S_t1-S_0)/(T+C_1)"
    # incremental span gradients for each additional spiked stage
    for j in range(2, len(S)):
        if j < len(C) and (C[j] - C[j - 1]) > EPS:
            key = f"g{j + 1}"
            g[key] = (S[j] - S[j - 1]) / (C[j] - C[j - 1])
            labels[key] = f"(S_t{j}-S_t{j-1})/(C_{j}-C_{j-1})"
    out["gradients"] = {k: float(v) for k, v in g.items()}
    out["gradient_labels"] = labels

    order = [k for k in sorted(g, key=lambda s: int(s[1:]))]
    if len(order) < 2:
        out["message"] = "no inequality evaluable"
        return out

    endpoint = order[-1]
    out["endpoint"] = endpoint
    lv, rv = g["g1"], g[endpoint]
    scale = max(abs(lv), abs(rv), 1.0)
    tol = CHAIN_ABS_TOL + CHAIN_REL_TOL * scale
    ok = bool(lv <= rv + tol)
    out["checks"] = [{"left": "g1", "right": endpoint,
                      "left_value": float(lv), "right_value": float(rv),
                      "pass": ok}]
    out["n_evaluable"] = 1
    out["pass"] = ok
    out["violation"] = 0.0 if ok else float((lv - rv) / scale)
    out["required_rule"] = f"g1 <= {endpoint}"
    out["rule_expanded"] = f"{labels['g1']} <= {labels[endpoint]}"
    out["coverage"] = "full" if len(order) >= 3 else "two_point"
    out["message"] = (f"{out['rule_expanded']} satisfied" if ok else
                      f"g1 > {endpoint} ({lv:.5g} > {rv:.5g})")
    return out


def threshold_candidates(theta0, manual_theta=None):
    """Full 0-255 sweep visited outward from theta0 = BG + 3 SD."""
    if manual_theta is not None:
        return [int(np.clip(manual_theta, 0, 255))]
    lo = max(0, theta0 - THRESHOLD_SEARCH_RADIUS)
    hi = min(255, theta0 + THRESHOLD_SEARCH_RADIUS)
    vals = list(range(lo, hi + 1, THRESHOLD_SEARCH_STEP))
    return sorted(vals, key=lambda t: (abs(t - theta0), t))


def evaluate_threshold(blank_path, fluor_paths, bf_paths, C, threshold,
                       auto_mode, manual_mode):
    blank_global = global_fluor_metrics(blank_path, threshold)
    paired = all(bf_paths[i] for i in range(len(fluor_paths)))
    stages, qc_errors = [], []
    if paired:
        for fp, bp in zip(fluor_paths, bf_paths):
            m = trap_metrics(fp, bp, threshold)
            stages.append(m)
            if not m["registration"]["ok"]:
                qc_errors.append(m["registration"]["error"])
            if m["N"] < MIN_VALID_TRAPS:
                qc_errors.append(f"Only {m['N']} valid traps; minimum is {MIN_VALID_TRAPS}")
        f0 = stages[0]["f"]
        if auto_mode:
            mode = "analog" if f0 > F_GATE else "digital"
        else:
            mode = manual_mode
        if mode == "digital":
            y = np.array([safe_lambda(m["f"]) for m in stages])
            S0 = 0.0
            S_abs = y.copy()
        else:
            blank_signal = blank_global["signal"]
            S_abs = np.array([m["signal"] for m in stages], float)
            y = S_abs - blank_signal
            S0 = blank_signal
    else:
        mode, f0 = "analog", np.nan
        stages = [global_fluor_metrics(p, threshold) for p in fluor_paths]
        S_abs = np.array([m["signal"] for m in stages], float)
        S0 = blank_global["signal"]
        y = S_abs - S0
    fit = fit_standard_addition(C, y)
    gate = gradient_gate(S0, list(S_abs), C, fit["T"])
    return {"threshold": int(threshold), "theta_distance": abs(threshold),
            "mode": mode, "paired_registration": paired, "f0": float(f0),
            "blank_signal": float(S0), "stage_signals": S_abs.tolist(),
            "y": y.tolist(), "fit": fit, "gate": gate,
            "stages": stages, "qc_errors": qc_errors}


def select_threshold(blank_path, fluor_paths, bf_paths, C, theta0,
                     auto_mode, manual_mode, manual_theta=None):
    """
    The threshold is auto-adjusted until the endpoint inequality holds:
    g1 <= g2 with two stages, g1 <= g3 with three, g1 <= g4 with four, and so on.
    The sweep starts at theta0 = BG + 3 SD and moves outward; it stops at the
    first threshold that satisfies the rule with valid QC and a negative
    x-intercept. If no threshold in 0-255 works, the smallest-violation
    candidate is returned and flagged non-reportable.
    """
    evaluations, solved = [], None
    for theta in threshold_candidates(theta0, manual_theta):
        try:
            ev = evaluate_threshold(blank_path, fluor_paths, bf_paths, C, theta,
                                    auto_mode, manual_mode)
        except Exception as exc:
            evaluations.append({"threshold": theta, "error": str(exc),
                                "theta_distance": abs(theta - theta0),
                                "qc_errors": [str(exc)]})
            continue
        ev["theta_distance"] = abs(theta - theta0)
        evaluations.append(ev)
        fit, gate = ev["fit"], ev["gate"]
        feasible = (not ev["qc_errors"] and np.isfinite(fit["T"]) and fit["T"] > 0
                    and fit["intercept_sign_ok"] and gate["n_evaluable"] >= 1)
        if feasible and gate["pass"] is True:
            solved = ev
            break

    if solved is not None:
        chosen = solved
        chosen["gate_solved"] = True
        chosen["selection_reason"] = (
            "Threshold auto-adjusted until "
            f"{chosen['gate']['required_rule']} held; nearest such threshold "
            "to theta0 = BG + 3 SD")
    else:
        scored = [e for e in evaluations if "fit" in e and e["gate"]["n_evaluable"] >= 1]
        if not scored:
            scored = [e for e in evaluations if "fit" in e]
        if not scored:
            raise RuntimeError("No threshold candidate could be evaluated")

        def rank(e):
            fit, gate = e["fit"], e["gate"]
            viol = gate.get("violation", np.inf)
            viol = viol if np.isfinite(viol) else 1e12
            return (1 if e.get("qc_errors") else 0,
                    0 if fit.get("intercept_sign_ok") else 1,
                    viol, e["theta_distance"], e["threshold"])

        chosen = min(scored, key=rank)
        chosen["gate_solved"] = False
        chosen["selection_reason"] = (
            "No threshold in 0-255 satisfied the endpoint inequality; "
            "minimum-violation candidate reported for diagnosis only")

    chosen["theta0"] = int(theta0)
    chosen["n_candidates"] = len(evaluations)
    chosen["candidate_summary"] = [{
        "threshold": e["threshold"], "distance": e["theta_distance"],
        "T": e.get("fit", {}).get("T"),
        "x_intercept": e.get("fit", {}).get("x_intercept"),
        "gate_rule": e.get("gate", {}).get("required_rule"),
        "gate_pass": e.get("gate", {}).get("pass"),
        "violation": e.get("gate", {}).get("violation"),
        "qc_errors": e.get("qc_errors", []), "error": e.get("error", "")
    } for e in evaluations]
    return chosen


fluor_paths = [None] * 5   # blank, T+C0, T+C1, T+C2, T+C3
bf_paths = [None] * 4      # T+C0 .. T+C3
fluor_info, fluor_preview = [], []
bf_info, bf_preview = [], []
last_result = None
_debounce_id = None


def schedule_recalc(*_):
    global _debounce_id
    if _debounce_id is not None:
        try: root.after_cancel(_debounce_id)
        except Exception: pass
    _debounce_id = root.after(DEBOUNCE_MS, calculate)


def select_image(target, idx):
    path = filedialog.askopenfilename(filetypes=[
        ("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All", "*.*")])
    if not path: return
    if target == "fluor":
        fluor_paths[idx] = path
        fluor_info[idx].config(text=Path(path).name[:20])
        if idx == 0:
            try:
                t, mu, sd = initial_threshold(path)
                threshold_scale.set(t)
                threshold_note.config(text=f"BG={mu:.1f} SD={sd:.2f} theta0={t}")
            except Exception as e:
                threshold_note.config(text=f"Threshold error: {e}")
    else:
        bf_paths[idx] = path
        bf_info[idx].config(text=Path(path).name[:20])
    refresh_mode_button(); schedule_recalc()


def clear_bf(idx):
    bf_paths[idx] = None
    bf_info[idx].config(text="none")
    bf_preview[idx].config(image=""); bf_preview[idx].image = None
    refresh_mode_button(); schedule_recalc()


def loaded_stage_count():
    n = 0
    for i in range(1, 5):
        if fluor_paths[i] is None: break
        n += 1
    return n


def read_concentrations(n):
    vals = [0.0]
    for var in (c1_var, c2_var, c3_var)[:max(0, n - 1)]:
        vals.append(float(var.get().strip()))
    if len(vals) != n: raise ValueError("Enter all Ci values for loaded stages")
    if any(vals[i] <= vals[i-1] for i in range(1, len(vals))):
        raise ValueError("Require 0 = C0 < C1 < C2 < C3")
    return vals


def all_loaded_bf(n):
    return n > 0 and all(bf_paths[i] is not None for i in range(n))


def refresh_mode_button():
    n = loaded_stage_count()
    if not all_loaded_bf(n):
        mode_button.config(text="ANALOG-ONLY (no paired BF)", state="disabled")
    else:
        mode_button.config(state="normal")
        if auto_mode.get():
            mode_button.config(text="AUTO (f0<=0.7 digital)")
        else:
            mode_button.config(text=f"MANUAL: {manual_mode.get().upper()}")


def toggle_mode():
    if auto_mode.get():
        auto_mode.set(False); manual_mode.set("digital")
    elif manual_mode.get() == "digital":
        manual_mode.set("analog")
    else:
        auto_mode.set(True)
    refresh_mode_button(); schedule_recalc()


def set_preview(label, arr, size=(145, 52)):
    if arr.ndim == 2: img = Image.fromarray(arr)
    else: img = Image.fromarray(arr.astype(np.uint8))
    img.thumbnail(size)
    tkimg = ImageTk.PhotoImage(img)
    label.config(image=tkimg); label.image = tkimg


def status_from_result(res):
    fit, gate = res["fit"], res["gate"]
    if res["qc_errors"] or not np.isfinite(fit["T"]) or not fit["intercept_sign_ok"]:
        return STATUS_FAIL
    if gate["pass"] is not True:
        return STATUS_OUT
    return STATUS_IN if gate["coverage"] == "full" else STATUS_PARTIAL


def calculate(*_):
    global last_result
    n = loaded_stage_count()
    if fluor_paths[0] is None or n < 2:
        result_label.config(text="Load PBS blank + T+C0 + T+C1 (minimum).")
        return
    try:
        C = read_concentrations(n)
        theta0, mu, sd = initial_threshold(fluor_paths[0])
        manual_theta = int(threshold_scale.get()) if manual_threshold.get() else None
        chosen = select_threshold(fluor_paths[0], fluor_paths[1:1+n], bf_paths[:n], C,
                                  theta0, auto_mode.get(), manual_mode.get(), manual_theta)
        chosen.update({"C": C, "blank_path": fluor_paths[0],
                       "fluor_paths": fluor_paths[1:1+n], "bf_paths": bf_paths[:n],
                       "blank_mu": mu, "blank_sd": sd,
                       "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                       "status": ""})
        chosen["status"] = status_from_result(chosen)
        last_result = chosen
    except Exception as e:
        result_label.config(text=f"Analysis error: {e}", foreground="#B91C1C")
        return

    threshold_scale.set(chosen["threshold"])
    threshold_note.config(
        text=f"theta0={theta0} -> theta={chosen['threshold']} "
             f"({chosen['n_candidates']} tried); "
             f"{'rule satisfied' if chosen.get('gate_solved') else 'NO solution in 0-255'}")
    fit, gate = chosen["fit"], chosen["gate"]
    r2 = "n/a (2-point)" if not np.isfinite(fit["r2"]) else f"{fit['r2']:.4f}"
    warn = "" if fit["intercept_sign_ok"] else " [invalid +x intercept]"
    result_label.config(text=f"T={fit['T']:.5g} fM{warn}\n"
                             f"x-int={fit['x_intercept']:.5g}; R2={r2}; "
                             f"mode={chosen['mode']}")
    status = chosen["status"]
    color = "#15803D" if status == STATUS_IN else ("#B91C1C" if status in (STATUS_OUT, STATUS_FAIL) else "#B45309")
    status_label.config(text=status + (f"\n{gate['message']}" if gate['message'] and gate['message'] != status else ""),
                        foreground=color)
    grads = "\n".join(f"{k}={v:.5g}" for k, v in
                      sorted(gate["gradients"].items(), key=lambda kv: int(kv[0][1:])))
    checks = "\n".join(f"{'OK' if x['pass'] else 'FAIL'}: {x['left']}<={x['right']} "
                       f"({x['left_value']:.4g} vs {x['right_value']:.4g})"
                       for x in gate["checks"])
    chain_label.config(text=f"rule: {gate.get('rule_expanded', gate['required_rule'])}\n"
                            + grads + ("\n" + checks if checks else ""))
    qc_label.config(text=("QC: PASS" if not chosen["qc_errors"] else
                          "QC: " + "; ".join(chosen["qc_errors"])))

    for i, stage in enumerate(chosen["stages"]):
        if chosen["paired_registration"]:
            set_preview(fluor_preview[i+1], stage["preview"])
            fluor_info[i+1].config(text=f"{stage['n_positive']}/{stage['N']} pos "
                                         f"f={stage['f']:.3f}")
            bf = segment_traps(bf_paths[i])
            set_preview(bf_preview[i], bf["preview"])
            bf_info[i].config(text=f"N={bf['N']} rej={sum(bf['rejected'].values())}")
        else:
            img = cached_gray(fluor_paths[i+1])
            set_preview(fluor_preview[i+1], img)
            fluor_info[i+1].config(text=f"S={stage['signal']:.3e}")
    set_preview(fluor_preview[0], cached_gray(fluor_paths[0]))
    fluor_info[0].config(text=f"S0={chosen['blank_signal']:.3e}")
    plot_result(chosen)


def plot_result(res):
    ax.clear()
    C = np.asarray(res["C"]); y = np.asarray(res["y"])
    fit = res["fit"]
    ax.scatter(C, y, color="#2563EB", s=55, label="standard-addition points")
    if np.isfinite(fit["a"]):
        lo = min(-1.2 * abs(fit["T"]), -1.0); hi = max(C) * 1.2 + EPS
        x = np.linspace(lo, hi, 200)
        ax.plot(x, fit["a"] * x + fit["b"], color="#2563EB")
    if np.isfinite(fit["x_intercept"]):
        ax.scatter([fit["x_intercept"]], [0], marker="*", s=120,
                   color="#16A34A", zorder=6,
                   label=f"x-int={fit['x_intercept']:.4g} -> T={fit['T']:.4g} fM")
    ax.axhline(0, color="black", lw=.7, ls="--")
    ax.axvline(0, color="grey", lw=.7, ls="--")
    ax.set_xlabel("Added concentration Ci (fM)")
    ax.set_ylabel("lambda" if res["mode"] == "digital" else "S_ti - S0 (a.u.)")
    ax.set_title(f"{res['mode']} | threshold={res['threshold']} | {res['status']}",
                 fontsize=9, color="#B91C1C" if res["status"] == STATUS_OUT else "black")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); canvas.draw_idle()


def export_audit():
    if last_result is None:
        messagebox.showwarning("No result", "Run the analysis first."); return
    folder = filedialog.askdirectory(title="Select audit-export folder")
    if not folder: return
    out = Path(folder)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out / f"GRASP_audit_{stamp}"
    base.mkdir(parents=True, exist_ok=True)
    res = dict(last_result)
    stages = res.pop("stages", [])
    hashes = {}
    for p in [res["blank_path"]] + res["fluor_paths"] + [x for x in res["bf_paths"] if x]:
        hashes[str(p)] = file_sha256(p)
    audit = {"software": {**AUDIT_CONFIG, "python": platform.python_version(),
                           "opencv": cv2.__version__, "numpy": np.__version__},
             "input_sha256": hashes, "result": res}
    with open(base / "audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    if res["paired_registration"]:
        for i, stage in enumerate(stages):
            with open(base / f"stage_{i}_trap_assignments.csv", "w", newline="", encoding="utf-8") as f:
                fields = ["trap_id", "positive", "overlap", "positive_pixels",
                          "integrated_intensity", "mean_intensity"]
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(stage["trap_rows"])
    messagebox.showinfo("Export complete", f"Audit exported to:\n{base}")


def clear_all():
    global last_result
    last_result = None; _IMG_CACHE.clear(); _BF_CACHE.clear(); _REG_CACHE.clear()
    for i in range(5):
        fluor_paths[i] = None; fluor_info[i].config(text="none")
        fluor_preview[i].config(image=""); fluor_preview[i].image = None
    for i in range(4):
        bf_paths[i] = None; bf_info[i].config(text="none")
        bf_preview[i].config(image=""); bf_preview[i].image = None
    for w in (result_label, status_label, chain_label, qc_label): w.config(text="")
    ax.clear(); canvas.draw_idle(); refresh_mode_button()


root = Tk()
root.title(f"{APP_NAME} v{APP_VERSION}")
PREVIEW_W, PREVIEW_H = 145, 52
CTRL_W = 262
root.minsize(940, 590)
root.resizable(True, True)
try:
    root.tk.call("tk", "scaling", min(float(root.tk.call("tk", "scaling")), 1.0))
except Exception:
    pass
style = ttk.Style(root)
for x in ("TLabel", "TButton", "TCheckbutton", "TEntry"):
    style.configure(x, font=("Arial", 8))
style.configure("Tiny.TButton", font=("Arial", 7), padding=(1, 0))
style.configure("TLabelframe.Label", font=("Arial", 8, "bold"))
main = ttk.Frame(root, padding=3)
main.grid(row=0, column=0, sticky=(N, S, E, W))
root.rowconfigure(0, weight=1); root.columnconfigure(0, weight=1)
main.rowconfigure(0, weight=1); main.columnconfigure(2, weight=1)

_ph = []
def blank_preview(parent):
    p = ImageTk.PhotoImage(Image.new("RGB", (PREVIEW_W, PREVIEW_H), (238, 238, 238)))
    _ph.append(p)
    w = Label(parent, image=p, bd=1, relief="sunken"); w.image = p
    return w

images = ttk.LabelFrame(main, text="Image inputs", padding=2)
images.grid(row=0, column=0, sticky=(N, W))
ttk.Label(images, text="Stage", width=10, font=("Arial", 8, "bold")).grid(row=0, column=0, sticky=W)
ttk.Label(images, text="Fluorescence", font=("Arial", 8, "bold")).grid(row=0, column=1, sticky=W)
ttk.Label(images, text="Bright-field", font=("Arial", 8, "bold")).grid(row=0, column=2, sticky=W, padx=(3, 0))
for i, title in enumerate(("S0 blank", "St0 T+C0", "St1 T+C1", "St2 T+C2", "St3 T+C3")):
    r = i + 1
    ttk.Label(images, text=title, width=10, font=("Arial", 8, "bold")).grid(row=r, column=0, sticky=(N, W))
    fc = ttk.Frame(images); fc.grid(row=r, column=1, sticky=W, pady=1)
    ttk.Button(fc, text="Fluor", width=6, style="Tiny.TButton",
               command=lambda i=i: select_image("fluor", i)).grid(row=0, column=0)
    fi = ttk.Label(fc, text="none", width=20, anchor=W, font=("Arial", 7))
    fi.grid(row=0, column=1); fluor_info.append(fi)
    fp = blank_preview(fc); fp.grid(row=1, column=0, columnspan=2); fluor_preview.append(fp)
    if i > 0:
        bc = ttk.Frame(images); bc.grid(row=r, column=2, sticky=W, padx=(3, 0), pady=1)
        ttk.Button(bc, text="BF", width=4, style="Tiny.TButton",
                   command=lambda j=i-1: select_image("bf", j)).grid(row=0, column=0)
        ttk.Button(bc, text="x", width=2, style="Tiny.TButton",
                   command=lambda j=i-1: clear_bf(j)).grid(row=0, column=1)
        bi = ttk.Label(bc, text="none", width=17, anchor=W, font=("Arial", 7))
        bi.grid(row=0, column=2); bf_info.append(bi)
        bp = blank_preview(bc); bp.grid(row=1, column=0, columnspan=3); bf_preview.append(bp)

controls = ttk.Frame(main, width=CTRL_W, padding=(4, 0, 2, 0))
controls.grid(row=0, column=1, sticky=(N, S, W)); controls.grid_propagate(False)

cf = ttk.LabelFrame(controls, text="Spikes (fM)", padding=3)
cf.grid(row=0, column=0, sticky=(E, W))
c1_var, c2_var, c3_var = StringVar(), StringVar(), StringVar()
for k, (nm, var) in enumerate((("C1", c1_var), ("C2", c2_var), ("C3", c3_var))):
    ttk.Label(cf, text=nm).grid(row=0, column=2 * k)
    e = ttk.Entry(cf, textvariable=var, width=6); e.grid(row=0, column=2 * k + 1)
    e.bind("<KeyRelease>", schedule_recalc)

pf = ttk.LabelFrame(controls, text="Threshold: auto-adjusted to satisfy rule", padding=3)
pf.grid(row=1, column=0, sticky=(E, W), pady=2)
threshold_scale = Scale(pf, from_=0, to=255, orient=HORIZONTAL, length=242,
                        width=10, font=("Arial", 7), command=schedule_recalc)
threshold_scale.set(20); threshold_scale.grid(row=0, column=0)
manual_threshold = BooleanVar(value=False)
ttk.Checkbutton(pf, text="Lock manual (off = auto-adjust)", variable=manual_threshold,
                command=schedule_recalc).grid(row=1, column=0, sticky=W)
threshold_note = ttk.Label(pf, text="theta0 = BG + 3 SD", wraplength=240, font=("Arial", 7))
threshold_note.grid(row=2, column=0, sticky=W)
ttk.Label(pf, text="2 stages: g1<=g2 | 3 stages: g1<=g3 | 4 stages: g1<=g4",
          wraplength=240, font=("Arial", 7), foreground="#555555").grid(row=3, column=0, sticky=W)

mf = ttk.LabelFrame(controls, text="Mode", padding=3)
mf.grid(row=2, column=0, sticky=(E, W))
auto_mode = BooleanVar(value=True); manual_mode = StringVar(value="digital")
mode_button = ttk.Button(mf, text="", command=toggle_mode, width=30)
mode_button.grid(row=0, column=0)

rf = ttk.LabelFrame(controls, text="Result: T = |x-intercept|", padding=3)
rf.grid(row=3, column=0, sticky=(E, W), pady=2)
result_label = ttk.Label(rf, text="", justify=LEFT, wraplength=240)
result_label.grid(row=0, column=0, sticky=W)
status_label = ttk.Label(rf, text="", justify=LEFT, wraplength=240, font=("Arial", 8, "bold"))
status_label.grid(row=1, column=0, sticky=W)

gf = ttk.LabelFrame(controls, text="Endpoint inequality", padding=3)
gf.grid(row=4, column=0, sticky=(E, W))
chain_label = ttk.Label(gf, text="", justify=LEFT, wraplength=240, font=("Consolas", 7))
chain_label.grid(row=0, column=0, sticky=W)

qf = ttk.LabelFrame(controls, text="QC", padding=3)
qf.grid(row=5, column=0, sticky=(E, W), pady=2)
qc_label = ttk.Label(qf, text="", wraplength=240, font=("Arial", 7))
qc_label.grid(row=0, column=0, sticky=W)

bt = ttk.Frame(controls); bt.grid(row=6, column=0, sticky=W)
ttk.Button(bt, text="Run", width=7, command=calculate).grid(row=0, column=0)
ttk.Button(bt, text="Audit", width=7, command=export_audit).grid(row=0, column=1)
ttk.Button(bt, text="Clear", width=7, command=clear_all).grid(row=0, column=2)

plot_frame = ttk.LabelFrame(main, text="Standard-addition fit", padding=2)
plot_frame.grid(row=0, column=2, sticky=(N, S, E, W), padx=(3, 0))
plot_frame.rowconfigure(0, weight=1); plot_frame.columnconfigure(0, weight=1)
fig = Figure(figsize=(4.7, 4.0), dpi=100); ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().grid(row=0, column=0, sticky=(N, S, E, W))

refresh_mode_button()
root.update_idletasks()
w = min(1420, max(940, images.winfo_reqwidth() + CTRL_W + 410))
h = min(890, max(590, images.winfo_reqheight() + 35))
root.geometry(f"{w}x{h}+20+20")
root.mainloop()
