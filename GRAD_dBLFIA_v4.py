# =============================================================================
#  GRASP_dBLFIA_fast_v2.py
#  Gated Regression with Adaptive Standard-addition Protocol
#  * Speed-optimised (imread/analysis cache, fast rolling-ball, debounce,
#    reused matplotlib canvas)
#  * NEW: bright-field images are optional. If none are provided, GRASP
#    falls back to analog-only mode and computes T from fluorescence alone.
# =============================================================================

import cv2
import numpy as np
from tkinter import (
    Tk, Label, Scale, StringVar, BooleanVar,
    HORIZONTAL, N, S, E, W, LEFT,
)
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings

warnings.simplefilter('ignore', np.RankWarning)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
F_GATE = 0.7
T_EST_4PT_THR = 100.0

RESPONSE_CONSTANTS = {
    "IgE":   {"k": 3.9e-2, "alpha": 1.1e3},
    "TNF-a": {"k": 2.4e-2, "alpha": 4.2e2},
    "NfL":   {"k": 1.2e-2, "alpha": 1.8e2},
}

DEBOUNCE_MS = 120

# -----------------------------------------------------------------------------
# Caches
# -----------------------------------------------------------------------------
_IMG_CACHE = {}       # path -> gray ndarray
_FLUOR_CACHE = {}     # (path, ft, bgr, mna, mxa) -> metrics
_BF_CACHE = {}        # (path, mna, mxa) -> metrics

def cached_imread(path):
    if path in _IMG_CACHE:
        return _IMG_CACHE[path]
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read: {path}")
    _IMG_CACHE[path] = gray
    return gray

# -----------------------------------------------------------------------------
# Image processing
# -----------------------------------------------------------------------------
def fast_rolling_ball(gray, radius=25):
    r = max(1, int(radius))
    scale = max(1, r // 4)
    if scale > 1:
        small = cv2.resize(gray, (gray.shape[1] // scale, gray.shape[0] // scale),
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray
    k = max(3, (r // scale) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_OPEN, kernel)
    if scale > 1:
        bg = cv2.resize(bg_small, (gray.shape[1], gray.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    else:
        bg = bg_small
    return bg

def detect_components(binary, min_area, max_area):
    num_lbl, cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_idx = np.where((areas >= min_area) & (areas <= max_area))[0] + 1
    out = np.zeros_like(binary)
    masks = []
    for i in keep_idx:
        m = cc == i
        out[m] = 255
        masks.append(m)
    return len(keep_idx), out, masks

def analyze_fluorescence(path, fluor_thresh, bg_radius, min_area, max_area):
    key = (path, int(fluor_thresh), int(bg_radius), int(min_area), int(max_area))
    if key in _FLUOR_CACHE:
        return _FLUOR_CACHE[key]
    gray = cached_imread(path)
    bg = fast_rolling_ball(gray, bg_radius)
    corr = cv2.subtract(gray, bg)
    blur = cv2.GaussianBlur(corr, (3, 3), 0)
    _, binary = cv2.threshold(blur, int(fluor_thresh), 255, cv2.THRESH_BINARY)
    n, filtered, masks = detect_components(binary, min_area, max_area)
    if masks:
        union = np.zeros_like(corr, dtype=bool)
        for m in masks:
            union |= m
        net_intensity = float(corr[union].sum())
    else:
        net_intensity = 0.0
    pw = 160
    preview = np.hstack([cv2.resize(g, (pw, pw), interpolation=cv2.INTER_AREA)
                         for g in (gray, corr, filtered)])
    metrics = {"n": n, "net_intensity": net_intensity, "preview": preview}
    _FLUOR_CACHE[key] = metrics
    if len(_FLUOR_CACHE) > 128:
        _FLUOR_CACHE.pop(next(iter(_FLUOR_CACHE)))
    return metrics

def analyze_brightfield(path, min_area, max_area):
    key = (path, int(min_area), int(max_area))
    if key in _BF_CACHE:
        return _BF_CACHE[key]
    gray = cached_imread(path)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    N, filtered, _ = detect_components(binary, min_area, max_area)
    pw = 200
    preview = np.hstack([cv2.resize(g, (pw, pw), interpolation=cv2.INTER_AREA)
                         for g in (gray, filtered)])
    metrics = {"N": N, "preview": preview}
    _BF_CACHE[key] = metrics
    if len(_BF_CACHE) > 64:
        _BF_CACHE.pop(next(iter(_BF_CACHE)))
    return metrics

# -----------------------------------------------------------------------------
# GRASP engine
# -----------------------------------------------------------------------------
def resolve_mode(f0, auto, manual, has_bf):
    """
    Mode-routing logic.
    If bright-field is unavailable → force analog regardless of auto/manual.
    """
    if not has_bf:
        return "analog"
    if auto:
        return "analog" if f0 > F_GATE else "digital"
    return manual

def fit_line(C, y):
    C = np.asarray(C, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(C) & np.isfinite(y)
    C, y = C[mask], y[mask]
    if len(C) < 2:
        return np.nan, np.nan, np.array([]), np.nan
    a, b = np.polyfit(C, y, 1)
    yf = a * C + b
    ss_r = float(np.sum((y - yf) ** 2))
    ss_t = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_t < 1e-30 else 1.0 - ss_r / ss_t
    return float(a), float(b), yf, r2

def safe_f(n_net, N):
    if N is None or N <= 0:
        return 0.0
    return float(min(max(n_net / N, 1e-9), 1 - 1e-9))

def run_grasp(fm_blank, fm_list, bm_list, auto, manual, analyte,
              user_C1=None, user_C2=None, use_4pt_if_allowed=False):
    """
    bm_list: list that may contain None entries. If all entries are None
             (or bm_list is None), the algorithm runs analog-only.
    """
    k = RESPONSE_CONSTANTS[analyte]["k"]
    alpha = RESPONSE_CONSTANTS[analyte]["alpha"]

    n_blank = fm_blank["n"]
    I_blank = fm_blank["net_intensity"]

    # Normalise bm_list
    if bm_list is None:
        bm_list = [None] * len(fm_list)
    has_bf_any = any(bm is not None for bm in bm_list)
    bm0 = bm_list[0] if len(bm_list) > 0 else None
    has_bf_stage1 = bm0 is not None

    # -------- Stage 1 --------
    n0_net = max(fm_list[0]["n"] - n_blank, 0)
    I0_net = max(fm_list[0]["net_intensity"] - I_blank, 0.0)
    N0 = bm0["N"] if has_bf_stage1 else None
    f0 = safe_f(n0_net, N0) if has_bf_stage1 else np.nan
    mode = resolve_mode(f0, auto, manual, has_bf_stage1)
    if mode == "digital":
        T_est = -np.log(1.0 - f0) / k
    else:
        T_est = I0_net / alpha

    # -------- Stage 2 --------
    C1 = float(user_C1) if user_C1 is not None else max(T_est, 1e-9)
    C2 = float(user_C2) if user_C2 is not None else 2.0 * C1
    four_pt = use_4pt_if_allowed and (T_est > T_EST_4PT_THR) and (len(fm_list) >= 4)
    C_list = [0.0, C1, C2, 4.0 * C1] if four_pt else [0.0, C1, C2]
    n = len(C_list)
    fm_use = fm_list[:n]
    bm_use = bm_list[:n] if bm_list is not None else [None] * n

    n_net = np.array([max(fm["n"] - n_blank, 0) for fm in fm_use], dtype=float)
    I_net = np.array([max(fm["net_intensity"] - I_blank, 0.0) for fm in fm_use], dtype=float)

    if mode == "digital":
        Ns = [bm["N"] if bm is not None else None for bm in bm_use]
        fs = [safe_f(nn, NN) for nn, NN in zip(n_net, Ns)]
        y = np.array([-np.log(1 - f) for f in fs])
    else:
        Ns = [bm["N"] if bm is not None else None for bm in bm_use]
        fs = [np.nan] * n
        y = I_net

    C_arr = np.array(C_list, dtype=float)
    a, b, yf, r2 = fit_line(C_arr, y)
    T = (-b / a) if (np.isfinite(a) and abs(a) > 1e-12) else np.nan

    return {"analyte": analyte, "mode": mode, "f0": f0, "T_est": T_est,
            "four_point": four_pt, "C": C_arr, "y": y,
            "ns_net": n_net.astype(int), "Ns": Ns, "fs": fs,
            "I_net": I_net, "n_blank": n_blank, "I_blank": I_blank,
            "a": a, "b": b, "r2": r2, "T": T, "k": k, "alpha": alpha,
            "has_bf": has_bf_any, "has_bf_stage1": has_bf_stage1}

# -----------------------------------------------------------------------------
# Runtime state + debounce
# -----------------------------------------------------------------------------
fluor_paths = [None] * 5   # 0 blank, 1..4 T+C0..C3
bf_paths    = [None] * 4
fluor_img_labels, bf_img_labels = [], []
fluor_info_labels, bf_info_labels = [], []

_debounce_id = None
def schedule_recalc(*_):
    global _debounce_id
    if _debounce_id is not None:
        try: root.after_cancel(_debounce_id)
        except Exception: pass
    _debounce_id = root.after(DEBOUNCE_MS, calculate_and_display)

# -----------------------------------------------------------------------------
# Reused matplotlib canvas
# -----------------------------------------------------------------------------
_fig = Figure(figsize=(6.2, 4.2), dpi=100)
_ax = _fig.add_subplot(111)
_canvas_holder = {"canvas": None}

def ensure_canvas():
    if _canvas_holder["canvas"] is None:
        c = FigureCanvasTkAgg(_fig, master=plot_frame)
        c.get_tk_widget().grid(row=0, column=0, sticky=(N, S, E, W))
        _canvas_holder["canvas"] = c
    return _canvas_holder["canvas"]

def plot_grasp(res):
    ensure_canvas()
    _ax.clear()
    C = res["C"]; y = res["y"]; mode = res["mode"]
    a = res["a"]; b = res["b"]; T = res["T"]; r2 = res["r2"]
    sig = "−ln(1 − f)" if mode == "digital" else "Net intensity (a.u.)"
    x_lo = min(-max(1.0, abs(T) * 1.3 if np.isfinite(T) else 1.0),
               float(np.min(C)) - 0.1 * max(1.0, float(np.max(C))))
    x_hi = float(np.max(C)) * 1.25 + 1e-9
    xl = np.linspace(x_lo, x_hi, 200)
    _ax.scatter(C, y, color="#2563EB", s=55, zorder=5, label="GRASP points")
    if np.isfinite(a):
        _ax.plot(xl, a * xl + b, color="#2563EB", lw=2.0, label="OLS fit")
    _ax.axhline(0, color="black", lw=0.7, ls="--")
    _ax.axvline(0, color="gray",  lw=0.8, ls="--")
    if np.isfinite(T):
        _ax.axvline(T, color="#16A34A", lw=1.4, ls=":", label=f"T = {T:.4f}")
        _ax.scatter([T], [0], color="#16A34A", marker="*", s=120, zorder=6)
    n_pts = 4 if res["four_point"] else 3
    tag = "analog-only" if not res["has_bf_stage1"] else mode
    _ax.set_title(f"GRASP | {res['analyte']} | {tag} | {n_pts}-pt | R²={r2:.4f}",
                  fontsize=10)
    _ax.set_xlabel("Added concentration Ci")
    _ax.set_ylabel(f"Signal yi ({sig})")
    _ax.grid(alpha=0.25); _ax.legend(fontsize=8)
    _fig.tight_layout()
    _canvas_holder["canvas"].draw_idle()

# -----------------------------------------------------------------------------
# Preview helpers
# -----------------------------------------------------------------------------
def show_fluor_preview(idx, metrics):
    img = Image.fromarray(metrics["preview"]); img.thumbnail((330, 112))
    tk = ImageTk.PhotoImage(img)
    fluor_img_labels[idx].config(image=tk); fluor_img_labels[idx].image = tk
    fluor_info_labels[idx].config(
        text=f"n={metrics['n']}  |  analog y={metrics['net_intensity']:.2e}")

def show_bf_preview(idx, metrics):
    img = Image.fromarray(metrics["preview"]); img.thumbnail((280, 112))
    tk = ImageTk.PhotoImage(img)
    bf_img_labels[idx].config(image=tk); bf_img_labels[idx].image = tk
    bf_info_labels[idx].config(text=f"N = {metrics['N']}")

def get_params():
    return dict(ft=int(fluor_thresh_sl.get()), bgr=int(bg_radius_sl.get()),
                mna=int(min_area_sl.get()), mxa=int(max_area_sl.get()))

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
def select_fluor(idx):
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All", "*.*")])
    if not path: return
    fluor_paths[idx] = path
    try:
        p = get_params()
        m = analyze_fluorescence(path, p["ft"], p["bgr"], p["mna"], p["mxa"])
        show_fluor_preview(idx, m)
        schedule_recalc()
    except Exception as e:
        fluor_info_labels[idx].config(text=f"Error: {e}")

def select_bf(idx):
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All", "*.*")])
    if not path: return
    bf_paths[idx] = path
    try:
        p = get_params()
        m = analyze_brightfield(path, p["mna"], p["mxa"])
        show_bf_preview(idx, m)
        schedule_recalc()
    except Exception as e:
        bf_info_labels[idx].config(text=f"Error: {e}")

def clear_bf(idx):
    bf_paths[idx] = None
    bf_info_labels[idx].config(text="No image (analog-only)")
    bf_img_labels[idx].config(image=''); bf_img_labels[idx].image = None
    schedule_recalc()

def needs_4pt(): return bool(use_4pt_var.get())
def required_channels(): return 4 if needs_4pt() else 3

def refresh_mode_btn():
    if auto_mode_var.get():
        mode_btn.config(text="Mode: AUTO (click → Manual Digital)", style="Auto.TButton")
    else:
        lbl = "Digital" if manual_mode_var.get() == "digital" else "Analog"
        mode_btn.config(text=f"Mode: MANUAL [{lbl}] (click → next)", style="Manual.TButton")

def toggle_mode():
    if auto_mode_var.get():
        auto_mode_var.set(False); manual_mode_var.set("digital")
    elif manual_mode_var.get() == "digital":
        manual_mode_var.set("analog")
    else:
        auto_mode_var.set(True)
    refresh_mode_btn(); schedule_recalc()

def delete_all():
    _IMG_CACHE.clear(); _FLUOR_CACHE.clear(); _BF_CACHE.clear()
    for i in range(5):
        fluor_paths[i] = None
        if i < len(fluor_info_labels):
            fluor_info_labels[i].config(text="No image selected")
            fluor_img_labels[i].config(image=''); fluor_img_labels[i].image = None
    for i in range(4):
        bf_paths[i] = None
        if i < len(bf_info_labels):
            bf_info_labels[i].config(text="No image (optional)")
            bf_img_labels[i].config(image=''); bf_img_labels[i].image = None
    for w in (stage1_lbl, counts_lbl, mode_lbl, slope_lbl, icpt_lbl, result_lbl):
        w.config(text="")
    if _canvas_holder["canvas"] is not None:
        _ax.clear(); _canvas_holder["canvas"].draw_idle()

def calculate_and_display(*_):
    need_n = required_channels()
    # Fluorescence is REQUIRED
    if fluor_paths[0] is None or any(fluor_paths[i] is None for i in range(1, 1 + need_n)):
        result_lbl.config(text="Fluorescence required: blank + T+C0..T+C2 (+ C3 if 4-pt).")
        return

    p = get_params()
    try:
        fm_blank = analyze_fluorescence(fluor_paths[0], p["ft"], p["bgr"], p["mna"], p["mxa"])
        fm_list = [analyze_fluorescence(fluor_paths[i], p["ft"], p["bgr"], p["mna"], p["mxa"])
                   for i in range(1, 1 + need_n)]
    except Exception as e:
        result_lbl.config(text=f"Fluor analysis error: {e}"); return

    # Bright-field is OPTIONAL
    bm_list = []
    bf_errors = []
    for i in range(need_n):
        if bf_paths[i] is None:
            bm_list.append(None)
        else:
            try:
                bm_list.append(analyze_brightfield(bf_paths[i], p["mna"], p["mxa"]))
            except Exception as e:
                bf_errors.append(f"BF#{i}: {e}")
                bm_list.append(None)

    C1_in = c1_var.get().strip(); C2_in = c2_var.get().strip()
    uC1 = None if C1_in in ("", "auto") else float(C1_in)
    uC2 = None if C2_in in ("", "auto") else float(C2_in)

    try:
        res = run_grasp(fm_blank, fm_list, bm_list,
                        auto=auto_mode_var.get(), manual=manual_mode_var.get(),
                        analyte=analyte_var.get(),
                        user_C1=uC1, user_C2=uC2,
                        use_4pt_if_allowed=needs_4pt())
    except Exception as e:
        result_lbl.config(text=f"GRASP error: {e}"); return

    # Status display
    if not res["has_bf_stage1"]:
        stage1_txt = (f"Stage 1 — [analog-only, no BF]  "
                      f"I0_net={res['I_net'][0]:.2e}  "
                      f"T_est={res['T_est']:.3f} fM  α={res['alpha']:.2e}")
    else:
        stage1_txt = (f"Stage 1 — f0={res['f0']:.4f}  mode={res['mode']}  "
                      f"T_est={res['T_est']:.3f} fM  k={res['k']:.2e}  α={res['alpha']:.2e}")
    stage1_lbl.config(text=stage1_txt)

    Ns_disp = ", ".join("—" if N is None else str(int(N)) for N in res["Ns"])
    counts_lbl.config(
        text=f"n_net=[{', '.join(map(str, res['ns_net']))}]  N=[{Ns_disp}]  "
             f"(blank n={res['n_blank']}, I={res['I_blank']:.1e})")
    mode_lbl.config(text=f"{'4-pt' if res['four_point'] else '3-pt'} OLS  "
                         f"mode={res['mode']}  C={np.round(res['C'], 4).tolist()}")
    slope_lbl.config(text=f"a = {res['a']:.5g}")
    icpt_lbl.config(text=f"b = {res['b']:.5g}")
    if not np.isfinite(res["T"]):
        result_lbl.config(text="Failed: slope near zero."); return
    msg = f"T = {res['T']:.4f} fM   R² = {res['r2']:.4f}"
    if bf_errors:
        msg += "  |  " + "; ".join(bf_errors)
    if not res["has_bf"]:
        msg += "  |  analog-only fallback (no BF supplied)"
    result_lbl.config(text=msg)

    show_fluor_preview(0, fm_blank)
    for i, m in enumerate(fm_list):
        show_fluor_preview(i + 1, m)
    for i, m in enumerate(bm_list):
        if m is not None:
            show_bf_preview(i, m)
    plot_grasp(res)

def auto_optimize_threshold():
    need_n = required_channels()
    if fluor_paths[0] is None or any(fluor_paths[i] is None for i in range(1, 1 + need_n)):
        messagebox.showwarning("Missing", "Select all fluorescence images first."); return
    p = get_params()
    best_t, best_r2 = fluor_thresh_sl.get(), -1.0
    for t in range(5, 251, 10):
        try:
            fm_blank = analyze_fluorescence(fluor_paths[0], t, p["bgr"], p["mna"], p["mxa"])
            fm_list = [analyze_fluorescence(fluor_paths[i], t, p["bgr"], p["mna"], p["mxa"])
                       for i in range(1, 1 + need_n)]
            bm_list = []
            for i in range(need_n):
                if bf_paths[i] is None:
                    bm_list.append(None)
                else:
                    bm_list.append(analyze_brightfield(bf_paths[i], p["mna"], p["mxa"]))
            C1_in = c1_var.get().strip(); C2_in = c2_var.get().strip()
            uC1 = None if C1_in in ("", "auto") else float(C1_in)
            uC2 = None if C2_in in ("", "auto") else float(C2_in)
            res = run_grasp(fm_blank, fm_list, bm_list,
                            auto=auto_mode_var.get(), manual=manual_mode_var.get(),
                            analyte=analyte_var.get(),
                            user_C1=uC1, user_C2=uC2,
                            use_4pt_if_allowed=needs_4pt())
            if np.isfinite(res["T"]) and res["r2"] > best_r2:
                best_r2, best_t = res["r2"], t
        except Exception:
            continue
    fluor_thresh_sl.set(best_t)
    result_lbl.config(text=f"Auto threshold = {best_t}  (R² = {best_r2:.4f})")
    calculate_and_display()

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
root = Tk()
root.title("GRASP / dB-LFIA  (v2 — analog-only fallback supported)")
root.resizable(True, True)

style = ttk.Style(root)
style.configure("Auto.TButton",   foreground="#1D4ED8", font=("", 9, "bold"))
style.configure("Manual.TButton", foreground="#B45309", font=("", 9, "bold"))

auto_mode_var   = BooleanVar(value=True)
manual_mode_var = StringVar(value="digital")
analyte_var     = StringVar(value="IgE")
use_4pt_var     = BooleanVar(value=False)

mf = ttk.Frame(root, padding="10 10 12 12")
mf.grid(row=0, column=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)

# Left: image panel
img_panel = ttk.Frame(mf)
img_panel.grid(row=0, column=0, sticky=(N, W, E, S), padx=(0, 12))

blank_fr = ttk.LabelFrame(img_panel, text=" ① Screening blank (PBS, fluorescence only) ",
                          padding="4 4 4 4")
blank_fr.grid(row=0, column=0, sticky=(W, E), pady=(0, 6))
ttk.Button(blank_fr, text="Fluorescence…", width=16,
           command=lambda: select_fluor(0)).grid(row=0, column=0, padx=4)
lbl0 = ttk.Label(blank_fr, text="No image selected", font=("", 8))
lbl0.grid(row=0, column=1, sticky=W); fluor_info_labels.append(lbl0)
il0 = Label(blank_fr, bd=1, relief="sunken")
il0.grid(row=1, column=0, columnspan=2, sticky=W, pady=(2, 0))
fluor_img_labels.append(il0)

PAIR_TITLES = [
    "② T + C0 (screening test)",
    "③ T + C1 (spike = T_est)",
    "④ T + C2 (spike = 2·T_est)",
    "⑤ T + C3 (optional 4-pt = 4·T_est)",
]
for k_idx, title in enumerate(PAIR_TITLES):
    pf = ttk.LabelFrame(img_panel, text=f" {title} ", padding="4 4 4 4")
    pf.grid(row=k_idx + 1, column=0, sticky=(W, E), pady=(0, 6))

    # Fluorescence
    fs = ttk.LabelFrame(pf, text=" Fluorescence  [required] ", padding="3 3 3 3")
    fs.grid(row=0, column=0, sticky=(N, W, E, S), padx=(0, 6))
    fi = k_idx + 1
    ttk.Button(fs, text="Select…", width=10, command=lambda fi=fi: select_fluor(fi)).grid(row=0, column=0, sticky=W)
    lblF = ttk.Label(fs, text="No image selected", font=("", 8))
    lblF.grid(row=0, column=1, sticky=W, padx=4); fluor_info_labels.append(lblF)
    ilF = Label(fs, bd=1, relief="sunken")
    ilF.grid(row=1, column=0, columnspan=2, sticky=W, pady=(2, 0))
    fluor_img_labels.append(ilF)

    # Bright-field (OPTIONAL)
    bs = ttk.LabelFrame(pf, text=" Bright-field  [OPTIONAL — analog-only if skipped] ",
                        padding="3 3 3 3")
    bs.grid(row=0, column=1, sticky=(N, W, E, S))
    bi = k_idx
    ttk.Button(bs, text="Select…", width=9,
               command=lambda bi=bi: select_bf(bi)).grid(row=0, column=0, sticky=W)
    ttk.Button(bs, text="Clear", width=6,
               command=lambda bi=bi: clear_bf(bi)).grid(row=0, column=1, sticky=W, padx=(2, 0))
    lblB = ttk.Label(bs, text="No image (optional)", font=("", 8))
    lblB.grid(row=0, column=2, sticky=W, padx=4); bf_info_labels.append(lblB)
    ilB = Label(bs, bd=1, relief="sunken")
    ilB.grid(row=1, column=0, columnspan=3, sticky=W, pady=(2, 0))
    bf_img_labels.append(ilB)

# Middle: controls
ctrl = ttk.Frame(mf); ctrl.grid(row=0, column=1, sticky=(N, W, E, S))

ana_fr = ttk.LabelFrame(ctrl, text=" Analyte ", padding="6 4")
ana_fr.grid(row=0, column=0, sticky=(W, E), pady=(0, 6))
ttk.Label(ana_fr, text="Analyte:").grid(row=0, column=0, sticky=E)
ttk.Combobox(ana_fr, textvariable=analyte_var, state="readonly",
             values=list(RESPONSE_CONSTANTS.keys()), width=10).grid(row=0, column=1, padx=4, sticky=W)

spike_fr = ttk.LabelFrame(ctrl, text=" Adaptive spikes (leave blank/auto → from T_est) ",
                          padding="6 4")
spike_fr.grid(row=1, column=0, sticky=(W, E), pady=(0, 6))
ttk.Label(spike_fr, text="C1 (fM):").grid(row=0, column=0, sticky=E)
c1_var = StringVar(value="auto")
ttk.Entry(spike_fr, textvariable=c1_var, width=10).grid(row=0, column=1, padx=4, sticky=W)
ttk.Label(spike_fr, text="C2 (fM):").grid(row=1, column=0, sticky=E)
c2_var = StringVar(value="auto")
ttk.Entry(spike_fr, textvariable=c2_var, width=10).grid(row=1, column=1, padx=4, sticky=W)
ttk.Checkbutton(spike_fr, text="Enable 4-pt when T_est > 100 fM",
                variable=use_4pt_var,
                command=schedule_recalc).grid(row=2, column=0, columnspan=2, sticky=W, pady=(3, 0))

proc_fr = ttk.LabelFrame(ctrl, text=" Processing parameters ", padding="6 4")
proc_fr.grid(row=2, column=0, sticky=(W, E), pady=(0, 6))

def mks(parent, text, row, lo, hi, init):
    ttk.Label(parent, text=text).grid(row=row, column=0, sticky=E)
    s = Scale(parent, from_=lo, to=hi, orient=HORIZONTAL,
              command=schedule_recalc, length=210)
    s.set(init); s.grid(row=row, column=1, padx=4, sticky=W)
    return s

fluor_thresh_sl = mks(proc_fr, "Fluor threshold:", 0, 0,   255, 20)
bg_radius_sl    = mks(proc_fr, "BG radius:",      1, 1,    80, 25)
min_area_sl     = mks(proc_fr, "Min area:",       2, 1,   200,  3)
max_area_sl     = mks(proc_fr, "Max area:",       3, 10, 10000, 5000)

mode_fr = ttk.LabelFrame(ctrl, text=" Signal mode ", padding="6 4")
mode_fr.grid(row=3, column=0, sticky=(W, E), pady=(0, 6))
mode_btn = ttk.Button(mode_fr, text="", command=toggle_mode, width=44)
mode_btn.grid(row=0, column=0, sticky=W, pady=(2, 3))
ttk.Label(mode_fr,
          text=f"AUTO: f0 > {F_GATE} → analog, else digital (requires BF)",
          font=("", 8)).grid(row=1, column=0, sticky=W)
ttk.Label(mode_fr,
          text="If no BF images are supplied, GRASP automatically runs analog-only.",
          font=("", 8, "italic")).grid(row=2, column=0, sticky=W)
refresh_mode_btn()

res_fr = ttk.LabelFrame(ctrl, text=" Results ", padding="6 4")
res_fr.grid(row=4, column=0, sticky=(W, E), pady=(0, 6))
stage1_lbl = ttk.Label(res_fr, text=""); stage1_lbl.grid(row=0, column=0, sticky=W)
counts_lbl = ttk.Label(res_fr, text=""); counts_lbl.grid(row=1, column=0, sticky=W)
mode_lbl   = ttk.Label(res_fr, text=""); mode_lbl.grid(row=2, column=0, sticky=W)
slope_lbl  = ttk.Label(res_fr, text=""); slope_lbl.grid(row=3, column=0, sticky=W)
icpt_lbl   = ttk.Label(res_fr, text=""); icpt_lbl.grid(row=4, column=0, sticky=W)
result_lbl = ttk.Label(res_fr, text="", justify=LEFT)
result_lbl.grid(row=5, column=0, sticky=W, pady=(5, 2))

btn_fr = ttk.Frame(ctrl); btn_fr.grid(row=5, column=0, sticky=W, pady=(4, 8))
ttk.Button(btn_fr, text="Run GRASP",          command=calculate_and_display).grid(row=0, column=0, padx=3)
ttk.Button(btn_fr, text="Auto-opt threshold", command=auto_optimize_threshold).grid(row=0, column=1, padx=3)
ttk.Button(btn_fr, text="Clear all",          command=delete_all).grid(row=0, column=2, padx=3)

plot_col = ttk.Frame(mf); plot_col.grid(row=0, column=2, sticky=(N, W, E, S), padx=(12, 0))
plot_frame = ttk.LabelFrame(plot_col, text=" GRASP regression plot ", padding="4 4 4 4")
plot_frame.grid(row=0, column=0, sticky=(N, S, E, W))

root.mainloop()
