#!/usr/bin/env python3
"""
GSPro CSV Viewer - plotting + filtering GUI for GSPro shot exports.

Requires: pandas, numpy, matplotlib (+ tkinter, which ships with standard Python
on Windows/macOS; on Debian/Ubuntu: sudo apt install python3-tk)

Usage:  python gspro_viewer.py [path/to/export.csv]
"""
import os
import re
import sys
import textwrap

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle, Wedge
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except ImportError:  # lets draw_plot() be imported/tested headless
    tk = None

PLOT_KINDS = [
    "Scatter (X vs Y)",
    "Line by shot (Y vs Shot #)",
    "Histogram (Y)",
    "Box plot (Y by club)",
    "Mean +/- SD (Y by club)",
    "Dispersion (Offline vs Carry)",
    "Correlation heatmap",
    "Impact location (HI vs VI)",
    "Face vs path (D-plane)",
    "Club face dashboard (6 panels)",
    "Ball flight laws (A-I grid)",
    "Vertical trajectory (side view)",
]
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
HEATMAP_COLS = ["Carry", "TotalDistance", "BallSpeed", "BackSpin", "SideSpin", "HLA",
                "VLA", "Decent", "PeakHeight", "Offline", "ClubSpeed", "Path", "AoA",
                "FaceToTarget", "FaceToPath", "Lie", "Loft", "DynamicLoft",
                "SmashFactor", "SpinLoft", "HI", "VI"]


# --------------------------------------------------------------------------- data
def load_data(path):
    df = pd.read_csv(path)
    if "DistanceToPin" in df.columns:  # "249.35 mtrs" -> 249.35
        df["DistanceToPin"] = pd.to_numeric(
            df["DistanceToPin"].astype(str).str.extract(r"(-?\d+\.?\d*)")[0],
            errors="coerce")
    if "Club" not in df.columns:
        df["Club"] = "all"
    df["Club"] = df["Club"].astype(str)
    if {"DynamicLoft", "AoA"} <= set(df.columns):  # derived: spin loft = dyn. loft - attack angle
        df["SpinLoft"] = df["DynamicLoft"] - df["AoA"]
    df.insert(0, "Shot", np.arange(1, len(df) + 1))
    return df


def numeric_columns(df):
    """Numeric columns that actually vary (hides constant ones like rawCarryLM)."""
    cols = [c for c in df.select_dtypes("number").columns if df[c].nunique() > 1]
    return cols


def club_colors(df):
    clubs = list(dict.fromkeys(df["Club"]))  # order of first appearance
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(clubs)}


# ---- Ball flight laws: the 9-way A-I chart (start line x curve), right-hander
# code: (name, start -1 pull / 0 straight / +1 push, curve -1 draw-hook / 0 straight / +1 fade-slice)
LAWS = {"A": ("Pull Draw/Hook", -1, -1), "B": ("Pull", -1, 0), "C": ("Pull Fade/Slice", -1, 1),
        "D": ("Straight Draw/Hook", 0, -1), "E": ("Straight", 0, 0), "F": ("Straight Slice", 0, 1),
        "G": ("Push Draw", 1, -1), "H": ("Push", 1, 0), "I": ("Push Fade/Slice", 1, 1)}
LAW_GRID = [["F", "E", "D"], ["I", "H", "G"], ["C", "B", "A"]]      # layout of the chart
LAW_COLORS = {"A": "#b71c1c", "B": "#e53935", "C": "#ff8a80",        # pulls  = reds
              "D": "#1b5e20", "E": "#43a047", "F": "#81c784",        # straight = greens
              "G": "#0d47a1", "H": "#1e88e5", "I": "#64b5f6"}        # pushes = blues
LAW_BY = {(s, c): k for k, (_, s, c) in LAWS.items()}
LAW_SOURCES = ["Club face (FaceToTarget)", "Ball start (HLA)"]


def add_flight_laws(df, face_tol=1.0, curve_tol=1.0, source=LAW_SOURCES[0]):
    """Adds 'Law' (A-I) and 'LawName'.  start = FaceToTarget (or HLA); curve = face-to-path =
    FaceToTarget - Path (+ = face open to path = fade/slice).  Bands are +/- tolerances in deg."""
    d = df.copy()
    col = "HLA" if source.startswith("Ball") else "FaceToTarget"
    if not {col, "FaceToTarget", "Path"} <= set(d.columns):
        d["Law"], d["LawName"] = np.nan, np.nan
        return d
    start = d[col].to_numpy(float)
    curve = (d["FaceToTarget"] - d["Path"]).to_numpy(float)
    s = np.where(start < -face_tol, -1, np.where(start > face_tol, 1, 0))
    c = np.where(curve < -curve_tol, -1, np.where(curve > curve_tol, 1, 0))
    ok = ~(np.isnan(start) | np.isnan(curve))
    d["Law"] = [LAW_BY[(a, b)] if k else np.nan for a, b, k in zip(s, c, ok)]
    d["LawName"] = d["Law"].map(lambda k: LAWS[k][0] if isinstance(k, str) else np.nan)
    return d


def filter_df(df, clubs, filters):
    d = df[df["Club"].isin(clubs)]
    for col, lo, hi in filters:
        if lo is not None:
            d = d[d[col] >= lo]
        if hi is not None:
            d = d[d[col] <= hi]
    return d


# --------------------------------------------------------------------------- plotting
def _trend(ax, x, y, color):
    if len(x) >= 3 and np.ptp(x) > 0:
        m, b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, m * xs + b, "--", color=color, lw=1.3, alpha=.9)


def _cov_ellipse(ax, x, y, color):
    if len(x) < 3:
        return
    vals, vecs = np.linalg.eigh(np.cov(x, y))
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), 2 * np.sqrt(max(vals[1], 0)),
                         2 * np.sqrt(max(vals[0], 0)), angle=angle, fill=False,
                         edgecolor=color, lw=1.6, ls="-"))


FACE_SHAPES = ["Auto (per club)", "Driver", "Hybrid", "Iron", "Off"]
# Face outlines traced from the GSPro face-impact art (right-hander, face-on, toe left / hosel right).
# Pixel coords of the reference image (y down).  poly = hitting-face outline, dot = sweet-spot
# (heat-map centre) which becomes data (0, 0), halfw = plate half-width in px,
# grooves = (x0, x1, y) segments, smooth = Chaikin-smooth the outline.
_FACE_PX = {
    "Driver": dict(
        dot=(232.9, 296.5), halfw=168.0, smooth=True,
        poly=[(77, 237), (90, 228), (120, 221), (160, 216), (200, 214), (240, 215), (280, 218),
              (320, 225), (353, 232), (380, 237), (402, 245), (413, 262), (393, 290),
              (367, 317), (340, 337), (307, 353), (267, 363), (220, 367), (180, 360),
              (147, 347), (113, 323), (90, 293), (77, 260)],
        grooves=[(99, 158, 249), (100, 154, 269), (107, 154, 288), (116, 157, 308),
                 (132, 164, 328), (158, 177, 348), (188, 285, 357),
                 (316, 374, 251), (319, 375, 271), (319, 368, 290), (316, 361, 310),
                 (309, 344, 329), (296, 316, 349)]),
    "Hybrid": dict(
        dot=(704.5, 315.5), halfw=149.0, smooth=True,
        poly=[(564, 245), (592, 235), (625, 229), (665, 227), (698, 230), (738, 240),
              (778, 255), (818, 268), (845, 277), (862, 285), (850, 305), (832, 323),
              (815, 345), (790, 362), (745, 375), (705, 381), (675, 377), (640, 367),
              (615, 356), (592, 330), (575, 297), (567, 267)],
        grooves=[(597, 697, 247), (597, 757, 260), (597, 785, 273), (597, 811, 286),
                 (597, 820, 298), (599, 820, 311), (606, 820, 324), (618, 808, 337),
                 (635, 790, 350)]),
    "Iron": dict(
        dot=(1203.0, 329.7), halfw=110.0, smooth=False,
        poly=[(1100, 208), (1130, 215), (1163, 223), (1197, 233), (1230, 245), (1263, 257),
              (1297, 268), (1320, 280), (1320, 380), (1100, 380)],
        grooves=[(1104, 1133, 229), (1104, 1170, 240), (1104, 1203, 251), (1104, 1233, 262),
                 (1104, 1262, 273), (1104, 1287, 284), (1104, 1309, 295)]
                + [(1104, 1315, y) for y in (306, 317, 328, 339, 351, 362, 373)]),
}
_FACE_SIZE = {"Driver": 1.0, "Hybrid": .88, "Iron": .78}   # relative to the half-width setting


def _face_kind(club):
    """Guess face shape from a GSPro club code (DR, 3W, H3, I5, PW, ...)."""
    c = club.upper()
    if re.fullmatch(r"(PW|SW|LW|GW|AW|UW)", c):
        return "Iron"
    if c in ("DR", "D", "1W") or re.fullmatch(r"\d*W|FW\d*", c):
        return "Driver"
    if re.fullmatch(r"H\d*|\d+H", c):
        return "Hybrid"
    return "Iron"


def _chaikin(P, n=3):
    P = np.asarray(P, float)
    for _ in range(n):
        Q = np.roll(P, -1, axis=0)
        new = np.empty((2 * len(P), 2))
        new[0::2] = .75 * P + .25 * Q
        new[1::2] = .25 * P + .75 * Q
        P = new
    return P


def _face_outline(ax, shape, w, color, heel_pos=True):
    d = _FACE_PX[shape]
    k = w * _FACE_SIZE[shape] / d["halfw"]
    sx = 1.0 if heel_pos else -1.0           # which side of the x axis the heel is on
    dx, dy = d["dot"]

    def tr(x, y):                            # image px -> data units, sweet spot at (0, 0)
        return (np.asarray(x, float) - dx) * k * sx, -(np.asarray(y, float) - dy) * k

    P = np.array(d["poly"], float)
    if d["smooth"]:
        P = _chaikin(P)
    X, Y = tr(P[:, 0], P[:, 1])
    patch = Polygon(np.c_[X, Y], closed=True, facecolor=to_rgba(color, .07),
                    edgecolor=color, lw=1.8, zorder=.8)
    ax.add_patch(patch)
    for x0, x1, y in d["grooves"]:
        gx, gy = tr([x0, x1], [y, y])
        ln, = ax.plot(gx, gy, color=color, lw=.8, alpha=.45, zorder=.7)
        ln.set_clip_path(patch)


def _draw_faces(ax, df, colors, shape, w, heel_pos=True):
    if shape == "Off":
        return
    for club in dict.fromkeys(df["Club"]):
        kind = _face_kind(club) if shape.startswith("Auto") else shape
        _face_outline(ax, kind, w, colors.get(club, "#555555"), heel_pos)
    ax.plot(0, 0, "o", ms=6, mfc="#e02020", mec="#7a0000", mew=1, zorder=.95)  # sweet spot
    ax.autoscale_view()


def _hi_label(heel_pos):
    return "HI  (toe  <-  |  ->  heel)" if heel_pos else "HI  (heel  <-  |  ->  toe)"


def _flight_path(hla, offline, carry, n=48):
    """Top-down flight line from the tee (0, 0) to the landing point (offline, carry).
    Starts along the launch direction `hla` (deg, + = right of target) and bends with constant
    curvature (circular arc) so it lands exactly on (offline, carry)."""
    chord = np.hypot(offline, carry)
    if not np.isfinite(chord) or chord <= 0 or carry <= 0:
        return None
    th = np.radians(hla if np.isfinite(hla) else 0.0)
    ph = np.arctan2(offline, carry)             # direction of the chord (from the target line)
    u = np.linspace(0, 1, n)
    psi = th + u * 2 * (ph - th)                # heading turns linearly -> circular arc
    dx, dy = np.sin(psi), np.cos(psi)
    x = np.r_[0, np.cumsum((dx[1:] + dx[:-1]) / 2)]
    y = np.r_[0, np.cumsum((dy[1:] + dy[:-1]) / 2)]
    raw = np.hypot(x[-1], y[-1])
    if raw < 1e-9:
        return np.linspace(0, offline, n), np.linspace(0, carry, n)
    k = chord / raw                             # chord direction is ph by symmetry
    return x * k, y * k


def _draw_flights(ax, d, o, colors):
    """Flight lines for every shot (thin) + a bold average flight per group (club / law / all)."""
    cb = o["color_by"]
    if cb == "Law":
        groups = [(c, LAW_COLORS[c], g) for c, g in d.dropna(subset=["Law"]).groupby("Law")]
    elif cb == "Club":
        groups = [(c, colors[c], g) for c, g in d.groupby("Club", sort=False)]
    else:
        groups = [("all", "#444444", d)]
    for _, col, g in groups:
        for hla, off, car in zip(g["HLA"], g["Offline"], g["Carry"]):
            p = _flight_path(hla, off, car)
            if p is not None:
                ax.plot(p[0], p[1], color=col, lw=.9, alpha=.35, zorder=.9, solid_capstyle="round")
        if o["means"] and len(g):
            p = _flight_path(g["HLA"].mean(), g["Offline"].mean(), g["Carry"].mean())
            if p is not None:
                ax.plot(p[0], p[1], color="white", lw=5.5, alpha=.9, zorder=3)
                ax.plot(p[0], p[1], color=col, lw=3, zorder=3.1, solid_capstyle="round")
                ax.annotate("", xy=(p[0][-1], p[1][-1]), xytext=(p[0][-4], p[1][-4]),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=0, mutation_scale=22),
                            zorder=3.2)
    ax.plot(0, 0, marker="^", ms=9, color="black", zorder=4)          # tee / start


def _side_path(carry, peak, vla, desc, n=60):
    """Side-view flight (distance, height) rebuilt from carry, peak height, launch angle and descent
    angle.  Rises with the launch slope to the peak, then falls to the landing point with the
    descent slope; carry and peak height are hit exactly.  Returns (x, y, x_of_peak) or None."""
    if not (np.isfinite(carry) and np.isfinite(peak)) or carry <= 0 or peak <= 0:
        return None
    t_up = np.tan(np.radians(vla)) if np.isfinite(vla) and vla > 0.5 else 4 * peak / (0.6 * carry)
    t_dn = np.tan(np.radians(desc)) if np.isfinite(desc) and desc > 0.5 else 1.6 * t_up
    t_up, t_dn = max(t_up, 1e-3), max(t_dn, 1e-3)
    xp = carry * float(np.clip(t_dn / (t_dn + t_up), .35, .80))      # where the peak sits
    p = float(np.clip(t_up * xp / peak, 1.2, 6.0))                    # shape of the climb
    q = float(np.clip(t_dn * (carry - xp) / peak, 1.2, 6.0))          # shape of the descent
    x1 = np.linspace(0, xp, n)
    x2 = np.linspace(xp, carry, n)[1:]
    y1 = peak * (1 - (1 - x1 / xp) ** p)
    y2 = peak * (1 - ((x2 - xp) / (carry - xp)) ** q)
    return np.r_[x1, x2], np.r_[y1, y2], xp


def _side_plot(fig, ax, d, o, colors, pts):
    """Vertical trajectories: thin line per shot + bold average flight per group (with roll)."""
    cb = o["color_by"]
    d = d.dropna(subset=["Carry", "PeakHeight"])
    if d.empty:
        ax.text(.5, .5, "No shots with carry / peak height", ha="center", va="center", transform=ax.transAxes)
        return
    cmap = norm = None
    if cb == "Law":
        groups = [(c, LAW_COLORS[c], g, f"{c}  {LAWS[c][0]}") for c, g in d.dropna(subset=["Law"]).groupby("Law")]
    elif cb == "Club":
        groups = [(c, colors[c], g, c) for c, g in d.groupby("Club", sort=False)]
    else:
        groups = [("all", PALETTE[0], d, "all shots")]
        if cb != "(none)" and cb in d.columns:
            v = d[cb].astype(float)
            cmap, norm = matplotlib.colormaps["viridis"], Normalize(v.min(), v.max())
    xmax = float(np.nanmax(d[["Carry", "TotalDistance"]].to_numpy(float))) if "TotalDistance" in d else float(d["Carry"].max())
    ymax, handles = 1.0, []
    for code, col, g, label in groups:
        handles.append(Line2D([0], [0], color=col, lw=2.5, label=f"{label} ({len(g)})"))
        for rec in g.to_dict("records"):
            p = _side_path(rec["Carry"], rec["PeakHeight"], rec.get("VLA", np.nan), rec.get("Decent", np.nan))
            if p is None:
                continue
            xs, ys, xp = p
            c = col if cmap is None else cmap(norm(rec[cb]))
            ax.plot(xs, ys, color=c, lw=.9, alpha=.5, zorder=2)
            ymax = max(ymax, float(ys.max()))
            law = f"  [{rec['Law']}]" if isinstance(rec.get("Law"), str) else ""
            pts.append((xp, float(ys.max()),
                        f"#{int(rec['Shot'])}  {rec['Club']}{law}\nCarry {rec['Carry']:.1f} m   Peak {rec['PeakHeight']:.1f} m\n"
                        f"Launch {rec.get('VLA', np.nan):.1f}\u00b0   Descent {rec.get('Decent', np.nan):.1f}\u00b0"))
        if o["means"] and len(g):
            m = {k: float(g[k].mean()) for k in ("Carry", "PeakHeight", "VLA", "Decent", "TotalDistance") if k in g}
            p = _side_path(m["Carry"], m["PeakHeight"], m.get("VLA", np.nan), m.get("Decent", np.nan))
            if p is not None:
                xs, ys, xp = p
                ax.plot(xs, ys, color="white", lw=5.5, alpha=.9, zorder=3)
                ax.plot(xs, ys, color=col, lw=3, zorder=3.1, solid_capstyle="round")
                tot = m.get("TotalDistance", np.nan)
                if np.isfinite(tot) and tot > m["Carry"]:
                    ax.plot([m["Carry"], tot], [0, 0], color=col, lw=2.4, ls=(0, (4, 3)), zorder=3.1)
                ax.scatter([xp], [m["PeakHeight"]], s=46, color=col, edgecolor="white", zorder=4)
                ax.annotate(f"{m['PeakHeight']:.1f} m", (xp, m["PeakHeight"]), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color=col)
                ymax = max(ymax, m["PeakHeight"])
    ax.axhspan(-ymax, 0, color="#cfe6cf", alpha=.7, zorder=0)
    ax.axhline(0, color="#5b8a5b", lw=1.4, zorder=1)
    ax.set_xlim(-.01 * xmax, xmax * 1.03)
    ax.set_ylim(-.04 * ymax, ymax * 1.15)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Vertical trajectory (side view): thin = each shot, bold = group mean (peak labelled, "
                 "dashed = roll)", fontsize=10)
    if cmap is not None:
        fig.colorbar(ScalarMappable(norm, cmap), ax=ax, label=cb)
    elif len(handles) > 1 or cb in ("Club", "Law"):
        ax.legend(handles=handles, title="Flight law" if cb == "Law" else "Club")
    if o["equal"]:
        ax.set_aspect("equal", adjustable="datalim")


def _law_grid(fig, df, o):
    """The chart: 3x3 cells F E D / I H G / C B A, each with the mean club face (black) and
    club path (red) of the shots in that category, plus counts."""
    d = df.dropna(subset=["Law"])
    N = len(d)
    ft, ct = o.get("face_tol", 1.0), o.get("curve_tol", 1.0)
    kx = 2.0                                           # angle exaggeration so arrows are readable
    maxn = max([int((d["Law"] == k).sum()) for k in LAWS] + [1])
    for r, row in enumerate(LAW_GRID):
        for c, code in enumerate(row):
            ax = fig.add_subplot(3, 3, r * 3 + c + 1)
            name, s, cv = LAWS[code]
            g = d[d["Law"] == code]
            n = len(g)
            ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.0, 1.0)
            ax.set_aspect("equal"); ax.axis("off")
            col = LAW_COLORS[code]
            ax.add_patch(Rectangle((-1.25, -1.0), 2.5, 2.0, fc=to_rgba(col, .06 + .5 * n / maxn),
                                   ec=to_rgba(col, .9), lw=1.6, zorder=0))
            ax.plot([-1.2, 1.2], [0, 0], ls="--", color="#999", lw=1, zorder=1)   # target line
            if n:
                face, path, a = g["FaceToTarget"].mean(), g["Path"].mean(), 1.0
            else:                                       # idealised picture for an empty cell
                face = s * 4.0; path = face - cv * 6.0; a = .25
            fa, pa = np.radians(face * kx), np.radians(path * kx)
            ax.add_patch(Wedge((0, 0), .34, np.degrees(-fa) - 90, np.degrees(-fa) + 90,
                               fc="#222222", ec="none", alpha=.9 * a if n else .15, zorder=2))
            for ang, colr in ((pa, "#e00000"), (fa, "black")):
                dx, dy = -np.cos(ang), np.sin(ang)      # target = left, +angle (right of target) = up
                ax.annotate("", xy=(.95 * dx, .95 * dy), xytext=(-.95 * dx, -.95 * dy),
                            arrowprops=dict(arrowstyle="-|>", color=colr, lw=2.4,
                                            mutation_scale=16, alpha=a), zorder=3)
            ax.text(0, .96, f"{code}   {name}", ha="center", va="top", fontsize=10,
                    fontweight="bold", color="black", zorder=4)
            if n:
                txt = (f"n = {n}  ({100 * n / N:.0f}%)\nface {face:+.1f}\u00b0   path {path:+.1f}\u00b0   "
                       f"F-P {face - path:+.1f}\u00b0")
            else:
                txt = "no shots"
            ax.text(0, -.96, txt, ha="center", va="bottom", fontsize=8, color="#222", zorder=4)
    fig.suptitle(f"Ball flight laws  -  {N} shots   |   black = club face, red = club path "
                 f"(angles drawn x{kx:g}, means per category)   |   straight band: start +/-{ft:g}\u00b0, "
                 f"curve +/-{ct:g}\u00b0", fontsize=10)


def _law_zones(ax, o):
    """Overlay the 9 chart zones on the Path (x) vs FaceToTarget (y) plot."""
    ft, ct = o.get("face_tol", 1.0), o.get("curve_tol", 1.0)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    for yv in (-ft, ft):
        ax.axhline(yv, color="#888", lw=.8, ls="--", alpha=.6)
    xs = np.array(xl)
    for cv in (-ct, ct):                                # face - path = cv  ->  y = x + cv
        ax.plot(xs, xs + cv, color="#888", lw=.8, ls="--", alpha=.6)
    ax.set_xlim(xl); ax.set_ylim(yl)
    yc = {-1: (yl[0] - ft) / 2, 0: 0.0, 1: (ft + yl[1]) / 2}
    span = xl[1] - xl[0]
    for code, (_, s, c) in LAWS.items():
        y0 = yc[s]
        x0 = y0 - c * (ct + .18 * span)
        if xl[0] < x0 < xl[1] and yl[0] < y0 < yl[1]:
            ax.text(x0, y0, code, ha="center", va="center", fontsize=16, fontweight="bold",
                    color=LAW_COLORS[code], alpha=.55, zorder=0)


def draw_plot(fig, df, o, colors):
    """Render onto fig. o: dict(kind,x,y,color_by,bins,trend,equal,means).
    Returns list of (x, y, label) points for hover tooltips (may be empty)."""
    fig.clear()
    pts = []
    kind, x, y = o["kind"], o["x"], o["y"]
    if "Law" not in df.columns:
        df = add_flight_laws(df, o.get("face_tol", 1.0), o.get("curve_tol", 1.0),
                             o.get("law_src", LAW_SOURCES[0]))
    if df.empty:
        fig.add_subplot(111).text(.5, .5, "No shots match the current filters",
                                  ha="center", va="center", transform=fig.gca().transAxes)
        return pts
    if kind == PLOT_KINDS[9]:  # club face dashboard
        panels = [("HI", "VI", "Impact location", False),
                  ("Path", "FaceToTarget", "Face vs path (diag: face = path)", True),
                  ("FaceToPath", "SideSpin", "Face-to-path vs side spin", False),
                  ("FaceToTarget", "HLA", "Face vs start direction", False),
                  ("AoA", "VLA", "Attack angle vs launch", False),
                  ("SpinLoft", "BackSpin", "Spin loft vs back spin", False)]
        for i, (xc, yc, title, diag) in enumerate(panels, 1):
            a = fig.add_subplot(2, 3, i)
            a.grid(alpha=.25)
            a.set_title(title, fontsize=10)
            if xc not in df or yc not in df:
                continue
            for club, g in df.groupby("Club", sort=False):
                g = g.dropna(subset=[xc, yc])
                a.scatter(g[xc], g[yc], s=22, color=colors[club], alpha=.8,
                          edgecolor="white", linewidth=.4, label=club)
                if len(g):
                    a.scatter(g[xc].mean(), g[yc].mean(), marker="X", s=110,
                              color=colors[club], edgecolor="black", linewidth=1, zorder=5)
            a.axhline(0, color="black", lw=.8, alpha=.5)
            a.axvline(0, color="black", lw=.8, alpha=.5)
            if xc == "HI":
                _draw_faces(a, df, colors, o.get("face_shape", FACE_SHAPES[0]),
                            o.get("face_w", 2.0), o.get("heel_pos", True))
                a.set_aspect("equal", adjustable="datalim")
                xc = "HI (toe <- | -> heel)" if o.get("heel_pos", True) else "HI (heel <- | -> toe)"
            if diag:
                xl, yl = a.get_xlim(), a.get_ylim()
                lo, hi = min(xl[0], yl[0]), max(xl[1], yl[1])
                a.plot([lo, hi], [lo, hi], ":", color="gray")
                a.set_xlim(xl); a.set_ylim(yl)
            a.set_xlabel(xc, fontsize=9); a.set_ylabel(yc, fontsize=9)
            if i == 1:
                a.legend(fontsize=8)
        fig.suptitle(f"Club face dashboard ({len(df)} shots; X = club mean)")
        return pts

    if kind == PLOT_KINDS[10]:  # ball flight laws grid
        _law_grid(fig, df, o)
        return pts

    ax = fig.add_subplot(111)
    ax.grid(alpha=.25)
    groups = list(df.groupby("Club", sort=False))

    def tip(row, xc, yc):
        law = f"  [{row['Law']}]" if isinstance(row.get("Law"), str) else ""
        return f"#{int(row['Shot'])}  {row['Club']}{law}\n{xc}: {row[xc]:.2f}\n{yc}: {row[yc]:.2f}"

    def scatter_xy(xc, yc, s=38):
        cb = o["color_by"]
        d = df.dropna(subset=[xc, yc])
        if cb == "Club":
            for club, g in d.groupby("Club", sort=False):
                ax.scatter(g[xc], g[yc], s=s, color=colors[club], label=club,
                           alpha=.85, edgecolor="white", linewidth=.5)
                if o["trend"]:
                    _trend(ax, g[xc].values, g[yc].values, colors[club])
                if o["means"]:
                    ax.scatter(g[xc].mean(), g[yc].mean(), marker="X", s=140,
                               color=colors[club], edgecolor="black", linewidth=1.2, zorder=5)
            ax.legend(title="Club")
        elif cb == "Law":
            for code in sorted(d["Law"].dropna().unique()):
                g = d[d["Law"] == code]
                ax.scatter(g[xc], g[yc], s=s, color=LAW_COLORS[code], alpha=.9, edgecolor="white",
                           linewidth=.5, label=f"{code}  {LAWS[code][0]} ({len(g)})")
            ax.legend(title="Flight law", fontsize=8)
            if o["trend"]:
                _trend(ax, d[xc].values, d[yc].values, "black")
        elif cb == "(none)":
            ax.scatter(d[xc], d[yc], s=s, color=PALETTE[0], alpha=.85,
                       edgecolor="white", linewidth=.5)
            if o["trend"]:
                _trend(ax, d[xc].values, d[yc].values, "black")
        else:
            sc = ax.scatter(d[xc], d[yc], c=d[cb], s=s, cmap="viridis", alpha=.9,
                            edgecolor="white", linewidth=.5)
            fig.colorbar(sc, ax=ax, label=cb)
            if o["trend"]:
                _trend(ax, d[xc].values, d[yc].values, "black")
        for _, r in d.iterrows():
            pts.append((r[xc], r[yc], tip(r, xc, yc)))
        return d

    if kind == PLOT_KINDS[0]:
        scatter_xy(x, y)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{y} vs {x}")
        if o["equal"]:
            ax.set_aspect("equal", adjustable="datalim")

    elif kind == PLOT_KINDS[1]:
        d = df.dropna(subset=[y])
        for club, g in d.groupby("Club", sort=False):
            ax.plot(g["Shot"], g[y], "-o", ms=4, lw=1, color=colors[club], label=club, alpha=.85)
            if o["means"]:
                ax.hlines(g[y].mean(), g["Shot"].min(), g["Shot"].max(),
                          colors=colors[club], linestyles=":", lw=1.5)
        for _, r in d.iterrows():
            pts.append((r["Shot"], r[y], tip(r, "Shot", y)))
        ax.set_xlabel("Shot #"); ax.set_ylabel(y); ax.set_title(f"{y} by shot")
        ax.legend(title="Club")

    elif kind == PLOT_KINDS[2]:
        d = df.dropna(subset=[y])
        edges = np.histogram_bin_edges(d[y], bins=o["bins"])
        for club, g in d.groupby("Club", sort=False):
            ax.hist(g[y], bins=edges, alpha=.55, color=colors[club], label=f"{club} (n={len(g)})",
                    edgecolor="white")
            if o["means"]:
                ax.axvline(g[y].mean(), color=colors[club], ls="--", lw=1.6)
        ax.set_xlabel(y); ax.set_ylabel("Shots"); ax.set_title(f"Distribution of {y}")
        ax.legend()

    elif kind in (PLOT_KINDS[3], PLOT_KINDS[4]):
        names = [c for c, _ in groups]
        data = [g[y].dropna().values for _, g in groups]
        pos = np.arange(1, len(names) + 1)
        if kind == PLOT_KINDS[3]:
            bp = ax.boxplot(data, widths=.55, patch_artist=True, showfliers=False)
            for patch, n in zip(bp["boxes"], names):
                patch.set_facecolor(colors[n]); patch.set_alpha(.35)
            rng = np.random.default_rng(0)
            for p, n, v in zip(pos, names, data):
                ax.scatter(p + rng.uniform(-.15, .15, len(v)), v, s=16, color=colors[n],
                           alpha=.8, zorder=3)
            ax.set_title(f"{y} by club")
        else:
            m = [v.mean() if len(v) else np.nan for v in data]
            sd = [v.std(ddof=1) if len(v) > 1 else 0 for v in data]
            ax.bar(pos, m, yerr=sd, capsize=6, color=[colors[n] for n in names], alpha=.8)
            ax.set_title(f"{y}: mean +/- SD by club")
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{n}\n(n={len(v)})" for n, v in zip(names, data)])
        ax.set_ylabel(y)

    elif kind == PLOT_KINDS[5]:
        d = df.dropna(subset=["Offline", "Carry"])
        scatter_xy("Offline", "Carry", s=42)
        if o.get("flight_lines") and {"HLA", "Offline", "Carry"} <= set(d.columns):
            _draw_flights(ax, d, o, colors)
        grouped = "club"
        if o["color_by"] == "Club":
            for club, g in d.groupby("Club", sort=False):
                _cov_ellipse(ax, g["Offline"].values, g["Carry"].values, colors[club])
        elif o["color_by"] == "Law":                 # ball flight law groups A-I
            grouped = "flight law"
            for code, g in d.dropna(subset=["Law"]).groupby("Law"):
                _cov_ellipse(ax, g["Offline"].values, g["Carry"].values, LAW_COLORS[code])
                ax.text(g["Offline"].mean(), g["Carry"].mean(), code, ha="center", va="center",
                        fontsize=15, fontweight="bold", color=LAW_COLORS[code], zorder=6,
                        bbox=dict(boxstyle="circle,pad=.15", fc="white", ec=LAW_COLORS[code],
                                  lw=1.4, alpha=.9))
        ax.axvline(0, color="black", lw=1, alpha=.6)
        ax.set_xlabel("Offline"); ax.set_ylabel("Carry")
        ax.set_title(f"Dispersion (ellipse = 1 sigma per {grouped}"
                     + ("; letter = group mean)" if grouped == "flight law" else ")"))
        if o["equal"]:
            ax.set_aspect("equal", adjustable="datalim")

    elif kind == PLOT_KINDS[6]:
        cols = [c for c in HEATMAP_COLS if c in df.columns and df[c].nunique() > 1]
        corr = df[cols].corr()
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(cols, fontsize=8)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = corr.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > .6 else "black")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="Pearson r")
        ax.set_title(f"Correlation ({len(df)} shots)")

    elif kind == PLOT_KINDS[7]:  # impact location on the face
        d = scatter_xy("HI", "VI", s=46)
        if o["color_by"] == "Club":
            for club, g in d.groupby("Club", sort=False):
                _cov_ellipse(ax, g["HI"].values, g["VI"].values, colors[club])
        fcol = colors if o["color_by"] == "Club" else {c: "#555555" for c in colors}
        hp = o.get("heel_pos", True)
        _draw_faces(ax, df, fcol, o.get("face_shape", FACE_SHAPES[0]), o.get("face_w", 2.0), hp)
        ax.axhline(0, color="black", lw=1, alpha=.6)
        ax.axvline(0, color="black", lw=1, alpha=.6)
        ax.set_xlabel(_hi_label(hp)); ax.set_ylabel("VI (vertical impact)")
        ax.set_title("Impact location (ellipse = 1 sigma per club). Try Color by SmashFactor")
        ax.set_aspect("equal", adjustable="datalim")

    elif kind == PLOT_KINDS[8]:  # D-plane style: path vs face
        scatter_xy("Path", "FaceToTarget", s=46)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        lo, hi = min(xl[0], yl[0]), max(xl[1], yl[1])
        ax.plot([lo, hi], [lo, hi], ":", color="gray", label="Face = Path (FaceToPath 0)")
        ax.set_xlim(xl); ax.set_ylim(yl)
        ax.axhline(0, color="black", lw=.8, alpha=.5)
        ax.axvline(0, color="black", lw=.8, alpha=.5)
        if not o.get("law_src", LAW_SOURCES[0]).startswith("Ball"):
            _law_zones(ax, o)
        ax.set_xlabel("Path (deg)"); ax.set_ylabel("FaceToTarget (deg)")
        ax.set_title("Face vs path  (letters = ball flight law zones)"); ax.legend()

    elif kind == PLOT_KINDS[11]:  # vertical trajectory (side view)
        _side_plot(fig, ax, df, o, colors, pts)
    return pts


# --------------------------------------------------------------------------- shot summary
# Units follow the GSPro UI: metres, km/h, degrees, rpm.  HI / VI (impact location) are millimetres.
STORY_BG = "#16211b"
ZONE_RED, ZONE_YEL, ZONE_GRN = "#f6caca", "#fbf4bd", "#cbe6cb"
KX = 3.0               # the story pictures exaggerate angles / lateral distances by this factor
FACE_HALF_MM = 52.0    # driver face half-width (mm) for the real-scale club face picture


def _ex(a):
    """Exaggerate an angle (deg) for the story pictures so small angles stay readable."""
    return float(np.degrees(np.arctan(KX * np.tan(np.radians(a)))))


def _vec(a):
    """Unit vector for a heading `a` degrees from the target line (up = target, + = right)."""
    t = np.radians(a)
    return np.array([np.sin(t), np.cos(t)])


def _nz(v, default=np.nan):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _robust(series):
    """Median and a robust spread (scaled MAD) so mishits don't distort a club's usual range."""
    v = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if len(v) == 0:
        return np.nan, np.nan
    med = float(np.median(v))
    s = float(np.median(np.abs(v - med)) * 1.4826)
    if not s > 0:
        s = float(v.std()) if len(v) > 1 else 0.0
    return med, s


def flight_model(df):
    """Two through-origin fits calibrated on the loaded shots (used for the 'optimal' values):
         curve [m]  = kc * SideSpin * Carry      (lateral movement beyond the straight start line)
         SideSpin   = ks * FaceToPath            (rpm per degree; face-to-path = face - path)"""
    kc, ks, n = 4.3e-5, 190.0, 0          # fallbacks if the file has too few usable shots
    need = ["Carry", "Offline", "HLA", "SideSpin", "FaceToTarget", "Path"]
    if all(c in df.columns for c in need):
        d = df
        if "SmashFactor" in d.columns:
            d = d[d["SmashFactor"] > 1.0]
        d = d[d["Carry"] > 20].dropna(subset=need)
        n = len(d)
        if n >= 8:
            curve = d["Offline"] - d["Carry"] * np.tan(np.radians(d["HLA"]))
            x = d["SideSpin"] * d["Carry"]
            f2p = d["FaceToTarget"] - d["Path"]
            if (x * x).sum() > 0 and (f2p * f2p).sum() > 0:
                a = (x * curve).sum() / (x * x).sum()
                b = (f2p * d["SideSpin"]).sum() / (f2p * f2p).sum()
                if a > 0 and b > 0:
                    kc, ks = float(a), float(b)
    return dict(kc=kc, ks=ks, n=n)


def shot_facts(row, model):
    g = lambda k: _nz(row.get(k))
    path, face, hla = g("Path"), g("FaceToTarget"), g("HLA")
    spin, carry, off = g("SideSpin"), g("Carry"), g("Offline")
    f = dict(path=path, face=face, hla=hla, spin=spin, carry=carry, off=off,
             total=g("TotalDistance"), f2p=face - path,        # + = face open to the path
             curve=off - carry * np.tan(np.radians(hla)),      # lateral drift away from the start line
             opt_spin=np.nan, opt_f2p=np.nan)
    if all(np.isfinite([spin, off, carry])) and carry >= 5:
        # spin that would have put this ball on the centre line, start line unchanged
        f["opt_spin"] = spin - off / (model["kc"] * carry)
        f["opt_f2p"] = f["f2p"] + (f["opt_spin"] - spin) / model["ks"]
    return f


def _path_name(p):
    return "Out-to-in" if p <= -1 else "In-to-out" if p >= 1 else "Neutral"


def _open_name(a, tol=0.5):
    return "Square" if abs(a) <= tol else ("Open" if a > 0 else "Closed")


def _lr(v, tol=0.0):
    return "-" if not np.isfinite(v) else ("Left" if v < -tol else "Right" if v > tol else "Straight")


# ---- story frames ---------------------------------------------------------------------------
def _story_axes(ax, title):
    ax.set_facecolor(STORY_BG)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2d3b33")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.axvline(0, color="#6f8fa8", lw=1.1, ls=(0, (6, 6)), alpha=.75, zorder=1)


def _ball(ax, x, y, r=.11):
    ax.add_patch(Circle((x, y), r, fc="#e9e9ee", ec="#b9bcc6", lw=1, zorder=5))
    ax.add_patch(Circle((x - .35 * r, y + .35 * r), .45 * r, fc="white", ec="none", alpha=.55, zorder=5.1))


def _bar(ax, cx, cy, length, thick, ang, **kw):
    """Club head drawn as a bar; `ang` = degrees CCW from +x.  Returns the heel (left) end."""
    t = np.radians(ang)
    ux, uy = np.cos(t), np.sin(t)
    vx, vy = -uy, ux
    hl, ht = length / 2, thick / 2
    pts = [(cx - ux * hl - vx * ht, cy - uy * hl - vy * ht), (cx + ux * hl - vx * ht, cy + uy * hl - vy * ht),
           (cx + ux * hl + vx * ht, cy + uy * hl + vy * ht), (cx - ux * hl + vx * ht, cy - uy * hl + vy * ht)]
    ax.add_patch(Polygon(pts, closed=True, **kw))
    return cx - ux * hl, cy - uy * hl


def _story_flight(f, x0, y0, y1):
    """Predicted flight drawn from (x0, y0) up to height y1 (lateral movement exaggerated xKX)."""
    p = _flight_path(f["hla"], f["off"], f["carry"])
    if p is None:
        return np.array([x0, x0]), np.array([y0, y1])
    s = (y1 - y0) / p[1][-1]
    return np.clip(x0 + KX * p[0] * s, -.82, .82), y0 + p[1] * s


def _card(ax, y_top, header, hcol, big, paras, result=None, w=1.84):
    wrap = lambda s: textwrap.wrap(s, 40) or [""]
    rl = wrap(result) if result else []
    LN, RN = .092, .10
    H = .06 + .105 + .20 + sum(LN * len(wrap(p)) + .035 for p in paras) + RN * len(rl) + .04
    ax.add_patch(FancyBboxPatch((-w / 2, y_top - H), w, H, boxstyle="round,pad=0,rounding_size=.04",
                                fc="#050a1a", ec="#2a3350", lw=1, alpha=.97, zorder=8))
    y = y_top - .06
    ax.text(0, y, header, color=hcol, fontsize=7.4, ha="center", va="top", zorder=9)
    y -= .105
    ax.text(0, y, big, color="white", fontsize=13, fontweight="bold", ha="center", va="top", zorder=9)
    y -= .20
    for p in paras:
        for line in wrap(p):
            ax.text(0, y, line, color="#c4ccd8", fontsize=7.2, ha="center", va="top", zorder=9)
            y -= LN
        y -= .035
    for line in rl:
        ax.text(0, y, line, color="white", fontsize=7.4, fontweight="bold", ha="center", va="top", zorder=9)
        y -= RN
    return y_top - H


def _table_card(ax, y_top, rows, w=1.84):
    """rows: (label, value, optimal_text | None), or None for a separator line."""
    RH, OH, SH = .108, .082, .055
    H = .06 + .14 + sum(SH if r is None else RH + (OH if r[2] else 0) for r in rows) + .04
    ax.add_patch(FancyBboxPatch((-w / 2, y_top - H), w, H, boxstyle="round,pad=0,rounding_size=.04",
                                fc="#050a1a", ec="#2a3350", lw=1, alpha=.97, zorder=8))
    y = y_top - .06
    ax.text(0, y, "FINAL RESULT", color="#9fb0c8", fontsize=7.4, ha="center", va="top", zorder=9)
    y -= .14
    for r in rows:
        if r is None:
            ax.plot([-w / 2 + .08, w / 2 - .08], [y - .01, y - .01], color="#2a3350", lw=.8, zorder=9)
            y -= SH
            continue
        lab, val, opt = r
        ax.text(-w / 2 + .08, y, lab, color="#9fb0c8", fontsize=7.3, ha="left", va="top", zorder=9)
        ax.text(w / 2 - .08, y, val, color="white", fontsize=7.3, ha="right", va="top", zorder=9)
        y -= RH
        if opt:
            ax.text(w / 2 - .08, y + .014, opt, color="#7be0b0", fontsize=6.8, ha="right", va="top", zorder=9)
            y -= OH


def _spin_arrows(ax, spin, centre=(0.0, -0.10), r=.19):
    """Cyan arrows around the ball showing the side-spin direction (seen from above)."""
    if not np.isfinite(spin) or abs(spin) < 150:
        return
    cw = spin > 0                                        # right spin = clockwise from above
    rad = -.45 if cw else .45
    cx, cy = centre
    for sx, (y0, y1) in ((1, (.07, -.12)), (-1, (-.07, .12))):
        a, b = (y0, y1) if cw else (y1, y0)
        ax.annotate("", xy=(cx + sx * r, cy + b), xytext=(cx + sx * r, cy + a),
                    arrowprops=dict(arrowstyle="-|>", color="#3ad0e0", lw=1.2, mutation_scale=9,
                                    connectionstyle=f"arc3,rad={rad}"), zorder=7)


def _story(axs, f):
    pa, fa = _ex(f["path"]) if np.isfinite(f["path"]) else 0.0, _ex(f["face"]) if np.isfinite(f["face"]) else 0.0
    B = np.array([0.0, -0.10])
    pv = _vec(pa)
    tail = B - 1.2 * pv
    fin = np.isfinite
    path, face, hla, spin, f2p = f["path"], f["face"], f["hla"], f["spin"], f["f2p"]

    def base(ax, title, hl_path=False):
        _story_axes(ax, title)
        fx, fy = _story_flight(f, B[0], B[1], 1.12)
        ax.plot(fx, fy, color="#f4f4f4", lw=1.1, alpha=.9, zorder=3)
        if hl_path:
            ax.plot([B[0], tail[0]], [B[1], tail[1]], color="#ff5a4f", lw=4, alpha=.35, zorder=2.5)
        ax.plot([B[0], tail[0]], [B[1], tail[1]], color="#e5483f", lw=1.3, zorder=3)
        _ball(ax, *B)

    head = dict(fc="#59606e", ec="#9aa3b2", lw=1, zorder=4)

    # 1 - start ---------------------------------------------------------------------------
    ax = axs[0]
    base(ax, "1/5  Start")
    ax.text(-.94, -1.08, "— ball flight", color="#f4f4f4", fontsize=6.8, va="bottom", zorder=9)
    ax.text(-.94, -.98, "— club path", color="#e5483f", fontsize=6.8, va="bottom", zorder=9)
    ax.text(.94, -1.08, "target line - -", color="#6f8fa8", fontsize=6.8, va="bottom", ha="right", zorder=9)
    ax.text(-.94, 1.08, f"angles and curve drawn x{KX:g}", color="#6e7d73", fontsize=6.5, ha="left", va="top")

    # 2 - club path -----------------------------------------------------------------------
    ax = axs[1]
    base(ax, "2/5  Club path", hl_path=True)
    A = B - .42 * pv
    hx, hy = _bar(ax, A[0], A[1], .9, .075, -pa, **head)
    ax.plot([hx, -1.05], [hy, hy], color="#7f8794", lw=1.2, zorder=3.5)
    pn = _path_name(path) if fin(path) else "-"
    desc = {"Out-to-in": "The club is travelling from the outside toward the inside through impact.",
            "In-to-out": "The club is travelling from the inside toward the outside through impact.",
            "Neutral": "The club is travelling along the target line through impact."}.get(pn, "")
    _card(ax, 1.1, "CLUB PATH", "#ff7aa8", f"{pn} {abs(path):.1f}°" if fin(path) else "-", [desc])

    # 3 - face to target ------------------------------------------------------------------
    ax = axs[2]
    base(ax, "3/5  Face to target")
    A = B - .30 * pv
    hx, hy = _bar(ax, A[0], A[1], .9, .075, -fa, fc="#59606e", ec="#4fd1c5", lw=1.6, zorder=4)
    ax.plot([hx, -1.05], [hy, hy], color="#7f8794", lw=1.2, zorder=3.5)
    ax.plot([A[0] - .45, A[0] + .45], [A[1], A[1]], color="white", lw=.8, ls=":", alpha=.35, zorder=3.6)
    nv = _vec(fa)
    ax.plot([A[0], A[0] + .55 * nv[0]], [A[1], A[1] + .55 * nv[1]], color="#4fd1c5", lw=1, ls=":", zorder=3.6)
    fn = _open_name(face) if fin(face) else "-"
    fdesc = {"Square": "The face is almost square to the target, so the ball should start close to the target line.",
             "Open": "The face is open to the target (pointing right of it), so the ball should start right.",
             "Closed": "The face is closed to the target (pointing left of it), so the ball should start left."}.get(fn, "")
    start = "-" if not fin(hla) else ("on the target line" if abs(hla) <= 1 else "left" if hla < 0 else "right")
    _card(ax, 1.1, "FACE TO TARGET", "#5aa9ff", f"{fn} {abs(face):.1f}°" if fin(face) else "-",
          [fdesc, "Face angle is the biggest influence on where the ball starts."], f"Result: starts {start}.")

    # 4 - face to path creates spin -------------------------------------------------------
    ax = axs[3]
    base(ax, "4/5  Face to path")
    A = B - .30 * pv
    hx, hy = _bar(ax, A[0], A[1], .9, .075, -fa, fc="#59606e", ec="#4fd1c5", lw=1.6, zorder=4)
    ax.plot([hx, -1.05], [hy, hy], color="#7f8794", lw=1.2, zorder=3.5)
    _bar(ax, A[0], A[1], .9, .075, -pa, fc="none", ec="#e5483f", lw=1, ls=":", zorder=4.1)
    _spin_arrows(ax, spin)
    pn2 = _open_name(f2p) if fin(f2p) else "-"
    sdir = "-" if not fin(spin) else ("left" if spin < 0 else "right")
    d2 = {"Square": "The face is square to the path, so there is little side spin and the ball flies straight.",
          "Open": "The face is open relative to the path. That tilts the spin axis and creates right spin.",
          "Closed": "The face is closed relative to the path. That tilts the spin axis and creates left spin."}.get(pn2, "")
    res = ("Result: almost no side spin, flies straight." if fin(spin) and abs(spin) < 150
           else f"Result: {sdir.capitalize()} {abs(spin):.0f} rpm, curves {sdir}." if fin(spin) else None)
    y_end = _card(ax, 1.1, "FACE TO PATH CREATES SPIN", "#3ad0e0",
                  f"{pn2} {abs(f2p):.1f}°" if fin(f2p) else "-", [d2], res)
    if fin(f["opt_f2p"]):
        o_dir = "left" if f["opt_spin"] < 0 else "right"
        msg = textwrap.wrap(f"{_open_name(f['opt_f2p'])} {abs(f['opt_f2p']):.1f}\u00b0 \u2192 "
                            f"{abs(f['opt_spin']):.0f} rpm {o_dir} = lands on centre line", 42)
        hh = .16 + .09 * len(msg)
        ax.add_patch(FancyBboxPatch((-.96, -1.12), 1.92, hh, boxstyle="round,pad=0,rounding_size=.04",
                                    fc="#06161c", ec="#1f6f78", lw=1, zorder=8))
        y = -1.12 + hh - .05
        ax.text(-.9, y, "OPTIMAL FACE TO PATH", color="#3ad0e0", fontsize=6.8, va="top", zorder=9)
        for line in msg:
            y -= .09
            ax.text(-.9, y - .015, line, color="#8ff0c8", fontsize=7.2, fontweight="bold", va="top", zorder=9)

    # 5 - final result --------------------------------------------------------------------
    ax = axs[4]
    _story_axes(ax, "5/5  Final result")
    fx, fy = _story_flight(f, 0.0, -1.1, .88)
    ax.plot(fx, fy, color="#f4f4f4", lw=1.1, alpha=.9, zorder=3)
    _ball(ax, fx[-1], fy[-1])
    _spin_arrows(ax, spin, centre=(fx[-1], fy[-1]))
    opt_f = (f"(Optimal {_open_name(f['opt_f2p'])} {abs(f['opt_f2p']):.1f}°)"
             if fin(f["opt_f2p"]) else None)
    opt_s = (f"(Optimal {_lr(f['opt_spin']).capitalize() if abs(f['opt_spin']) >= 1 else ''} "
             f"{abs(f['opt_spin']):.0f} rpm)" if fin(f["opt_spin"]) else None)
    num = lambda v, fmt: "-" if not fin(v) else fmt.format(v)
    rows = [("Club path", f"{_path_name(path)} {abs(path):.1f}°" if fin(path) else "-", None),
            ("Face angle", f"{_open_name(face)} {abs(face):.1f}°" if fin(face) else "-", None),
            ("Face to path", f"{_open_name(f2p)} {abs(f2p):.1f}°" if fin(f2p) else "-", opt_f),
            ("Push/pull", f"{'Left' if hla < 0 else 'Right'} {abs(hla):.1f}°" if fin(hla) else "-", None),
            ("Side spin", f"{_lr(spin).capitalize()} {abs(spin):.0f} rpm" if fin(spin) else "-", opt_s),
            None,
            ("Carry", num(f["carry"], "{:.1f} m"), None),
            ("Offline", f"{_lr(f['off'])} {abs(f['off']):.1f} m" if fin(f["off"]) else "-", None),
            ("Curve", f"{_lr(f['curve'])} {abs(f['curve']):.1f} m" if fin(f["curve"]) else "-", None)]
    _table_card(ax, .62, rows)


# ---- club face + flight panels ---------------------------------------------------------------
def _face_panel(ax, row, ref, heel_pos):
    ax.set_title("Club face", loc="left", fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_axis_off()
    kind = _face_kind(str(row["Club"]))
    _face_outline(ax, kind, FACE_HALF_MM, "#4a4f5c", heel_pos)
    if {"HI", "VI"} <= set(ref.columns):
        ax.scatter(ref["HI"], ref["VI"], s=7, color="#8a8f99", alpha=.4, zorder=2)
    hi, vi = _nz(row.get("HI")), _nz(row.get("VI"))
    if np.isfinite(hi) and np.isfinite(vi):
        for s, a, c in ((520, .20, "#2b6cff"), (300, .30, "#2fd36b"), (160, .55, "#ffd93a"), (70, .95, "#ee2a1f")):
            ax.scatter([hi], [vi], s=s, color=c, alpha=a, zorder=5, linewidths=0)
        toe = -hi if heel_pos else hi
        ax.text(.5, -.01, f"{abs(toe):.1f} mm {'toe' if toe > 0 else 'heel'}  ·  "
                          f"{abs(vi):.1f} mm {'high' if vi > 0 else 'low'}\n(grey = other {row['Club']} shots)",
                transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#555")
    ax.autoscale_view()


def _flight_panel(ax, f, ref):
    car, off, tot, hla = f["carry"], f["off"], f["total"], f["hla"]
    ax.set_facecolor("#d8ecd8")
    ax.grid(color="white", alpha=.7, lw=.8)
    ax.axvline(0, color="#4d7a55", lw=1.2, zorder=1)
    if {"Offline", "Carry"} <= set(ref.columns):
        ax.scatter(ref["Offline"], ref["Carry"], s=8, color="#6f8a72", alpha=.4, zorder=2)
    ex = off
    ymax = max([v for v in (car, tot, 10.0) if np.isfinite(v)]) * 1.12
    p = _flight_path(hla, off, car) if np.isfinite(car) and np.isfinite(off) else None
    if p is not None:
        ax.plot(p[0], p[1], color="#16202e", lw=2.2, zorder=4)
        ax.scatter([off], [car], s=50, color="#16202e", zorder=5)
        left = off > 0
        ax.text(off, car, f"{car:.1f} m carry  " if left else f"  {car:.1f} m carry", fontsize=8,
                va="bottom", ha="right" if left else "left", color="#16202e", zorder=6)
        if np.isfinite(tot) and tot > car:
            th = np.radians(hla if np.isfinite(hla) else 0.0)
            psi = 2 * np.arctan2(off, car) - th              # heading at landing
            ex, ey = off + (tot - car) * np.sin(psi), car + (tot - car) * np.cos(psi)
            ax.plot([off, ex], [car, ey], color="#16202e", lw=1.8, ls=(0, (4, 3)), zorder=4)
            ax.text(ex, ey, f"{tot:.1f} m total  " if ex > 0 else f"  {tot:.1f} m total", fontsize=8,
                    va="bottom", ha="right" if ex > 0 else "left", color="#16202e", zorder=6)
    ax.plot(0, 0, marker="^", ms=9, color="black", zorder=6)
    lim = max(12.0, 1.4 * max(abs(off) if np.isfinite(off) else 0, abs(ex) if np.isfinite(ex) else 0))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Offline (m)  ← left | right →", fontsize=8)
    ax.set_ylabel("Distance (m)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title("Flight (top-down)", loc="left", fontsize=10, fontweight="bold")


# ---- range bars --------------------------------------------------------------------------------
def _side_panel(ax, row, ref):
    car, pk = _nz(row.get("Carry")), _nz(row.get("PeakHeight"))
    vla, dec, tot = _nz(row.get("VLA")), _nz(row.get("Decent")), _nz(row.get("TotalDistance"))
    ax.set_title("Flight (side view)", loc="left", fontsize=10, fontweight="bold")
    ax.set_facecolor("#e9f2fb")
    xmax = max([v for v in (car, tot, 10.0) if np.isfinite(v)])
    ymax = max(pk if np.isfinite(pk) else 0.0, 1.0) * 1.4
    if {"Carry", "PeakHeight"} <= set(ref.columns):                    # the club's other shots, faint
        others = ref[ref["Shot"] != row["Shot"]]
        if len(others) > 80:
            others = others.sample(80, random_state=0)
        for rec in others.to_dict("records"):
            q = _side_path(rec["Carry"], rec["PeakHeight"], rec.get("VLA", np.nan), rec.get("Decent", np.nan))
            if q is not None:
                ax.plot(q[0], q[1], color="#7f93a8", lw=.7, alpha=.22, zorder=2)
    ax.axhspan(-.32 * ymax, 0, color="#cfe6cf", zorder=1)
    ax.axhline(0, color="#5b8a5b", lw=1.4, zorder=3)
    p = _side_path(car, pk, vla, dec)
    if p is not None:
        xs, ys, xp = p
        ax.fill_between(xs, 0, ys, color="#16202e", alpha=.07, zorder=3)
        ax.plot(xs, ys, color="#16202e", lw=2.4, zorder=5)
        ax.plot([xp, xp], [0, pk], color="#16202e", lw=.9, ls=":", zorder=4)
        ax.scatter([xp], [pk], s=42, color="#d9480f", zorder=6)
        ax.text(xp, pk + .04 * ymax, f"peak {pk:.1f} m", ha="center", va="bottom", fontsize=8,
                color="#d9480f", fontweight="bold")
        if np.isfinite(vla) and vla > 0:        # launch / descent tangents, readout in the corner
            dx = .16 * car
            ax.plot([0, dx], [0, dx * np.tan(np.radians(vla))], color="#d9480f", lw=1.2, ls="--", zorder=4)
        if np.isfinite(dec) and dec > 0:
            dx = .12 * car
            ax.plot([car - dx, car], [dx * np.tan(np.radians(dec)), 0], color="#d9480f", lw=1.2, ls="--", zorder=4)
        ax.text(.03, .97, f"launch {vla:.1f}\u00b0   descent {dec:.1f}\u00b0", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color="#d9480f", fontweight="bold", zorder=7)
        ax.scatter([car], [0], s=36, color="#16202e", zorder=6)
        has_roll = np.isfinite(tot) and tot > car
        ax.text(car, -.05 * ymax, f"carry {car:.1f} m", ha="right" if has_roll else "center",
                va="top", fontsize=8, color="#16202e")
        if has_roll:
            ax.plot([car, tot], [0, 0], color="#16202e", lw=2, ls=(0, (4, 3)), zorder=5)
            ax.text(tot, -.17 * ymax, f"total {tot:.1f} m", ha="right", va="top", fontsize=8, color="#16202e")
    ax.plot(0, 0, marker="^", ms=8, color="black", zorder=6)
    ax.set_xlim(-.02 * xmax, xmax * 1.04)
    ax.set_ylim(-.32 * ymax, ymax)
    ax.set_yticks([t for t in ax.get_yticks() if t >= 0])             # no negative heights
    ax.set_xlabel("Distance (m)", fontsize=8)
    ax.set_ylabel("Height (m)", fontsize=8)
    ax.tick_params(labelsize=8)


def _mk(name, vtxt, v, a, b, green, yellow, hl="", hr=""):
    return dict(name=name, vtxt=vtxt, v=v, a=a, b=b, green=green, yellow=yellow, hl=hl, hr=hr)


def _nice_ticks(a, b):
    t = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(a, b)
    return [x for x in t if a - 1e-9 <= x <= b + 1e-9]


def _bar_group(ax, title, items, ncols=2):
    ax.set_axis_off()
    nrows = int(np.ceil(len(items) / ncols))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    for i, it in enumerate(items):
        r, c = divmod(i, ncols)
        x0, w, yb = c + .03, .90, nrows - 1 - r
        a, b = it["a"], it["b"]
        pos = lambda v: float(np.clip((v - a) / (b - a), 0, 1))
        ax.text(x0, yb + .96, it["name"], fontsize=8.5, fontweight="bold", va="top")
        ax.text(x0 + w, yb + .96, it["vtxt"], fontsize=8.5, va="top", ha="right")
        y0, h = yb + .36, .30
        ax.add_patch(Rectangle((x0, y0), w, h, fc=ZONE_RED, ec="none"))
        for rng_list, col in ((it["yellow"], ZONE_YEL), (it["green"], ZONE_GRN)):
            for lo, hi in rng_list:
                p0, p1 = pos(lo), pos(hi)
                if p1 > p0:
                    ax.add_patch(Rectangle((x0 + w * p0, y0), w * (p1 - p0), h, fc=col, ec="none"))
        for t in _nice_ticks(a, b):
            ax.plot([x0 + w * pos(t)] * 2, [y0, y0 + h], color="#b5a8a8", lw=.6, zorder=3)
            ax.text(x0 + w * pos(t), y0 - .03, f"{t:g}", fontsize=6.5, ha="center", va="top", color="#666")
        if it["hl"]:
            ax.text(x0 + .01 * w, y0 + h / 2, it["hl"], fontsize=6.3, va="center", ha="left", color="#777", zorder=4)
        if it["hr"]:
            ax.text(x0 + .99 * w, y0 + h / 2, it["hr"], fontsize=6.3, va="center", ha="right", color="#777", zorder=4)
        if np.isfinite(it["v"]):
            xv = x0 + w * pos(it["v"])
            ax.plot([xv, xv], [y0 - .05, y0 + h + .05], color="#2530d0", lw=2.4, zorder=5, solid_capstyle="butt")


def _summary_items(row, f, ref, heel_pos):
    kind = _face_kind(str(row["Club"]))
    fin = np.isfinite

    def perf(key, name, unit, nd, hl, hr, nonneg=True):
        v = _nz(row.get(key))
        c, s = _robust(ref[key]) if key in ref.columns else (np.nan, np.nan)
        if not fin(c):
            c = v if fin(v) else 0.0
        if not (fin(s) and s > 0):
            s = max(abs(c) * .1, 1.0)
        a, b = c - 3 * s, c + 3 * s
        if nonneg:
            a = max(a, 0.0)
        txt = "-" if not fin(v) else f"{v:.{nd}f}{unit}"
        return _mk(name, txt, v, a, b, [(c - .7 * s, c + .7 * s)], [(c - 1.6 * s, c + 1.6 * s)], hl, hr)

    launch = [perf("ClubSpeed", "Club speed", " km/h", 1, "Slow", "Fast"),
              perf("BallSpeed", "Ball speed", " km/h", 1, "Slow", "Fast"),
              perf("VLA", "Launch angle", "°", 1, "Low", "High", nonneg=False),
              perf("BackSpin", "Backspin", " rpm", 0, "Low", "High"),
              perf("PeakHeight", "Peak height", " m", 1, "Low", "High"),
              perf("Decent", "Descent angle", "\u00b0", 1, "Shallow", "Steep"),
              perf("Carry", "Carry", " m", 1, "Short", "Long"),
              perf("TotalDistance", "Total distance", " m", 1, "Short", "Long")]

    p, h, sp, off, f2p = f["path"], f["hla"], f["spin"], f["off"], f["f2p"]
    c_f = f["opt_f2p"] if fin(f["opt_f2p"]) else 0.0
    c_s = f["opt_spin"] if fin(f["opt_spin"]) else 0.0
    lr = lambda v: "L" if v < 0 else "R"
    shape = [
        _mk("Club path", "-" if not fin(p) else f"{abs(p):.1f}° {lr(p)}  {_path_name(p)}", p, -12, 12,
            [(-3, 3)], [(-6, 6)], "Out to in", "In to out"),
        _mk("Face to path", "-" if not fin(f2p) else f"{abs(f2p):.1f}° {_open_name(f2p)}", f2p, -20, 20,
            [(c_f - 1.5, c_f + 1.5)], [(c_f - 3.5, c_f + 3.5)], "Closed", "Open"),
        _mk("Push / Pull", "-" if not fin(h) else f"{abs(h):.1f}° {lr(h)}", h, -10, 10,
            [(-2, 2)], [(-4, 4)], "Pull (L)", "Push (R)"),
        _mk("Sidespin", "-" if not fin(sp) else f"{abs(sp):.0f} rpm {lr(sp)}", sp, -6000, 6000,
            [(c_s - 400, c_s + 400)], [(c_s - 1000, c_s + 1000)], "Left", "Right"),
        _mk("Offline (carry)", "-" if not fin(off) else f"{abs(off):.1f} m {lr(off)}", off, -30, 30,
            [(-5, 5)], [(-12, 12)], "Left", "Right")]

    aoa = _nz(row.get("AoA"))
    z = {"Driver": ([(1, 6)], [(-2, 1), (6, 9)]), "Hybrid": ([(-4, 0)], [(-7, -4), (0, 3)]),
         "Iron": ([(-6, -2)], [(-9, -6), (-2, 1)])}[kind]
    hi, vi = _nz(row.get("HI")), _nz(row.get("VI"))
    toe = (-hi if heel_pos else hi) if fin(hi) else np.nan
    impact = [
        _mk("Angle of attack", "-" if not fin(aoa) else f"{aoa:.1f}°", aoa, -10, 10, z[0], z[1], "Low", "High"),
        _mk("Impact heel / toe", "-" if not fin(toe) else f"{abs(toe):.1f} mm {'Toe' if toe > 0 else 'Heel'}",
            toe, -6, 6, [(-2.5, 2.5)], [(-4.5, 4.5)], "Heel", "Toe"),
        _mk("Impact high / low", "-" if not fin(vi) else f"{abs(vi):.1f} mm {'High' if vi > 0 else 'Low'}",
            vi, -6, 6, [(-2.5, 2.5)], [(-4.5, 4.5)], "Low", "High")]
    return launch, shape, impact


# ---- the whole page ------------------------------------------------------------------------------
def shot_summary(fig, row, ref, model, heel_pos=True):
    """Draw the full shot summary (5-frame Shot Story, club face, flight, range bars) onto fig."""
    fig.clear()
    f = shot_facts(row, model)
    outer = fig.add_gridspec(3, 1, height_ratios=[4.0, 2.9, 2.1], hspace=.40,
                             left=.025, right=.985, top=.92, bottom=.04)
    g0 = outer[0].subgridspec(1, 5, wspace=.10)                                   # the 5 story frames
    g1 = outer[1].subgridspec(1, 4, width_ratios=[.75, 1.1, 1.35, 2.0], wspace=.36)
    g2 = outer[2].subgridspec(1, 2, width_ratios=[3, 2], wspace=.12)
    axs = [fig.add_subplot(g0[0, i]) for i in range(5)]
    _story(axs, f)
    _face_panel(fig.add_subplot(g1[0, 0]), row, ref, heel_pos)
    _flight_panel(fig.add_subplot(g1[0, 1]), f, ref)
    _side_panel(fig.add_subplot(g1[0, 2]), row, ref)
    launch, shape, impact = _summary_items(row, f, ref, heel_pos)
    _bar_group(fig.add_subplot(g1[0, 3]), "Launch & Distance", launch)
    _bar_group(fig.add_subplot(g2[0, 0]), "Shape & Direction", shape)
    _bar_group(fig.add_subplot(g2[0, 1]), "Impact & Contact", impact)
    law = row.get("Law")
    law_txt = f"   ·   Law {law}  {LAWS[law][0]}" if isinstance(law, str) and law in LAWS else ""
    fig.text(.025, .965, f"{row['Club']}   ·   Shot #{int(row['Shot'])}{law_txt}",
             fontsize=15, fontweight="bold", va="center")
    topin = _nz(row.get("DistanceToPin"))
    fig.text(.985, .965, "" if not np.isfinite(topin) else f"to pin {topin:.1f} m", fontsize=10,
             ha="right", va="center", color="#555")
    fig.text(.985, .008,
             f"Story angles drawn x{KX:g}. 'Optimal' = spin / face-to-path that would put this ball on the centre line, "
             f"estimated from {model['n']} shots in this file. Bars: performance = this club's median ± spread; "
             f"direction = fixed bands.", fontsize=6.5, ha="right", color="#777")


# --------------------------------------------------------------------------- GUI
class SummaryWindow:
    """Pop-up shot summary: 5-frame Shot Story + club face + flight + range bars.
    Left / Right arrow keys (or the buttons) step through the shots in the Data table order."""

    def __init__(self, app, shots, idx):
        self.app, self.shots, self.idx = app, shots, idx
        self.win = tk.Toplevel(app.root)
        self.win.geometry("1500x1010")
        bar = ttk.Frame(self.win)
        bar.pack(fill="x")
        ttk.Button(bar, text="\u25C0 Prev", command=lambda: self.step(-1)).pack(side="left", padx=4, pady=3)
        self.lbl = ttk.Label(bar, text="")
        self.lbl.pack(side="left", padx=8)
        ttk.Button(bar, text="Next \u25B6", command=lambda: self.step(1)).pack(side="left")
        ttk.Button(bar, text="Save image...", command=self.save).pack(side="right", padx=4)
        self.fig = Figure(figsize=(15, 10.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        for key, d in (("<Left>", -1), ("<Right>", 1)):
            self.win.bind(key, lambda e, d=d: self.step(d))
        self.win.bind("<Escape>", lambda e: self.win.destroy())
        self.win.focus_set()
        self.draw()

    def step(self, d):
        self.idx = (self.idx + d) % len(self.shots)
        self.draw()

    def draw(self):
        app = self.app
        cls = app.cls
        shot = self.shots[self.idx]
        row = cls[cls["Shot"] == shot].iloc[0]
        keep = cls[~cls["Shot"].isin(app.disabled)]
        ref = keep[keep["Club"] == row["Club"]]
        if ref.empty:
            ref = cls[cls["Club"] == row["Club"]]
        shot_summary(self.fig, row, ref, flight_model(keep), app.hi_pos.get() == "Heel")
        self.lbl.config(text=f"Shot #{shot}    ({self.idx + 1} / {len(self.shots)})    "
                             f"use the \u2190 \u2192 keys to step through shots")
        self.win.title(f"Shot summary - #{shot} {row['Club']}")
        self.canvas.draw_idle()

    def save(self):
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf")])
        if p:
            self.fig.savefig(p, dpi=150)


class App:
    def __init__(self, root, path=None):
        self.root = root
        root.title("GSPro CSV Viewer")
        root.geometry("1450x880")
        ttk.Style().theme_use("clam")
        self.df = None
        self.filters = []          # (col, lo, hi)
        self.disabled = set()      # Shot numbers switched off in the Data tab
        self.sort_col, self.sort_asc = None, True
        self.active = None         # filtered AND enabled shots (what gets plotted)
        self.cls = None            # all shots + Law/LawName columns
        self.filtered = None
        self.pts = []
        self.colors = {}
        self._build_ui()
        if path and os.path.exists(path):
            self.load(path)
        else:
            root.after(200, self.open_dialog)

    # ---- layout
    def _build_ui(self):
        outer = ttk.Frame(self.root)
        outer.pack(side="left", fill="y")
        lc = tk.Canvas(outer, width=310, highlightthickness=0)
        lsb = ttk.Scrollbar(outer, orient="vertical", command=lc.yview)
        left = ttk.Frame(lc, padding=6)
        left.bind("<Configure>", lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.create_window((0, 0), window=left, anchor="nw")
        lc.configure(yscrollcommand=lsb.set)
        lc.pack(side="left", fill="y")
        lsb.pack(side="right", fill="y")

        def _wheel(e):
            lc.yview_scroll(-1 if (getattr(e, "delta", 0) > 0 or e.num == 4) else 1, "units")

        def _wheel_on(_e):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                lc.bind_all(ev, _wheel)

        def _wheel_off(_e):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                lc.unbind_all(ev)
        lc.bind("<Enter>", _wheel_on)
        lc.bind("<Leave>", _wheel_off)
        right = ttk.Frame(self.root)
        right.pack(side="left", fill="both", expand=True)

        bar = ttk.Frame(left); bar.pack(fill="x")
        ttk.Button(bar, text="Open CSV...", command=self.open_dialog).pack(side="left", expand=True, fill="x")
        ttk.Button(bar, text="Export filtered...", command=self.export).pack(side="left", expand=True, fill="x")
        self.info = ttk.Label(left, text="No file loaded", foreground="#555")
        self.info.pack(anchor="w", pady=(4, 6))

        # clubs
        f = ttk.LabelFrame(left, text="Clubs", padding=4); f.pack(fill="x", pady=3)
        self.club_lb = tk.Listbox(f, selectmode="extended", height=5, exportselection=False)
        self.club_lb.pack(fill="x")
        self.club_lb.bind("<<ListboxSelect>>", lambda e: self.refresh())
        b = ttk.Frame(f); b.pack(fill="x", pady=2)
        ttk.Button(b, text="All", width=6, command=lambda: self._clubs(True)).pack(side="left")
        ttk.Button(b, text="None", width=6, command=lambda: self._clubs(False)).pack(side="left", padx=3)

        # plot options
        f = ttk.LabelFrame(left, text="Plot", padding=4); f.pack(fill="x", pady=3)
        self.kind = tk.StringVar(value=PLOT_KINDS[0])
        self.xv, self.yv, self.cv = tk.StringVar(), tk.StringVar(), tk.StringVar(value="Club")
        self.bins = tk.IntVar(value=15)
        self.trend = tk.BooleanVar(value=False)
        self.means = tk.BooleanVar(value=True)
        self.equal = tk.BooleanVar(value=False)
        self.flight = tk.BooleanVar(value=False)
        self.kind_cb = self._combo(f, "Type", self.kind, PLOT_KINDS)
        self.x_cb = self._combo(f, "X", self.xv, [])
        self.y_cb = self._combo(f, "Y", self.yv, [])
        self.c_cb = self._combo(f, "Color by", self.cv, ["Club"])
        r = ttk.Frame(f); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Hist bins").pack(side="left")
        ttk.Spinbox(r, from_=3, to=100, width=5, textvariable=self.bins,
                    command=self.refresh).pack(side="left", padx=4)
        for text, var in (("Trend line", self.trend), ("Show means", self.means),
                          ("Equal aspect", self.equal),
                          ("Flight lines (Dispersion plot)", self.flight)):
            ttk.Checkbutton(f, text=text, variable=var, command=self.refresh).pack(anchor="w")
        self.face_shape = tk.StringVar(value=FACE_SHAPES[0])
        self.face_w = tk.DoubleVar(value=2.0)
        self._combo(f, "Face shape", self.face_shape, FACE_SHAPES)
        self.hi_pos = tk.StringVar(value="Heel")
        self._combo(f, "+HI is", self.hi_pos, ["Heel", "Toe"])
        r = ttk.Frame(f); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Face half-width").pack(side="left")
        sp = ttk.Spinbox(r, from_=0.2, to=30, increment=0.1, width=6,
                         textvariable=self.face_w, command=self.refresh)
        sp.pack(side="left", padx=4)
        sp.bind("<Return>", lambda e: self.refresh())
        ttk.Label(r, text="(HI/VI units)", foreground="#777").pack(side="left")

        # ball flight laws
        f = ttk.LabelFrame(left, text="Ball flight laws (A-I)", padding=4); f.pack(fill="x", pady=3)
        self.law_src = tk.StringVar(value=LAW_SOURCES[0])
        self.face_tol = tk.DoubleVar(value=1.0)
        self.curve_tol = tk.DoubleVar(value=1.0)
        self._combo(f, "Start line", self.law_src, LAW_SOURCES)
        for label, var in (("Start +/- deg", self.face_tol), ("Curve +/- deg", self.curve_tol)):
            r = ttk.Frame(f); r.pack(fill="x", pady=1)
            ttk.Label(r, text=label, width=12).pack(side="left")
            sp = ttk.Spinbox(r, from_=0, to=10, increment=0.25, width=6, textvariable=var,
                             command=self.refresh)
            sp.pack(side="left")
            sp.bind("<Return>", lambda e: self.refresh())
        ttk.Label(f, text="Start = club face (or HLA) vs target; curve = face-to-path "
                          "(face - path). Inside the +/- band = 'straight'.",
                  foreground="#777", wraplength=270, justify="left").pack(anchor="w", pady=(2, 0))

        # filters
        f = ttk.LabelFrame(left, text="Range filters", padding=4); f.pack(fill="x", pady=3)
        self.fcol = tk.StringVar(); self.flo = tk.StringVar(); self.fhi = tk.StringVar()
        self.fcol_cb = self._combo(f, "Column", self.fcol, [], redraw=False)
        r = ttk.Frame(f); r.pack(fill="x", pady=2)
        ttk.Label(r, text="min").pack(side="left")
        ttk.Entry(r, textvariable=self.flo, width=8).pack(side="left", padx=3)
        ttk.Label(r, text="max").pack(side="left")
        ttk.Entry(r, textvariable=self.fhi, width=8).pack(side="left", padx=3)
        ttk.Button(r, text="Add", width=5, command=self.add_filter).pack(side="left")
        self.flist = tk.Listbox(f, height=5, exportselection=False)
        self.flist.pack(fill="x", pady=2)
        b = ttk.Frame(f); b.pack(fill="x")
        ttk.Button(b, text="Remove", command=self.del_filter).pack(side="left")
        ttk.Button(b, text="Clear", command=self.clear_filters).pack(side="left", padx=3)
        ttk.Label(f, text="Tip: filter Smash/Carry to drop mishits", foreground="#777").pack(anchor="w")

        # right: tabs
        nb = ttk.Notebook(right); nb.pack(fill="both", expand=True)
        pf = ttk.Frame(nb); nb.add(pf, text="Plot")
        self.fig = Figure(figsize=(8, 6), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        NavigationToolbar2Tk(self.canvas, pf).update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self._hover)

        sf = ttk.Frame(nb); nb.add(sf, text="Stats (by club, for Y)")
        cols = ("Club", "N", "Mean", "Std", "Min", "Median", "Max")
        self.stats = ttk.Treeview(sf, columns=cols, show="headings", height=8)
        for c in cols:
            self.stats.heading(c, text=c); self.stats.column(c, width=90, anchor="center")
        self.stats.pack(fill="x", padx=6, pady=6)
        ttk.Label(sf, text="Ball flight laws: shots per category (enabled shots only)").pack(
            anchor="w", padx=6)
        self.laws = ttk.Treeview(sf, show="headings", height=9)
        self.laws.pack(fill="x", padx=6, pady=6)

        df_ = ttk.Frame(nb); nb.add(df_, text="Data")
        tb = ttk.Frame(df_); tb.pack(fill="x", padx=4, pady=3)
        ttk.Label(tb, text="Click a box in the Off column (or select rows + Space) to disable a "
                           "shot. Click a heading to sort. Right-click a row for its shot summary.",
                  foreground="#555").pack(side="left")
        ttk.Button(tb, text="Enable all", command=self.enable_all).pack(side="right")
        ttk.Button(tb, text="Enable selected",
                   command=lambda: self._toggle(self.tree.selection(), False)).pack(side="right", padx=3)
        ttk.Button(tb, text="Disable selected",
                   command=lambda: self._toggle(self.tree.selection(), True)).pack(side="right")
        self.tree = ttk.Treeview(df_, show="headings", selectmode="extended")
        self.tree.tag_configure("off", foreground="#a0a0a0")
        self.tree.bind("<Button-1>", self._tree_click)
        for seq in ("<Button-3>", "<Button-2>", "<Control-Button-1>"):   # right-click (Win/Linux/macOS)
            self.tree.bind(seq, self._tree_menu)
        self.tree.bind("<space>", lambda e: (self._toggle(self.tree.selection()), "break")[1])
        vs = ttk.Scrollbar(df_, orient="vertical", command=self.tree.yview)
        hs = ttk.Scrollbar(df_, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.pack(side="right", fill="y"); hs.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        self.annot = None

    def _combo(self, parent, label, var, values, redraw=True):
        r = ttk.Frame(parent); r.pack(fill="x", pady=1)
        ttk.Label(r, text=label, width=9).pack(side="left")
        cb = ttk.Combobox(r, textvariable=var, values=values, state="readonly", width=20)
        cb.pack(side="left", fill="x", expand=True)
        if redraw:
            cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        return cb

    # ---- data
    def open_dialog(self):
        p = filedialog.askopenfilename(title="Open GSPro CSV",
                                       filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p:
            self.load(p)

    def load(self, path):
        try:
            self.df = load_data(path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e)); return
        self.colors = club_colors(self.df)
        num = numeric_columns(self.df)
        self.x_cb["values"] = num; self.y_cb["values"] = num
        self.c_cb["values"] = ["Club", "Law", "(none)"] + num
        self.fcol_cb["values"] = num
        self.xv.set("Carry" if "Carry" in num else num[0])
        self.yv.set("TotalDistance" if "TotalDistance" in num else num[1])
        self.fcol.set(num[0]); self.cv.set("Club")
        self.club_lb.delete(0, "end")
        for c in self.colors:
            self.club_lb.insert("end", c)
        self._clubs(True, refresh=False)
        self.filters.clear(); self.flist.delete(0, "end")
        self.disabled.clear(); self.sort_col, self.sort_asc = None, True
        self.info.config(text=f"{os.path.basename(path)}: {len(self.df)} shots")
        self.refresh()

    def _clubs(self, on, refresh=True):
        self.club_lb.selection_clear(0, "end")
        if on:
            self.club_lb.selection_set(0, "end")
        if refresh:
            self.refresh()

    def add_filter(self):
        col = self.fcol.get()
        try:
            lo = float(self.flo.get()) if self.flo.get().strip() else None
            hi = float(self.fhi.get()) if self.fhi.get().strip() else None
        except ValueError:
            messagebox.showwarning("Filter", "min/max must be numbers"); return
        if not col or (lo is None and hi is None):
            return
        self.filters.append((col, lo, hi))
        self.flist.insert("end", f"{col}: {'-inf' if lo is None else lo} .. {'inf' if hi is None else hi}")
        self.flo.set(""); self.fhi.set("")
        self.refresh()

    def del_filter(self):
        for i in reversed(self.flist.curselection()):
            self.flist.delete(i); del self.filters[i]
        self.refresh()

    def clear_filters(self):
        self.filters.clear(); self.flist.delete(0, "end"); self.refresh()

    def export(self):
        if self.active is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:  # filtered + enabled shots only
            self.active.drop(columns=["Shot"]).to_csv(p, index=False)

    # ---- drawing
    def refresh(self):
        """Filters/clubs changed: recompute the subset, redraw everything incl. the table."""
        if self.df is None:
            return
        clubs = [self.club_lb.get(i) for i in self.club_lb.curselection()]
        ft, ct, src = self._law_params()
        self.cls = add_flight_laws(self.df, ft, ct, src)
        self.filtered = filter_df(self.cls, clubs, self.filters)
        self._replot()
        self._fill_table()

    def _law_params(self):
        def num(var, default):
            try:
                return max(float(var.get()), 0.0)
            except (tk.TclError, ValueError):
                return default
        return num(self.face_tol, 1.0), num(self.curve_tol, 1.0), self.law_src.get()

    def _replot(self):
        """Redraw plot + stats from filtered shots that are not disabled."""
        self.active = self.filtered[~self.filtered["Shot"].isin(self.disabled)]
        try:
            face_w = max(float(self.face_w.get()), 0.05)
        except (tk.TclError, ValueError):
            face_w = 2.0
        ft, ct, src = self._law_params()
        opts = dict(kind=self.kind.get(), x=self.xv.get(), y=self.yv.get(),
                    face_tol=ft, curve_tol=ct, law_src=src,
                    face_shape=self.face_shape.get(), face_w=face_w,
                    heel_pos=(self.hi_pos.get() == "Heel"),
                    color_by=self.cv.get(), bins=self.bins.get(), trend=self.trend.get(),
                    means=self.means.get(), equal=self.equal.get(),
                    flight_lines=self.flight.get())
        self.annot = None
        self.pts = draw_plot(self.fig, self.active, opts, self.colors)
        self.canvas.draw_idle()
        n_off = int(self.filtered["Shot"].isin(self.disabled).sum())
        self.info.config(text=f"{len(self.active)} / {len(self.df)} shots plotted"
                              + (f"  ({n_off} disabled)" if n_off else ""))
        self._fill_stats(opts["y"])

    def _fill_stats(self, y):
        self.stats.delete(*self.stats.get_children())
        d = self.active
        self._fill_laws(d)
        if d.empty or y not in d:
            return
        for club, g in list(d.groupby("Club", sort=False)) + [("ALL", d)]:
            v = g[y].dropna()
            if len(v):
                self.stats.insert("", "end", values=(club, len(v), f"{v.mean():.2f}",
                                  f"{v.std(ddof=1):.2f}" if len(v) > 1 else "-",
                                  f"{v.min():.2f}", f"{v.median():.2f}", f"{v.max():.2f}"))

    def _fill_laws(self, d):
        clubs = list(dict.fromkeys(d["Club"])) if len(d) else []
        cols = ["Law", "Name"] + clubs + ["ALL"]
        self.laws["columns"] = cols
        for c in cols:
            self.laws.heading(c, text=c)
            self.laws.column(c, width=170 if c == "Name" else 85, stretch=False,
                             anchor="w" if c == "Name" else "center")
        self.laws.delete(*self.laws.get_children())
        if d.empty:
            return
        for code in "ABCDEFGHI":
            row = [code, LAWS[code][0]]
            for sub_d in [d[d["Club"] == c] for c in clubs] + [d]:
                n = int((sub_d["Law"] == code).sum())
                row.append(f"{n}  ({100 * n / len(sub_d):.0f}%)" if n else "-")
            self.laws.insert("", "end", values=row)

    # ---- data tab: sorting + per-row disable
    def _fill_table(self):
        d = self.filtered
        cols = ["Off"] + list(d.columns)
        self.tree["columns"] = cols
        for c in cols:
            arrow = (" \u25B2" if self.sort_asc else " \u25BC") if c == self.sort_col else ""
            self.tree.heading(c, text=c + arrow, command=lambda c=c: self._sort_by(c))
            self.tree.column(c, width=48 if c == "Off" else 120 if c == "LawName" else 88, stretch=False,
                             anchor="center" if c == "Off" else "e")
        self.tree.delete(*self.tree.get_children())
        if self.sort_col == "Off":
            d = (d.assign(_off=d["Shot"].isin(self.disabled))
                   .sort_values(["_off", "Shot"], ascending=[self.sort_asc, True]).drop(columns="_off"))
        elif self.sort_col in d.columns:
            d = d.sort_values(self.sort_col, ascending=self.sort_asc, kind="stable", na_position="last")
        si = list(d.columns).index("Shot")
        for row in d.itertuples(index=False):
            shot = int(row[si])
            off = shot in self.disabled
            self.tree.insert("", "end", iid=str(shot), tags=("off",) if off else (),
                             values=["\u2612" if off else "\u2610"]
                                    + [f"{v:.3f}" if isinstance(v, float) else v for v in row])

    def _sort_by(self, col):
        if col == self.sort_col:
            self.sort_asc = not self.sort_asc
        else:
            self.sort_col, self.sort_asc = col, True
        self._fill_table()

    def _tree_click(self, ev):
        if self.tree.identify_region(ev.x, ev.y) != "cell" or self.tree.identify_column(ev.x) != "#1":
            return
        iid = self.tree.identify_row(ev.y)
        if iid:
            self._toggle([iid])
            return "break"

    def _tree_menu(self, ev):
        iid = self.tree.identify_row(ev.y)
        if not iid:
            return
        if iid not in self.tree.selection():
            self.tree.selection_set(iid)
        m = tk.Menu(self.tree, tearoff=0)
        m.add_command(label=f"Shot summary  (#{iid})...", command=lambda: self.open_summary(iid))
        m.add_separator()
        m.add_command(label="Disable / enable selected shot(s)",
                      command=lambda: self._toggle(self.tree.selection()))
        try:
            m.tk_popup(ev.x_root, ev.y_root)
        finally:
            m.grab_release()
        return "break"

    def open_summary(self, iid):
        if self.cls is None:
            return
        shots = [int(i) for i in self.tree.get_children()]       # current table order
        if int(iid) in shots:
            SummaryWindow(self, shots, shots.index(int(iid)))

    def _toggle(self, iids, state=None):
        """state None = flip, True = disable, False = enable."""
        if self.df is None:
            return
        for iid in iids:
            shot = int(iid)
            off = (shot not in self.disabled) if state is None else state
            if off:
                self.disabled.add(shot)
            else:
                self.disabled.discard(shot)
            if self.tree.exists(iid):
                self.tree.set(iid, "Off", "\u2612" if off else "\u2610")
                self.tree.item(iid, tags=("off",) if off else ())
        self._replot()
        if self.sort_col == "Off":
            self._fill_table()

    def enable_all(self):
        self.disabled.clear()
        self._replot()
        self._fill_table()

    def _hover(self, ev):
        if not self.pts or not self.fig.axes:
            return
        ax = self.fig.axes[0]
        if ev.inaxes is not ax:
            return
        xy = ax.transData.transform(np.array([(p[0], p[1]) for p in self.pts], dtype=float))
        dist = np.hypot(xy[:, 0] - ev.x, xy[:, 1] - ev.y)
        i = int(np.nanargmin(dist))
        if dist[i] > 12:
            if self.annot is not None and self.annot.get_visible():
                self.annot.set_visible(False); self.canvas.draw_idle()
            return
        if self.annot is None:
            self.annot = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                                     bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=.95),
                                     fontsize=8, zorder=10)
        self.annot.xy = (self.pts[i][0], self.pts[i][1])
        self.annot.set_text(self.pts[i][2]); self.annot.set_visible(True)
        self.canvas.draw_idle()


def main():
    if tk is None:
        sys.exit("tkinter not available. Install it (Debian/Ubuntu: sudo apt install python3-tk).")
    path = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    App(root, path)
    root.mainloop()


if __name__ == "__main__":
    main()
