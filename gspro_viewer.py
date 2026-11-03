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

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Polygon, Rectangle, Wedge
from matplotlib.colors import to_rgba

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
    return pts


# --------------------------------------------------------------------------- GUI
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
                           "shot. Click a heading to sort.", foreground="#555").pack(side="left")
        ttk.Button(tb, text="Enable all", command=self.enable_all).pack(side="right")
        ttk.Button(tb, text="Enable selected",
                   command=lambda: self._toggle(self.tree.selection(), False)).pack(side="right", padx=3)
        ttk.Button(tb, text="Disable selected",
                   command=lambda: self._toggle(self.tree.selection(), True)).pack(side="right")
        self.tree = ttk.Treeview(df_, show="headings", selectmode="extended")
        self.tree.tag_configure("off", foreground="#a0a0a0")
        self.tree.bind("<Button-1>", self._tree_click)
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
