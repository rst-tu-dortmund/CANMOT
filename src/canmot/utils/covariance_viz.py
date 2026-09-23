"""Covariance overview visualisation for paper figures.

Renders all nuScenes tracking classes side-by-side in BEV, showing rotated
boxes with measurement-noise (R) and CV-prediction covariance ellipses.

*Local* covariances are rotated into the map frame exactly as the Kalman
filter does (``R_world = Rot @ R_local @ Rot.T``), while *global*
covariances are already in the map frame and used as-is.

Both are overlaid on the same boxes in different colours / line-styles so
the effect of the rotation is immediately visible.

Usage (CLI)::

    python -m canmot.utils.covariance_viz \\
        --local-r  config/model_cfg/matrix_r/centerpoint/local/trainval/sample_cov.yaml \\
        --global-r config/model_cfg/matrix_r/centerpoint/global/trainval/sample_cov.yaml \\
        --local-q  config/model_cfg/matrix_q/centerpoint/local/trainval/CV_sample_cov.yaml \\
        --global-q config/model_cfg/matrix_q/centerpoint/global/trainval/CV_sample_cov.yaml \\
        -o covariance_overview
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ── nuScenes class definitions ──────────────────────────────────────────

CLASS_IDX_TO_NAME = {
    0: "bicycle",
    1: "bus",
    2: "car",
    3: "motorcycle",
    4: "pedestrian",
    5: "trailer",
    6: "truck",
}

PRETTY_CLASS_NAME = {
    "bicycle":    "bic.",
    "bus":        "bus",
    "car":        "car",
    "motorcycle": "moto.",
    "pedestrian": "ped.",
    "trailer":    "tra.",
    "truck":      "tru.",
}

# Standard nuScenes average BEV sizes: (width, length) in metres
CLASS_SIZES_WL: dict[str, tuple[float, float]] = {
    "bicycle":    (0.60,  1.70),
    "bus":        (2.94, 11.17),
    "car":        (1.93,  4.63),
    "motorcycle": (0.77,  2.11),
    "pedestrian": (0.67,  0.73),
    "trailer":    (2.90, 12.29),
    "truck":      (2.51,  6.93),
}

# Typical speed along heading (m/s) for the CV-prediction arrow
CLASS_TYPICAL_SPEED: dict[str, float] = {
    "bicycle":    2+CLASS_SIZES_WL["bicycle"][1],
    "bus":        3.0+CLASS_SIZES_WL["bus"][1],
    "car":        2.5+CLASS_SIZES_WL["car"][1],
    "motorcycle": 2.5+CLASS_SIZES_WL["motorcycle"][1],
    "pedestrian": 1.2+CLASS_SIZES_WL["pedestrian"][1],
    "trailer":    3.0+CLASS_SIZES_WL["trailer"][1],
    "truck":      3.0+CLASS_SIZES_WL["truck"][1],
}

# Default display order: smallest footprint → largest
DEFAULT_CLASS_ORDER = [4, 0, 3, 2, 6, 1, 5]
# pedestrian, bicycle, motorcycle, car, truck, bus, trailer

# ── Geometry helpers ────────────────────────────────────────────────────


def _ellipse_points(
    center: np.ndarray,
    cov_2x2: np.ndarray,
    scale: float = 1.0,
    n: int = 64,
):
    """Return *(xs, ys)* tracing a 2-D covariance ellipse.

    Parameters
    ----------
    center : (2,) array – ellipse centre.
    cov_2x2 : (2, 2) array – 2-D covariance matrix.
    scale : float – radius multiplier (1.0 = 1-σ).
    n : int – number of points on the ellipse.
    """
    C = np.asarray(cov_2x2, dtype=float)
    C = 0.5 * (C + C.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 1e-12, None)
    radii = scale * np.sqrt(eigvals)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)])              # (2, n)
    pts = (eigvecs @ (radii[:, None] * circle)).T           # (n, 2)
    return pts[:, 0] + center[0], pts[:, 1] + center[1]


def _box_corners(
    cx: float, cy: float, length: float, width: float, yaw: float,
):
    """Closed BEV oriented-rectangle (5 vertices)."""
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array(
        [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]],
        dtype=float,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=float)
    world = local @ R.T + np.array([cx, cy])
    return world[:, 0], world[:, 1]


# ── CV-model matrices ──────────────────────────────────────────────────


def _cv_F(dt: float) -> np.ndarray:
    """10×10 CV state-transition matrix.

    State vector: ``[x, y, z, w, l, h, vx, vy, vz, ry]``.
    """
    F = np.eye(10)
    F[0, 6] = dt   # x += dt·vx
    F[1, 7] = dt   # y += dt·vy
    F[8, 8] = 0.0  # vz decays (unobserved, zeroed each step)
    return F


def _cv_H() -> np.ndarray:
    """9×10 observation matrix (``has_velo=True``).

    Measurement: ``[x, y, z, w, l, h, vx, vy, ry]``
    State:       ``[x, y, z, w, l, h, vx, vy, vz, ry]``
    """
    H = np.zeros((9, 10))
    for i in range(8):      # x y z w l h vx vy  →  state 0…7
        H[i, i] = 1.0
    H[8, 9] = 1.0           # ry → state 9
    return H


def _rot_meas(yaw: float) -> np.ndarray:
    """9×9 measurement-space rotation (positions *and* velocities)."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.eye(9)
    R[0, 0] = c;  R[0, 1] = -s
    R[1, 0] = s;  R[1, 1] =  c
    R[6, 6] = c;  R[6, 7] = -s
    R[7, 6] = s;  R[7, 7] =  c
    return R


def _rot_state(yaw: float) -> np.ndarray:
    """10×10 state-space rotation (positions *and* velocities)."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.eye(10)
    R[0, 0] = c;  R[0, 1] = -s
    R[1, 0] = s;  R[1, 1] =  c
    R[6, 6] = c;  R[6, 7] = -s
    R[7, 6] = s;  R[7, 7] =  c
    return R


# ── YAML I/O ────────────────────────────────────────────────────────────


def _load_cov_yaml(path: str) -> list[np.ndarray]:
    """Load a list of per-class covariance matrices from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return [np.asarray(m, dtype=float) for m in data]


# ── Main rendering function ─────────────────────────────────────────────


def render_covariance_overview(
    local_r_path: str,
    global_r_path: str,
    local_q_path: str,
    global_q_path: str,
    output_path: str = "covariance_overview",
    *,
    yaw: float = np.pi / 12,   # <- half of pi/6
    dt: float = 0.5,
    scale: float = 1.0,
    save_pdf: bool = True,
    save_png: bool = True,
    dpi: int = 300,
    figsize_width: float = 3.3,
    class_order: Optional[list[int]] = None,
    # visibility scaling: scale each class so its LENGTH matches this reference
    # (keeps proportions correct within each class; improves readability for small classes)
    reference_length_m: float = 4.63,  # car length
    max_scale_up: float = 6.0,         # avoid ridiculous zoom for pedestrians
) -> matplotlib.figure.Figure:
    R_loc = _load_cov_yaml(local_r_path)
    R_glo = _load_cov_yaml(global_r_path)
    Q_loc = _load_cov_yaml(local_q_path)
    Q_glo = _load_cov_yaml(global_q_path)

    # nuScenes class order (as requested)
    default_order = DEFAULT_CLASS_ORDER  # bicycle, bus, car, motorcycle, pedestrian, trailer, truck
    order = class_order if class_order is not None else default_order

    F = _cv_F(dt)
    H = _cv_H()

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rot2 = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=float)

    RM = _rot_meas(yaw)   # fixed yaw for the plot
    RS = _rot_state(yaw)

    # ---- per-class slots + scaling (to make small classes visible) ----
    slots: list[dict] = []
    abs_cos, abs_sin = abs(cos_yaw), abs(sin_yaw)
    for ci in order:
        nm = CLASS_IDX_TO_NAME[ci]
        w, l = CLASS_SIZES_WL[nm]
        v = CLASS_TYPICAL_SPEED[nm]

        # scale each class so that its length is ~reference_length_m
        # s = float(np.clip(reference_length_m / max(l, 1e-6), 1.0, max_scale_up))
        s = 1.0

        # axis-aligned bbox extent of rotated box (in *scaled* coords)
        bex = (l * abs_cos + w * abs_sin) * s
        bey = (l * abs_sin + w * abs_cos) * s

        adx = (dt * v * cos_yaw) * s
        ady = (dt * v * sin_yaw) * s

        # rough padding
        pad = 1.5
        slot_w = max(bex, abs(adx) + 2.5) + pad
        slot_h = max(bey, abs(ady) + 2.5) + pad

        slots.append(dict(ci=ci, name=PRETTY_CLASS_NAME[nm], w=w, l=l, v=v, s=s, slot_w=slot_w, slot_h=slot_h))

    gap = 0.8
    origin_pad = 1.8
    x_cur = origin_pad
    for s in slots:
        s["cx"] = x_cur + s["slot_w"] / 2.0
        s["cy"] = 0.0
        x_cur += s["slot_w"] + gap
    total_x = x_cur

    max_sh = max(s["slot_h"] for s in slots) if slots else 5.0
    label_margin = 1.5
    y_lo = -max_sh / 2.0 - label_margin
    y_hi =  max_sh / 2.0 + 2.0

    data_w = total_x + 0.5
    data_h = (y_hi - y_lo)
    fig_h = max(figsize_width * data_h / max(data_w, 1e-6), 1.6)

    rc_backup = matplotlib.rcParams.copy()
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 0.8,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "path.simplify": False,
        "path.simplify_threshold": 0.0,
    })

    fig, ax = plt.subplots(figsize=(figsize_width, fig_h))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.grid(True, linewidth=0.25, alpha=0.25, color="gray")

    axins = zoomed_inset_axes(ax, 2, loc="upper left")

    # ---- only two colors (local/global); box/arrow in neutral gray ----
    COL_LOCAL = "#1f77b4"   # local
    COL_GLOBAL = "#d62728"  # global
    COL_BOX = "#333333"
    COL_ARR = "#555555"

    legend_handles = {}

    def _legend_once(label: str, *, color: str, linewidth: float):
        if label not in legend_handles:
            legend_handles[label] = plt.Line2D([], [], color=color, lw=linewidth, label=label)

    # NOTE: we draw everything in *class-local* coords first, then apply:
    #   world = slot_center + s * local
    # Covariances must scale by s^2.
    for slot in slots:
        ci, nm = slot["ci"], slot["name"]
        cx_slot, cy_slot = float(slot["cx"]), float(slot["cy"])
        w, l, v, s = float(slot["w"]), float(slot["l"]), float(slot["v"]), float(slot["s"])

        Rl, Rg = R_loc[ci], R_glo[ci]
        Ql, Qg = Q_loc[ci], Q_glo[ci]

        # local coordinate system: box center at (0,0)
        cx0, cy0 = 0.0, 0.0
        # arrow tip in local coords
        px0 = dt * v * cos_yaw
        py0 = dt * v * sin_yaw

        # helper: apply scaling+translation to arrays
        def T(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return cx_slot + s * x, cy_slot + s * y

        # box
        bx0, by0 = _box_corners(cx0, cy0, l, w, yaw)
        bx, by = T(bx0, by0)
        ax.plot(bx, by, color=COL_BOX, lw=0.8, zorder=5)
        axins.plot(bx, by, color=COL_BOX, lw=0.8, zorder=5)

        # _legend_once("box", color=COL_BOX, linewidth=0.8)

        # arrow
        ax.annotate(
            "",
            xy=(cx_slot + s * px0, cy_slot + s * py0),
            xytext=(cx_slot + s * cx0, cy_slot + s * cy0),
            arrowprops=dict(arrowstyle="->", color=COL_ARR, lw=0.8, mutation_scale=8),
            zorder=6,
        )
        axins.annotate(
            "",
            xy=(cx_slot + s * px0, cy_slot + s * py0),
            xytext=(cx_slot + s * cx0, cy_slot + s * cy0),
            arrowprops=dict(arrowstyle="->", color=COL_ARR, lw=0.8, mutation_scale=8),
            zorder=6,
        )
        # _legend_once("CV prediction", color=COL_ARR, linewidth=0.8)

        # draw cov sets
        for tag, Rm, Qm, is_local in [
            ("local",  Rl, Ql, True),
            ("global", Rg, Qg, False),
        ]:
            color = COL_LOCAL if is_local else COL_GLOBAL
            label_prefix = "local" if is_local else "global"

            if is_local:
                R_eff = RM @ Rm @ RM.T
                Q_eff = RS @ Qm @ RS.T
            else:
                R_eff = Rm
                Q_eff = Qm

            # (1) position cov at center
            cov_pos = R_eff[np.ix_([0, 1], [0, 1])]
            # scale covariance by s^2 (because coordinates are scaled by s)
            cov_pos_s = (s * s) * cov_pos
            ex0, ey0 = _ellipse_points(np.array([cx_slot, cy_slot]) * 0 + np.array([cx_slot, cy_slot]) * 0, cov_pos_s, scale=scale)  # dummy
            # easier: ellipse_points expects center; we already want world center:
            ex, ey = _ellipse_points(np.array([cx_slot + s*cx0, cy_slot + s*cy0]), cov_pos_s, scale=scale)
            ax.plot(ex, ey, color=color, lw=0.9, alpha=0.95, zorder=7)
            axins.plot(ex, ey, color=color, lw=0.9, alpha=0.95, zorder=7)
            _legend_once(f"{label_prefix} cov", color=color, linewidth=0.9)

            # (2) extent cov at front-right corner (in local coords)
            var_w = max(float(R_eff[3, 3]), 1e-12)
            var_l = max(float(R_eff[4, 4]), 1e-12)
            corner0 = rot2 @ np.array([0.5 * l, 0.5 * w], dtype=float)  # local corner offset
            ext_cov = rot2 @ np.diag([var_l, var_w]) @ rot2.T
            ext_cov_s = (s * s) * ext_cov
            ex2, ey2 = _ellipse_points(
                np.array([cx_slot + s * corner0[0], cy_slot + s * corner0[1]]),
                ext_cov_s,
                scale=scale,
                n=48,
            )
            ax.plot(ex2, ey2, color=color, lw=0.7, alpha=0.65, zorder=7)
            axins.plot(ex2, ey2, color=color, lw=0.7, alpha=0.65, zorder=7)

            # (3) predicted center covariance at arrow tip
            P0 = H.T @ R_eff @ H
            Pp = F @ P0 @ F.T + Q_eff
            cov_pred = Pp[np.ix_([0, 1], [0, 1])]
            cov_pred_s = (s * s) * cov_pred

            # IMPORTANT: same transformed arrow tip is the center
            ex3, ey3 = _ellipse_points(
                np.array([cx_slot + s * px0, cy_slot + s * py0]),
                cov_pred_s,
                scale=scale,
            )
            ax.plot(ex3, ey3, color=color, lw=1.1, alpha=0.95, zorder=8)
            axins.plot(ex3, ey3, color=color, lw=1.1, alpha=0.95, zorder=8)

        # class label (below)
        ax.text(
            cx_slot, cy_slot - max_sh / 2.0 - 0.15,
            nm,
            ha="center", va="top",
            fontsize=6, style="italic",
        )
        # axins.text(
        #     cx_slot, cy_slot - max_sh / 2.0 - 0.15,
        #     nm,
        #     ha="center", va="top",
        #     fontsize=6, style="italic",
        # )

    # ---- legend: outside axis, not inside ----
    # Use fig.legend so it doesn't steal axis space.
    fig.legend(
        handles=list(legend_handles.values()),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
    # leave room at top for legend
    fig.tight_layout(rect=[0.0, 0.0, 0.0, 0.0])
    # fig.tight_layout(pad=0.1)

    ax.set_xlim(-1.0, total_x + 0.2)
    ax.set_ylim(y_lo - 0.8, y_hi + 10.0)

    # zoom in small objects
    axins.set_xlim(3.3, 17.1)
    axins.set_ylim(-1.5, 3)

    # remove tick labels
    axins.xaxis.set_ticklabels('')
    axins.yaxis.set_ticklabels('')

    axins.text(
        4.2, 2.8,
        "2x",
        ha="center", va="top",
        fontsize=6, style="italic",
    )

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", alpha=0.4)

    out_abs = os.path.abspath(output_path)
    outdir = os.path.dirname(out_abs)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    base = output_path.removesuffix(".pdf").removesuffix(".png")

    if save_pdf:
        fig.savefig(base + ".pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    if save_png:
        fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")

    matplotlib.rcParams.update(rc_backup)
    return fig


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Render covariance overview figure for paper.",
    )
    p.add_argument("--local-r",  required=True,
                   help="Path to local R sample_cov.yaml")
    p.add_argument("--global-r", required=True,
                   help="Path to global R sample_cov.yaml")
    p.add_argument("--local-q",  required=True,
                   help="Path to local Q CV_sample_cov.yaml")
    p.add_argument("--global-q", required=True,
                   help="Path to global Q CV_sample_cov.yaml")
    p.add_argument("-o", "--output", default="covariance_overview",
                   help="Output base path (extensions added automatically)")
    p.add_argument("--yaw",   type=float, default=np.pi / 3,
                   help="Box heading angle [rad] (default: π/6)")
    p.add_argument("--dt",    type=float, default=0.5,
                   help="CV prediction time-step [s]")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Ellipse scale (1.0 = 1-σ)")
    p.add_argument("--pdf",   action="store_true", default=True)
    p.add_argument("--png",   action="store_true", default=False)
    p.add_argument("--dpi",   type=int,   default=300)
    p.add_argument("--width", type=float, default=3.3,
                   help="Figure width [inches]")
    args = p.parse_args()

    render_covariance_overview(
        local_r_path=args.local_r,
        global_r_path=args.global_r,
        local_q_path=args.local_q,
        global_q_path=args.global_q,
        output_path=args.output,
        yaw=args.yaw,
        dt=args.dt,
        scale=args.scale,
        save_pdf=args.pdf,
        save_png=args.png,
        dpi=args.dpi,
        figsize_width=args.width,
    )
    print(f"Saved → {args.output}")
