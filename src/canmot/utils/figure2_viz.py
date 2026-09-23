import logging
import os
from typing import Optional, Union
import numpy as np

CENTER_COV_COLOR = "red"
EXTENT_COV_COLOR = "blue"
VELOCITY_COV_COLOR = "orange"

def _yaw_from_quaternion_wxyz(quat: np.ndarray) -> float:
    qw, qx, qy, qz = quat
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _ellipse_points_from_cov(center: np.ndarray, cov_2x2: np.ndarray, scale: float, n: int = 64):
    cov_2x2 = 0.5 * (cov_2x2 + cov_2x2.T)
    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    eigvals = np.clip(eigvals, 1e-10, None)
    radii = scale * np.sqrt(eigvals)

    angles = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=0)
    points = (eigvecs @ (radii[:, None] * circle)).T + center[None, :]
    return points[:, 0], points[:, 1]


def _box_polygon_xy(x: float, y: float, length: float, width: float, yaw: float):
    l2 = 0.5 * float(length)
    w2 = 0.5 * float(width)
    corners_local = np.array(
        [[l2, w2], [l2, -w2], [-l2, -w2], [-l2, w2], [l2, w2]],
        dtype=float,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    corners_world = corners_local @ rot.T
    corners_world[:, 0] += x
    corners_world[:, 1] += y
    return corners_world[:, 0], corners_world[:, 1]


def _extent_cov_points(
    center_xy: np.ndarray,
    length: float,
    width: float,
    yaw: float,
    var_length: float,
    var_width: float,
    scale: float,
):
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=float)

    corner_local = np.array([0.5 * float(length), 0.5 * float(width)], dtype=float)
    corner_xy = center_xy + rot @ corner_local

    extent_cov_local = np.diag(
        [max(float(var_length), 1e-10), max(float(var_width), 1e-10)]
    )
    extent_cov_world = rot @ extent_cov_local @ rot.T
    sx, sy = _ellipse_points_from_cov(
        corner_xy, extent_cov_world, scale=scale, n=48
    )
    return sx, sy


def render_prediction_snapshot(
    base_infos: np.ndarray,
    base_states_meas_cov: np.ndarray,
    all_valid_ids: np.ndarray,
    frame_id: int,
    seq_id: int,
    predict_step: int,
    dt: float,
    output_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Render the scene snapshot used for Figure 2 as a PDF."""
    logger = logger if logger is not None else logging.getLogger(__name__)

    if base_infos is None or len(base_infos) == 0:
        return

    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("Visualization enabled but matplotlib is not available. Skipping snapshot.")
        return

    clp = matplotlib.colors.TABLEAU_COLORS
    tab_colors = list(clp.values())
    class_id_to_color = {
        id: tab_colors[id] for id in range(7)
    }

    scale_pos = scale_vel = scale_extent = 1.0
    center_stat = "median"
    draw_boxes = True
    draw_cv_lines = True
    draw_position_cov = True
    draw_velocity_cov = True
    draw_extent_cov = True
    score_threshold = 0.2

    base_infos = np.asarray(base_infos)
    base_states_meas_cov = np.asarray(base_states_meas_cov)
    all_valid_ids = np.asarray(all_valid_ids).astype(int)
    centers = base_infos[:, :2].astype(float)
    if center_stat == "mean":
        shift_xy = np.mean(centers, axis=0)
    else:
        shift_xy = np.median(centers, axis=0)

    # --- output paths ---
    seq_dir = os.path.join(output_dir, f"seq_{int(seq_id):03d}")
    os.makedirs(seq_dir, exist_ok=True)
    base_name = f"frame_{int(frame_id):06d}_step_{int(predict_step):06d}"

    # --- figure setup ---
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
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    # ax.set_title(f"Tracker BEV | seq={seq_id} frame={frame_id} step={predict_step}")
    ax.set_xlabel(f"x-shifted ({center_stat})")
    ax.set_ylabel(f"y-shifted ({center_stat})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.3)

    # legend de-duplication
    legend_drawn = set()

    def _plot_line(xy_x, xy_y, *, label: str, linestyle: str, linewidth: float, alpha: float, color: Optional[str] = None):
        if label in legend_drawn or label is None:
            ax.plot(xy_x, xy_y, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label="_nolegend_", color=color)
        else:
            ax.plot(xy_x, xy_y, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label, color=color)
            legend_drawn.add(label)

    def _plot_point(x, y, *, label: Union[str, None], marker: str, markersize: float, alpha: float, color: Optional[str] = None):
        if label in legend_drawn or label is None:
            ax.plot([x], [y], marker=marker, markersize=markersize, alpha=alpha, linestyle="None", label="_nolegend_", color=color)
        else:
            ax.plot([x], [y], marker=marker, markersize=markersize, alpha=alpha, linestyle="None", label=label, color=color)
            legend_drawn.add(label)

    def _draw_layer(
        infos: np.ndarray,
        covariances: np.ndarray,
        layer_prefix: str,
        is_overlay: bool,
    ) -> None:
        if infos is None or covariances is None or len(infos) == 0:
            return

        linestyle = "--" if is_overlay else "-"
        point_alpha = 0.65 if is_overlay else 0.95
        line_alpha = 0.35 if is_overlay else 0.55
        cov_alpha = 0.55 if is_overlay else 0.85

        for idx in range(len(infos)):
            info = infos[idx]
            cov = covariances[idx]
            track_id = int(all_valid_ids[idx])

            x = float(info[0]) - float(shift_xy[0])
            y = float(info[1]) - float(shift_xy[1])
            width = float(info[3])
            length = float(info[4])
            vx = float(info[6])
            vy = float(info[7])
            quat = info[8:12].astype(float)
            yaw = _yaw_from_quaternion_wxyz(quat)
            class_label = int(info[13]) if info.shape[0] > 13 else -1
            color = class_id_to_color.get(class_label, "red")

            score = float(info[12]) if info.shape[0] > 12 else 1.0
            if score < score_threshold:
                continue

            # Center
            # _plot_point(
            #     x, y,
            #     label=f"center ({layer_prefix})" if layer_prefix is not None else f"center",
            #     marker="o",
            #     markersize=4.5,
            #     alpha=point_alpha,
            #     color=color,
            # )

            # Optional: annotate (kept lightweight)
            # ax.text(x, y, f"{track_id}", fontsize=7, alpha=0.7)

            # Boxes
            if draw_boxes:
                bx, by = _box_polygon_xy(x, y, length, width, yaw)
                _plot_line(
                    bx, by,
                    label=None,
                    linestyle=linestyle,
                    linewidth=1.0,
                    alpha=line_alpha,
                    color=color,
                )

        for idx in range(len(infos)):
            info = infos[idx]
            cov = covariances[idx]
            track_id = int(all_valid_ids[idx])

            x = float(info[0]) - float(shift_xy[0])
            y = float(info[1]) - float(shift_xy[1])
            width = float(info[3])
            length = float(info[4])
            vx = float(info[6])
            vy = float(info[7])
            quat = info[8:12].astype(float)
            yaw = _yaw_from_quaternion_wxyz(quat)
            class_label = int(info[13]) if info.shape[0] > 13 else -1
            color = class_id_to_color.get(class_label, "red")

            score = float(info[12]) if info.shape[0] > 12 else 1.0
            if score < score_threshold:
                continue


            # CV velocity arrow
            if draw_cv_lines:
                x_tip = x + float(dt) * vx
                y_tip = y + float(dt) * vy
                ax.annotate(
                    "",
                    xy=(x_tip, y_tip),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", linewidth=1.2, alpha=0.75),
                )
                # Dummy legend handle for arrows
                # arrow_label = f"velocity ({layer_prefix})" if layer_prefix is not None else "velocity"
                # if arrow_label not in legend_drawn:
                #     ax.plot([np.nan], [np.nan], linestyle=linestyle, linewidth=1.2, alpha=0.75, label=arrow_label, color="black")
                #     legend_drawn.add(arrow_label)

            # Position covariance ellipse
            if draw_position_cov and cov.shape[0] >= 2:
                cov_xy = cov[np.ix_([0, 1], [0, 1])].astype(float)
                ex, ey = _ellipse_points_from_cov(np.array([x, y], dtype=float), cov_xy, scale=scale_pos, n=64)
                _plot_line(
                    ex, ey,
                    label=f"center cov ({layer_prefix})" if layer_prefix is not None else "position cov",
                    linestyle=linestyle,
                    linewidth=1.3,
                    alpha=cov_alpha,
                    color=CENTER_COV_COLOR,
                )

            # Velocity covariance ellipse at (x+dt*vx, y+dt*vy)
            has_vel_cov = cov.shape[0] >= 8
            if draw_velocity_cov and has_vel_cov:
                cov_v = cov[np.ix_([6, 7], [6, 7])].astype(float)
                vel_anchor = np.array([x + float(dt) * vx, y + float(dt) * vy], dtype=float)
                vex, vey = _ellipse_points_from_cov(vel_anchor, cov_v, scale=scale_vel, n=48)
                _plot_line(
                    vex, vey,
                    label=f"velocity cov ({layer_prefix})" if layer_prefix is not None else "velocity cov",
                    linestyle=linestyle,
                    linewidth=1.1,
                    alpha=cov_alpha,
                    color=VELOCITY_COV_COLOR,
                )

            # Extent covariance ellipse near one corner
            has_extent_cov = cov.shape[0] >= 5
            if draw_extent_cov and has_extent_cov:
                ex, ey = _extent_cov_points(
                    center_xy=np.array([x, y], dtype=float),
                    length=length,
                    width=width,
                    yaw=yaw,
                    var_length=float(cov[4, 4]),
                    var_width=float(cov[3, 3]),
                    scale=scale_extent,
                )
                _plot_line(
                    ex, ey,
                    label=f"extent cov ({layer_prefix})" if layer_prefix is not None else "extent cov",
                    linestyle=linestyle,
                    linewidth=1.1,
                    alpha=cov_alpha,
                    color=EXTENT_COV_COLOR,
                )

            # Optional: hover-like info (not interactive)
            # You can enable per-point labels if needed:
            # ax.text(x, y, f"t={track_id} c={class_label} s={score:.2f}", fontsize=6, alpha=0.6)

    _draw_layer(
        infos=base_infos,
        covariances=base_states_meas_cov,
        layer_prefix=None,
        is_overlay=False,
    )

    # Legend (compact)
    if len(legend_drawn) > 0:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3, frameon=False)

    # Tight bounds around data for nicer snapshots
    try:
        xs = base_infos[:, 0].astype(float) - float(shift_xy[0])
        ys = base_infos[:, 1].astype(float) - float(shift_xy[1])
        if len(xs) > 0:
            pad = 5.0
            ax.set_xlim(float(xs.min()) - pad, float(xs.max()) + pad)
            ax.set_ylim(float(ys.min()) - pad, float(ys.max()) + pad)
    except Exception:
        pass

    fig.tight_layout()

    ax.set_xlim(-50, -12)
    ax.set_ylim(-8, 35)
    pdf_path = os.path.join(seq_dir, f"{base_name}.pdf")
    try:
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    except Exception as error:
        logger.warning("Could not export Figure 2 snapshot: %s", error)

    plt.close(fig)
