"""Physical-state comparisons for the offline failure report (no simulator required)."""

import html

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def features(raw):
    """Derive interpretable coordinates; cube geometry is privileged simulation data."""
    heading = raw["pose_angle"]
    dx, dy = raw["cube_x"] - raw["pose_x"], raw["cube_y"] - raw["pose_y"]
    forward = dx * np.cos(heading) + dy * np.sin(heading)
    lateral = -dx * np.sin(heading) + dy * np.cos(heading)
    return {
        "x": raw["pose_x"], "y": raw["pose_y"],
        "heading": np.degrees(heading),
        "left": raw["lwheel"], "right": raw["rwheel"],
        "speed": (raw["lwheel"] + raw["rwheel"]) / 2,
        "turn": raw["rwheel"] - raw["lwheel"],
        "ax": raw["accel_x"], "ay": raw["accel_y"], "az": raw["accel_z"],
        "cube_forward": forward, "cube_lateral": lateral,
        "distance": np.hypot(dx, dy), "bearing": np.degrees(np.arctan2(lateral, forward)),
    }


# Each entry is a purposeful comparison rather than an exhaustive pair matrix.
COMPARISONS = [
    ("initial", "cube_forward", "cube_lateral", "Initial cube location (privileged)",
     "Cube forward (mm)", "Cube lateral (mm)", 1.0, 1.0),
    ("initial", "distance", "bearing", "Initial approach geometry (privileged)",
     "Cube distance (mm)", "Cube bearing (degrees)", 1.0, 1.0),
    ("final", "x", "y", "Terminal robot position",
     "Robot x from spawn (mm)", "Robot y from spawn (mm)", 1.0, 1.0),
    ("final", "left", "right", "Terminal wheel coordination",
     "Left wheel (rad/s)", "Right wheel (rad/s)", 0.05, 0.05),
    ("final", "heading", "speed", "Terminal heading and motion",
     "Heading from spawn (degrees)", "Mean wheel speed (rad/s)", 1.0, 0.05),
    ("final", "speed", "turn", "Terminal forward and turning motion",
     "Mean wheel speed (rad/s)", "Right minus left wheel speed (rad/s)", 0.05, 0.05),
    ("final", "ax", "ay", "Terminal cube horizontal acceleration",
     "Cube acceleration x (g)", "Cube acceleration y (g)", 0.01, 0.01),
    ("final", "speed", "az", "Terminal motion and cube vertical acceleration",
     "Mean wheel speed (rad/s)", "Cube acceleration z (g)", 0.05, 0.01),
]


def state_comparison_report(output, episodes):
    """Save paired scatter PNGs with pooled axis limits and one point per episode."""
    complete = [ep for ep in episodes if "initial_raw_state" in ep and "raw_state" in ep]
    successes = sum(bool(ep["is_success"]) for ep in complete)
    failures = len(complete) - successes
    parts = ['<section><h2>State comparisons: success versus failure</h2>',
             f'<p>State coverage: {len(complete)}/{len(episodes)} episodes '
             f'({successes} successes, {failures} failures). One point per episode; '
             'each pair uses identical axes. Physical units, before normalization/clipping.</p>',
             '<p>Initial cube geometry describes starting conditions and uses privileged simulator '
             'positions. Terminal robot/cube measurements describe outcomes, not advance failure '
             'predictions. Successes stop at contact; failures usually stop at timeout, so the '
             'terminal measurements occur at different elapsed times. Cube acceleration is the '
             'cube sensor, not robot acceleration. Mean/difference wheel speeds are motion proxies.</p>']
    prepared = [(ep["is_success"], features(ep["initial_raw_state"]), features(ep["raw_state"]))
                for ep in complete]
    skipped = []
    positions = []
    for ep in complete:
        raw = ep["initial_raw_state"]
        robot = np.asarray([raw["pose_x"], raw["pose_y"]], dtype=float)
        cube = np.asarray([raw["cube_x"], raw["cube_y"]], dtype=float)
        if np.all(np.isfinite([*robot, *cube])):
            positions.append((bool(ep["is_success"]), robot, cube))
    if positions:
        pooled = np.asarray([point for _, robot, cube in positions for point in (robot, cube)])
        low, high = pooled.min(axis=0), pooled.max(axis=0)
        padding = np.maximum((high - low) * 0.08, 1.0)
        fig = Figure(figsize=(11, 5.2), layout="constrained", facecolor="#101820")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2, sharex=True, sharey=True)
        for ax, success, color in zip(axes, (True, False), ("#55cfa3", "#ff907f")):
            subset = [(robot, cube) for outcome, robot, cube in positions if outcome == success]
            ax.set_facecolor("#1c2935")
            for robot, cube in subset:
                ax.plot((robot[0], cube[0]), (robot[1], cube[1]),
                        color="#8293a3", alpha=0.35, linewidth=1, zorder=1)
            if subset:
                robots, cubes = (np.asarray(points) for points in zip(*subset))
                ax.scatter(robots[:, 0], robots[:, 1], s=34, marker="o", c="#55cfa3",
                           edgecolors="none", label="Cozmo", zorder=2)
                ax.scatter(cubes[:, 0], cubes[:, 1], s=42, marker="X", c="#ffd166",
                           edgecolors="none", label="Cube", zorder=3)
                ax.legend(facecolor="#1c2935", labelcolor="#eeeeee")
            else:
                ax.text(0.5, 0.5, "No episodes in this group", transform=ax.transAxes,
                        ha="center", color="white")
            ax.set(xlim=(low[0] - padding[0], high[0] + padding[0]),
                   ylim=(low[1] - padding[1], high[1] + padding[1]),
                   xlabel="X from spawn (mm)", ylabel="Y from spawn (mm)")
            ax.set_title(f'{"Success" if success else "Failure"} (n={len(subset)})', color=color)
            ax.tick_params(colors="#dddddd", labelleft=True)
            ax.xaxis.label.set_color("#eeeeee")
            ax.yaxis.label.set_color("#eeeeee")
            ax.grid(alpha=0.15)
            ax.set_aspect("equal", adjustable="box")
        title = "Initial Cozmo and cube positions"
        fig.suptitle(title, color="#eeeeee", fontsize=14)
        name = "comparison_initial_cozmo_cube_positions.png"
        fig.savefig(output / name, dpi=130)
        fig.clear()
        parts.append(f'<figure><img class="comparison" src="{name}" '
                     f'alt="{html.escape(title)}: connected Cozmo and cube positions by outcome"></figure>')
    for phase, xkey, ykey, title, xlabel, ylabel, xmin_span, ymin_span in COMPARISONS:
        points = np.asarray([[initial[xkey], initial[ykey]] if phase == "initial"
                             else [final[xkey], final[ykey]]
                             for _, initial, final in prepared], dtype=float).reshape(-1, 2)
        valid = np.all(np.isfinite(points), axis=1)
        if not valid.any() or np.any(np.ptp(points[valid], axis=0) < [xmin_span, ymin_span]):
            skipped.append(title)
            continue
        outcomes = np.asarray([bool(success) for success, _, _ in prepared])
        low, high = points[valid].min(axis=0), points[valid].max(axis=0)
        padding = (high - low) * 0.08
        fig = Figure(figsize=(11, 4.2), layout="constrained", facecolor="#101820")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2, sharex=True, sharey=True)
        for ax, success, color in zip(axes, (True, False), ("#55cfa3", "#ff907f")):
            subset = points[valid & (outcomes == success)]
            ax.set_facecolor("#1c2935")
            ax.scatter(subset[:, 0], subset[:, 1], s=23, alpha=0.7, c=color, edgecolors="none")
            ax.set(xlim=(low[0] - padding[0], high[0] + padding[0]),
                   ylim=(low[1] - padding[1], high[1] + padding[1]), xlabel=xlabel, ylabel=ylabel)
            ax.set_title(f'{"Success" if success else "Failure"} (n={len(subset)})', color=color)
            ax.tick_params(colors="#dddddd", labelleft=True)
            ax.xaxis.label.set_color("#eeeeee")
            ax.yaxis.label.set_color("#eeeeee")
            ax.grid(alpha=0.15)
            if not len(subset):
                ax.text(0.5, 0.5, "No episodes in this group", transform=ax.transAxes,
                        ha="center", color="white")
            if (xkey, ykey) in (("x", "y"), ("cube_forward", "cube_lateral")):
                ax.set_aspect("equal", adjustable="box")
        fig.suptitle(title, color="#eeeeee", fontsize=14)
        name = f"comparison_{phase}_{xkey}_{ykey}.png"
        fig.savefig(output / name, dpi=130)
        fig.clear()
        parts.append(f'<figure><img class="comparison" src="{name}" '
                     f'alt="{html.escape(title)}: successes left, failures right"></figure>')
    if skipped:
        parts.append('<p>Omitted for insufficient variation/data: ' + html.escape('; '.join(skipped)) + '.</p>')
    if len(complete) < len(episodes):
        parts.append('<p>Some older episodes have no saved states. Run '
                     '<code>python -m experiments.collect_failures --update-plots</code> '
                     'to replay missing episodes and fill this coverage.</p>')
    parts.append('<p>Fixed head/lift and redundant sine/cosine comparisons are excluded.</p></section>')
    return ''.join(parts)
