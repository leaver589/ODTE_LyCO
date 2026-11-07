"""Compact vector figures matching the current manuscript's experimental panels."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import numpy as np
from scipy.stats import t as student_t

from experiments import SCENARIOS, METHODS, CDF_METHODS, CDF_X, ENERGY_METHODS

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
MODES = ("all_fast", "all_slow", "base")
SERVICES = ("status", "diagnostic", "regional")
COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^")
OFFSETS = (0.18, 0.0, -0.18)


def save(figure, path):
    figure.savefig(path, format="pdf", metadata={"Title": path.stem, "Creator": "ODTE-LyCO reproduction"})
    plt.close(figure)


def _draw_weight_axes(figure, axes, results, config, duration, *, legend=True):
    scenario_clock = config.simulation["slots"] * config.slot_seconds / 60
    for index, (scene, axis) in enumerate(zip(SCENARIOS, axes)):
        runs = [r for r in results if r["scene"] == scene and r["method"] == "odte_lyco"]
        weights = np.asarray([r["weights"] for r in runs])
        times = weights[0, :, 0]
        means = weights[:, :, 1:].mean(axis=0)
        marker = "." if len(times) == 1 else None
        scenario = config.scenarios[SCENARIOS[scene]]
        axis.axvspan(scenario.evaluation_event_start * scenario_clock,
                    scenario.evaluation_event_end * scenario_clock, color="#f4d2d2", alpha=.6)
        axis.axvspan(scenario.evaluation_event_end * scenario_clock,
                    min(scenario_clock, duration), color="#dceaf6", alpha=.7)
        if duration > scenario_clock:
            axis.axvspan(scenario_clock, duration, color="#eef1f4", alpha=.85)
            axis.axvline(scenario_clock, color="#777777", ls=":", lw=.6)
        axis.plot(times, means[:, 0], color="#d62728", lw=.85, marker=marker, ms=1.5)
        axis.plot(times, means[:, 1], color="#1f77b4", lw=.85, marker=marker, ms=1.5)
        axis.axhline(1, color="#555555", ls="--", lw=.65)
        axis.text(.02, .96, f"({chr(97 + index)})", transform=axis.transAxes, va="top")
        axis.set(xlim=(0, duration), ylim=(.25, 1.8), yticks=[.4, .8, 1.2, 1.6],
                 xticks=np.linspace(0, duration, 4), xlabel="Time (min)")
        axis.grid(alpha=.22, lw=.45)
        axis.tick_params(labelsize=6.8, length=2, pad=2)
    axes[0].set_ylabel("Normalized weight")
    if legend:
        handles = [Patch(fc="#f4d2d2", label="Event"), Patch(fc="#dceaf6", label="Recovery"),
                   Patch(fc="#eef1f4", label="Post-event"),
                   Line2D([], [], color="#d62728", lw=.85, label="ODTE weight"),
                   Line2D([], [], color="#1f77b4", lw=.85, label="Energy weight"),
                   Line2D([], [], color="#555555", lw=.65, ls="--", label="Baseline")]
        figure.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
                      bbox_to_anchor=(.54, -.01), fontsize=6.1, columnspacing=.95)


def weight_figure(results, config, path):
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 1.67), sharey=True)
    figure.subplots_adjust(left=.075, right=.975, bottom=.32, top=.98, wspace=.16)
    duration = max(r.get("weight_slots", r["slots"]) for r in results) * config.slot_seconds / 60
    _draw_weight_axes(figure, axes, results, config, duration)
    save(figure, path)


def event_pairs(events):
    pairs = []
    for event in events:
        if event["phase"] == "started":
            pairs.append([event, None])
        else:
            pairs[-1][1] = event
    return pairs


def sample_values(trace, key, times):
    points = trace["points"]
    recorded = np.asarray([p["time_seconds"] for p in points])
    indices = np.searchsorted(recorded, times, side="right") - 1
    return np.asarray([points[i][key] for i in indices])


def state_runs(boundaries, states):
    first = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[first]:
            yield boundaries[first], boundaries[index], int(states[first])
            first = index


def window_max(times, values, horizon):
    boundaries = np.r_[np.arange(0.0, horizon, 30.0), horizon]
    states = [int(values[(times >= a) & (times < b)].max())
              for a, b in zip(boundaries[:-1], boundaries[1:])]
    return boundaries, np.asarray(states)


def draw_strip(axis, intervals, colors, names, label):
    axis.set(ylim=(0, 1), yticks=[.5], yticklabels=[label])
    axis.tick_params(axis="both", length=0, labelbottom=False, labelsize=7.2, pad=3)
    for spine in axis.spines.values():
        spine.set_visible(False)
    renderer = axis.figure.canvas.get_renderer()
    font = matplotlib.font_manager.FontProperties(family="DejaVu Sans", size=7.2)
    for left, right, state in intervals:
        axis.axvspan(left, right, facecolor=colors[state], linewidth=0)
        x0, x1 = axis.transData.transform(((left, 0), (right, 0)))[:, 0]
        text_width = renderer.get_text_width_height_descent(names[state], font, False)[0]
        if x1 - x0 > text_width + 6:
            midpoint = axis.transData.inverted().transform(((x0 + x1) / 2, 0))[0]
            axis.text(midpoint, .5, names[state], ha="center", va="center", fontsize=7.2)


def draw_trajectory(axis, trace, pairs, horizon, mode_index):
    color, marker, offset = COLORS[mode_index], MARKERS[mode_index], OFFSETS[mode_index]
    time = 0.0
    y = float(trace["points"][0]["location"] == "cloud") + offset
    size = 1.8 if mode_index == 0 else 3.1
    for start, finish in pairs:
        begin = float(start["time_seconds"])
        end = float(finish["time_seconds"]) if finish else horizon
        target = float(finish["target"] == "cloud") + offset if finish else y
        axis.plot([time, begin], [y, y], color=color, lw=.75)
        axis.plot([begin, end], [y, target], color=color, lw=.65, ls=":")
        axis.plot(begin, y, marker=marker, ms=size, mfc="white", mec=color, mew=.65, zorder=4)
        if finish:
            axis.plot(end, target, marker=marker, ms=size, mfc=color, mec=color, mew=.35, zorder=4)
        time, y = end, target
    axis.plot([time, horizon], [y, y], color=color, lw=.75)


def migration_figure(results, path):
    traces = {(r["scene"], r["method"], service): r["timeline"][service]
              for r in results if "timeline" in r for service in SERVICES}
    pairs = {key: event_pairs(trace["events"]) for key, trace in traces.items()}
    horizon = results[0]["slots"] * .1
    width, height = 7.16, 2.70
    figure = plt.figure(figsize=(width, height), dpi=144)
    for panel, scene in enumerate(SCENARIOS):
        anchors = np.unique([0, horizon] + [float(e["time_seconds"])
            for mode in MODES[1:] for service in SERVICES
            for pair in pairs[scene, mode, service] for e in pair if e is not None])
        display = np.r_[0, np.cumsum(np.minimum(np.diff(anchors), 12))]
        display /= display[-1]
        scale = (lambda values, a=anchors, d=display: np.interp(values, a, d),
                 lambda values, a=anchors, d=display: np.interp(values, d, a))
        left, plot_width = panel * (width / 3) + .39, width / 3 - .45
        figure.text((left + plot_width / 2) / width, 2.43 / height,
                    f"({chr(97 + panel)})", ha="center", va="center", fontsize=8)
        strips = []
        for row in range(5):
            axis = figure.add_axes([left / width, (2.26 - row * .135) / height,
                                    plot_width / width, .108 / height])
            axis.set_xscale("function", functions=scale)
            axis.set_xlim(0, horizon)
            strips.append(axis)
        lanes = []
        for row, service in enumerate(SERVICES):
            axis = figure.add_axes([left / width, (1.295 - row * .425) / height,
                                    plot_width / width, .395 / height])
            axis.set_xscale("function", functions=scale)
            axis.set(xlim=(0, horizon), ylim=(-.32, 1.85), yticks=[0, 1], yticklabels=["E", "C"])
            axis.tick_params(axis="both", labelsize=7.2, length=2, pad=1.5)
            axis.tick_params(axis="x", bottom=False, labelbottom=False)
            axis.set_facecolor("#fbfcfd")
            axis.text(.015, .98, service.title(), transform=axis.transAxes, va="top")
            for location in (0, 1):
                axis.axhline(location, color="#d2dade", lw=.45, zorder=0)
            for index, mode in enumerate(MODES):
                draw_trajectory(axis, traces[scene, mode, service],
                                pairs[scene, mode, service], horizon, index)
            lanes.append(axis)
        times = np.unique([p["time_seconds"] for mode in MODES for service in SERVICES
                           for p in traces[scene, mode, service]["points"]])
        pressure = np.max([sample_values(traces[scene, mode, service], "task_pressure", times)
                           for mode in MODES for service in SERVICES], axis=0)
        bounds, states = window_max(times, np.digitize(pressure, (.25, .55, .80)), horizon)
        draw_strip(strips[0], state_runs(bounds, states),
                   ("#eaf4ec", "#fff7e5", "#fcebdd", "#f7d9d9"),
                   ("Low", "Elevated", "High", "Critical"), "Task")
        for index, mode in enumerate(MODES):
            edge = np.max([sample_values(traces[scene, mode, service], "edge_pressure", times)
                           for service in SERVICES], axis=0)
            bounds, states = window_max(times, np.digitize(edge, (.35, .70)), horizon)
            draw_strip(strips[index + 1], state_runs(bounds, states),
                       ("#eaf4ec", "#fff7e5", "#fcebdd"),
                       ("Available", "Strained", "Saturated"), f"Edge {'FSB'[index]}")
        ranks = {"normal": 0, "degraded": 1, "interrupted": 2}
        link = np.max([[ranks[v] for v in sample_values(
            traces[scene, mode, service], "communication_state", times)]
            for mode in MODES for service in SERVICES], axis=0)
        intervals = list(state_runs(np.r_[times, horizon], link))
        draw_strip(strips[4], intervals, ("#eaf3fa", "#fff7e5", "#f7d9d9"),
                   ("Normal", "Degraded", "Interrupted"), "Link")
        for a, b, state in intervals:
            if state == 2:
                for axis in lanes:
                    axis.axvspan(a, b, fc="#f7d9d9", ec="#d8a7a7", hatch="////", lw=0, zorder=0)
        ticks = sorted({0., horizon} | {float(start["time_seconds"])
            for mode in MODES[1:] for service in SERVICES for start, _ in pairs[scene, mode, service]})
        lanes[-1].set_xticks(ticks, [f"{v:g}" for v in ticks], rotation=55, ha="right")
        lanes[-1].tick_params(axis="x", bottom=True, labelbottom=True)
    handles = [Line2D([], [], color=COLORS[i], marker=MARKERS[i], ms=3, lw=.8,
                      label=("All-fast (F)", "All-slow (S)", "Base (B)")[i]) for i in range(3)]
    handles += [(Line2D([], [], color="#657078", marker="o", mfc="white", ls="none", ms=3),
                 Line2D([], [], color="#657078", marker="o", mfc="#657078", ls="none", ms=3)),
                Patch(fc="#f7d9d9", ec="#d8a7a7", hatch="////", label="Link outage")]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.01),
                  labels=("All-fast (F)", "All-slow (S)", "Base (B)", "Start / switch", "Link outage"),
                  handler_map={tuple: HandlerTuple(ndivide=None)}, ncol=5, fontsize=7.2,
                  frameon=False, handlelength=1.4, columnspacing=1.1)
    figure.text(.5, .017, "Elapsed time (s)", ha="center")
    save(figure, path)


def cdf_figure(results, path):
    styles = (("#087e8b", "-"), ("#db8500", "--"), ("#616161", "-"),
              ("#526ea6", ":"), ("#b14675", "-."))
    titles = ("S1: Velocity drift", "S2: Frac-hit diagnosis", "S3: Zipper fracturing")
    with plt.rc_context({"font.family": "STIXGeneral", "font.size": 9}):
        figure, axes = plt.subplots(1, 3, figsize=(7.16, 125 / 72), sharey=True)
        figure.subplots_adjust(left=.075, right=.975, bottom=.33, top=.97, wspace=.16)
        for index, (scene, axis) in enumerate(zip(SCENARIOS, axes)):
            for method, (color, style) in zip(CDF_METHODS, styles):
                cdfs = [r["cdf"] for r in results if r["scene"] == scene and r["method"] == method]
                axis.step(CDF_X * 100, np.mean(cdfs, axis=0), where="post", color=color,
                          ls=style, lw=1.35 if method == "odte_lyco" else 1.05,
                          label=METHODS[method][2])
            axis.axhline(.05, color="#777777", ls=(0, (2, 3)), lw=.6)
            axis.text(.03, .95, f"({chr(97 + index)}) {titles[index]}", transform=axis.transAxes,
                      ha="left", va="top", fontsize=9)
            axis.set(xlim=(0, 100), ylim=(0, .75), xticks=[0, 25, 50, 75, 100],
                     yticks=[0, .25, .50, .75], xlabel="Event-window ODTE (%)")
            axis.tick_params(length=2.7, pad=2, labelsize=8)
            axis.grid(alpha=.35, lw=.45)
        axes[0].set_ylabel("Cumulative probability", fontsize=9, y=.46)
        figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
                      bbox_to_anchor=(.54, -.01), ncol=5, frameon=False,
                      fontsize=8.5, columnspacing=1.2, handlelength=2.7)
        save(figure, path)


def _draw_energy_axes(figure, axes, results, *, legend=True, bottom=.34, top=.95,
                      panel_labels=False):
    slots = results[0]["slots"]
    first, last = slots * .1 / 60, round(slots * 5 / 3) * .1 / 60
    maximum = 0.
    for index, (scene, axis) in enumerate(zip(SCENARIOS, axes)):
        if panel_labels:
            axis.text(.02, .96, f"({chr(100 + index)})", transform=axis.transAxes,
                      va="top")
        axis.axvspan(first, last, color="#e8f0f7")
        axis.axvline(first, ls=":", color="#777777", lw=.65)
        for method, color, marker, style in zip(ENERGY_METHODS, ("#1f5b8f", "#d95f02"), ("o", "s"), ("-", "--")):
            runs = [r for r in results if r["scene"] == scene and r["method"] == method]
            traces = np.asarray([r["checkpoints"] for r in runs])
            times = traces[0, :, 0]
            values = traces[:, :, 1]
            mean = values.mean(axis=0)
            half = (student_t.ppf(.975, len(runs) - 1) * values.std(axis=0, ddof=1) / np.sqrt(len(runs))
                    if len(runs) > 1 else np.zeros_like(mean))
            maximum = max(maximum, float(np.max(mean + half)))
            axis.fill_between(times, mean - half, mean + half, color=color, alpha=.15, lw=0)
            axis.plot(times, mean, color=color, marker=marker, ms=2.0, mfc="white", mew=.5, lw=.85, ls=style,
                      label="Full ODTE-LyCO" if method == "odte_lyco" else "Without Y feedback")
        full = [r for r in results if r["scene"] == scene and r["method"] == "odte_lyco"]
        grid = np.linspace(0, last, 301)
        occupancy = np.mean([np.any([(grid >= a) & (grid < b) for a, b in r["episodes"]], axis=0)
                             for r in full], axis=0)
        axis.fill_between(grid, 0, .075 * occupancy, color="#b8b8b8",
                          transform=axis.get_xaxis_transform(), zorder=3, lw=0)
        axis.set(xlim=(0, last), xticks=np.linspace(0, last, 4), xlabel="Time (min)")
        axis.tick_params(length=2.5, pad=1.5, labelsize=6.5)
        axis.grid(alpha=.25, lw=.45)
    axes[0].set_ylim(0, maximum * 1.12)
    axes[0].set_ylabel("Mean debt (J/sensor)", y=.40)

    if legend:
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles + [Patch(fc="#b8b8b8")], labels + ["Later-event occupancy"],
                      loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.53, -.035),
                      fontsize=6.5, handlelength=2, columnspacing=1.5)


def energy_figure(results, path):
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 1.35), sharey=True)
    figure.subplots_adjust(left=.085, right=.975, bottom=.34, top=.95, wspace=.17)
    _draw_energy_axes(figure, axes, results)
    save(figure, path)


def combined_weight_energy_figure(results, config, path):
    figure = plt.figure(figsize=(7.16, 2.92))
    grid = figure.add_gridspec(2, 3, left=.075, right=.975, bottom=.18, top=.98,
                               hspace=.58, wspace=.16, height_ratios=(1.0, .90))
    weight_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    energy_axes = [figure.add_subplot(grid[1, 0])]
    energy_axes.extend(figure.add_subplot(grid[1, index], sharey=energy_axes[0])
                       for index in range(1, 3))
    duration = max(r.get("weight_slots", r["slots"]) for r in results) * config.slot_seconds / 60
    _draw_weight_axes(figure, weight_axes, results, config, duration, legend=False)
    _draw_energy_axes(figure, energy_axes, results, legend=False, panel_labels=True)

    weight_handles = [Patch(fc="#f4d2d2", label="Event"), Patch(fc="#dceaf6", label="Recovery"),
                      Patch(fc="#eef1f4", label="Post-event"),
                      Line2D([], [], color="#d62728", lw=.85, label="ODTE weight"),
                      Line2D([], [], color="#1f77b4", lw=.85, label="Energy weight"),
                      Line2D([], [], color="#555555", lw=.65, ls="--", label="Baseline")]
    figure.legend(handles=weight_handles, loc="lower center", ncol=6, frameon=False,
                  bbox_to_anchor=(.52, .47), fontsize=5.6, columnspacing=.75,
                  handlelength=1.6)
    energy_handles, energy_labels = energy_axes[0].get_legend_handles_labels()
    figure.legend(energy_handles + [Patch(fc="#b8b8b8")], energy_labels + ["Later-event occupancy"],
                  loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.53, .015),
                  fontsize=5.8, handlelength=1.7, columnspacing=1.0)
    save(figure, path)


def write_figures(results, config, output):
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    main = [r for r in results if r["experiment"] == "main"]
    energy = [r for r in results if r["experiment"] == "energy"]
    if main:
        migration_figure(main, folder / "migration_comparison.pdf")
        cdf_figure(main, folder / "event_odte_cdf_compact.pdf")
    if energy:
        weight_figure(energy, config, folder / "weight_evolution.pdf")
        combined_weight_energy_figure(energy, config, folder / "weight_y_feedback_combined.pdf")
        energy_figure(energy, folder / "y_feedback_three_debt.pdf")
    elif main:
        weight_figure(main, config, folder / "weight_evolution.pdf")
