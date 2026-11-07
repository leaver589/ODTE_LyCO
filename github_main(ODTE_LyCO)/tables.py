"""Manuscript Tables III--V, calculated at the seed level."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from experiments import METHODS, SCENARIOS, ENERGY_METHODS

ROOT = Path(__file__).resolve().parent
FIELDS = (
    "mean_odte", "p05_odte", "event_p05_odte", "p05_min_component",
    "timely_completion_rate", "mean_delay_slots", "p95_delay_slots",
    "mean_total_queue", "total_sensor_energy", "energy_budget_ratio",
    "final_energy_debt_per_sensor", "migrations",
)
HEADERS = ("ODTE (%)", "P05 (%)", "E05 (%)", "W05 (%)", "On-time (%)",
           "Mean latency (slots)", "P95 latency (slots)", "Q (service slots)",
           "E (J)", "Ratio", "Y final (J/sensor)", "Migrations")
QUALITY = set(FIELDS[:5])
LAYERS = set(FIELDS[:3] + FIELDS[5:8] + FIELDS[11:])
ENERGY = set(FIELDS[8:11])
REPORTED = {
    "all_fast": LAYERS, "all_slow": LAYERS, "fixed_edge": LAYERS,
    "base": LAYERS | ENERGY | {"p05_min_component"},
    "dynamic": set(FIELDS[:4]) | ENERGY,
    "odte_lyco": QUALITY | ENERGY,
    "lydroo": set(FIELDS[:2]) | set(FIELDS[8:10]),
    "qeco": set(FIELDS[:2]) | set(FIELDS[8:10]),
}


def mean_ci(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return None, None
    mean = float(values.mean())
    half_width = (float(student_t.ppf(0.975, len(values) - 1)
                        * values.std(ddof=1) / np.sqrt(len(values)))
                  if len(values) > 1 else None)
    return mean, half_width


def seed_values(results, scene, method, field):
    lookup = {(r["scene"], r["method"], r["seed"]): r["metrics"] for r in results}
    seeds = sorted({r["seed"] for r in results})
    if scene == "Macro":
        # Pair the same seed across scenarios before estimating uncertainty.
        return [float(np.mean([lookup[s, method, seed][field] for s in SCENARIOS]))
                for seed in seeds]
    return [lookup[scene, method, seed][field] for seed in seeds]


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main_cells(results, scene, method, latex=False):
    cells = []
    for field in FIELDS:
        if field not in REPORTED[method]:
            cells.append(r"\textemdash" if latex else "--")
            continue
        mean, half = mean_ci(seed_values(results, scene, method, field))
        scale = 100 if field in QUALITY else 1
        digits = 3 if field in {"total_sensor_energy", "energy_budget_ratio",
                                "final_energy_debt_per_sensor"} else 2
        if method in ("lydroo", "qeco") and field == "total_sensor_energy":
            digits = 2
        ci = field in FIELDS[:3] or field == "energy_budget_ratio"
        text = f"{mean * scale:.{digits}f}"
        if ci and half is not None:
            hdigits = 4 if field == "energy_budget_ratio" else 2
            error = f"{half * scale:.{hdigits}f}"
            text = (rf"\mbox{{${text}\pm{error}$}}" if latex else f"{text} ± {error}")
        cells.append(text)
    return cells


def table_main(results, folder):
    csv_rows, tex_rows = [], []
    for scene in (*SCENARIOS, "Macro"):
        if tex_rows:
            tex_rows.append(r"\addlinespace[2pt]")
        title = "Equal-scenario macro average" if scene == "Macro" else scene
        tex_rows.append(rf"\multicolumn{{13}}{{l}}{{\textit{{{title}}}}}\\")
        for method, (_, _, label) in METHODS.items():
            csv_rows.append([scene, label, *main_cells(results, scene, method)])
            name = r"\textbf{ODTE\_LyCO}" if method == "odte_lyco" else label
            tex_rows.append(" & ".join([name, *main_cells(results, scene, method, True)]) + r"\\")
    write_csv(folder / "table_3.csv", ["Scenario", "Method", *HEADERS], csv_rows)
    seeds = len({r["seed"] for r in results})
    latex = r"""\begin{table*}[!t]
\caption{Ablation and Learning-Baseline Results}
\label{tab:ablation_results}\label{tab:rl_baseline_comparison}
\centering\footnotesize
\setlength{\tabcolsep}{2.4pt}
\renewcommand{\arraystretch}{1.00}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l*{12}{r}@{}}
\toprule
& \multicolumn{5}{c}{Entanglement quality (\%)}
& \multicolumn{3}{c}{Latency/backlog} & \multicolumn{3}{c}{Sensor energy} & Migration\\
\cmidrule(lr){2-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}\cmidrule(l){13-13}
Method & ODTE & P05 & E05 & W05 & On-time & Mean & P95 & $Q$ & $E$ (J) & Ratio & $\bar Y_f$ & Count\\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{\textwidth}\footnotesize
""" + f"Means over {seeds} seeds; " + r"""$\pm$: 95\% Student-$t$ interval half-widths.
Macro entries equally weight scenarios; percentiles are computed within runs.
Intervals are untruncated; 0.00 half-widths may reflect rounding.
Latency/$Q$: slots/service slots; $\bar Y_f$: J/sensor.
DW: dynamic weighting; dash: unreported.
\end{minipage}
\end{table*}
"""
    (folder / "table_3.tex").write_text(latex, encoding="utf-8")


Y_FIELDS = (
    (r"E05 (\%)", "E05 (%)", "event_p05_odte", 100, 2),
    (r"$\mathrm{ODTE}_p$ (\%)", "ODTE p (%)", "mean_odte_p", 100, 2),
    (r"On-time$_p$ (\%)", "On-time p (%)", "ontime_p", 100, 2),
    (r"$E_{150}$ (J)", "E150 (J)", "energy_150", 1, 2),
    (r"$E_p$ (J)", "Ep (J)", "energy_p", 1, 2),
    (r"Ratio$_{150}$", "Ratio150", "ratio_150", 1, 3),
    (r"Ratio$_p$", "Ratio p", "ratio_p", 1, 3),
    (r"$\bar Y_f$ (J/sensor)", "Y final (J/sensor)", "debt_final", 1, 3),
)


def table_energy(results, folder):
    slots = results[0]["slots"]
    first, last = slots * 0.1 / 60, round(slots * 5 / 3) * 0.1 / 60
    csv_rows, tex_rows = [], []
    for latex_label, label, field, scale, digits in Y_FIELDS:
        values = []
        for scene in SCENARIOS:
            for method in ENERGY_METHODS:
                mean, _ = mean_ci(seed_values(results, scene, method, field))
                values.append(f"{mean * scale:.{digits}f}" if mean is not None else "--")
        csv_rows.append([label.replace("150", f"{last:g}"), *values])
        tex_rows.append(" & ".join([latex_label.replace("150", f"{last:g}"), *values]) + r"\\")
    write_csv(folder / "table_4.csv",
              ["Metric", *(f"{s} {p}" for s in SCENARIOS for p in ("Full", "No-Y"))], csv_rows)
    seeds = len({r["seed"] for r in results})
    latex = r"""\begin{table}[!t]
""" + rf"\caption{{Energy-Feedback Ablation ($k=2$, {last:g} min)}}" + r"""
\label{tab:y_feedback}
\centering\footnotesize
\setlength{\tabcolsep}{1.3pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l*{6}{r}@{}}
\toprule
& \multicolumn{2}{c}{S1} & \multicolumn{2}{c}{S2} & \multicolumn{2}{c}{S3}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
Metric & Full & No-$Y$ & Full & No-$Y$ & Full & No-$Y$\\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{\columnwidth}\footnotesize
""" + f"Means of {seeds} paired seeds; $p$: {first:g}--{last:g} min. " + r"""E05: original event window;
on-time: later-burst arrivals. Energy sums eight sensors; Ratio: actual energy/period budget.
\end{minipage}
\end{table}
"""
    (folder / "table_4.tex").write_text(latex, encoding="utf-8")


def table_board(folder):
    data = json.loads((ROOT / "data/rk3566_published_measurements.json").read_text(encoding="utf-8"))
    rows, tex_rows = [], []
    for row in data["methods"]:
        values = [row["parameters"] if row["parameters"] else "--",
                  f'{row["throughput_slots_per_s"]:.2f}', f'{row["p95_ms"]:.2f}',
                  f'{row["power_w"]:.3f}']
        rows.append([row["method"], *values])
        name = r"\textbf{ODTE\_LyCO}" if row["method"] == "ODTE-LyCO" else row["method"]
        tex_rows.append(" & ".join([name, *map(str, values)]).replace("--", r"\textemdash") + r"\\")
    write_csv(folder / "table_5.csv", ["Method", "Parameters", "Throughput (slots/s)", "P95 (ms)", "Power (W)"], rows)
    latex = r"""\begin{table}[!t]
\caption{Controller Overhead on RK3566}
\label{tab:controller_overhead}
\centering\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
Method & Parameters & \shortstack{Throughput(slots/s)}
& \shortstack{P95(ms)} & \shortstack{Power(W)}\\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{\columnwidth}\footnotesize
Parameters: inference weights and biases; -: no trainable weights or biases.
Power: total board power under the test workload.
\end{minipage}
\end{table}
"""
    (folder / "table_5.tex").write_text(latex, encoding="utf-8")


def write_tables(results, output):
    folder = output / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    main = [r for r in results if r["experiment"] == "main"]
    energy = [r for r in results if r["experiment"] == "energy"]
    if main:
        table_main(main, folder)
    if energy:
        table_energy(energy, folder)
    table_board(folder)
