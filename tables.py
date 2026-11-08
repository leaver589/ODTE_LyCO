"""Manuscript Tables III--IV, calculated at the seed level."""

import csv

import numpy as np
from scipy.stats import t as student_t

from experiments import METHODS, SCENARIOS, ENERGY_METHODS

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
            name = r"\textbf{ODTE-LyCO}" if method == "odte_lyco" else label
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
DW: dynamic weighting. Every method includes all twelve metrics.
\end{minipage}
\end{table*}
"""
    (folder / "table_3.tex").write_text(latex, encoding="utf-8")


Y_FIELDS = (
    ("E05 (%)", "event_p05_odte", 100, 2),
    ("On-time p (%)", "ontime_p", 100, 2),
    ("E150 (J)", "energy_150", 1, 2),
    ("Y final (J/sensor)", "debt_final", 1, 3),
)


def table_energy(results, folder):
    duration = round(results[0]["slots"] * 5 / 3) * 0.1 / 60
    csv_rows, tex_rows = [], []
    for scene in SCENARIOS:
        for method, label in zip(ENERGY_METHODS, ("ODTE-LyCO", "No-Y")):
            values = []
            for _, field, scale, digits in Y_FIELDS:
                mean, _ = mean_ci(seed_values(results, scene, method, field))
                if mean is None:
                    raise ValueError(f"Later-burst arrivals required for {scene}, {method}; increase slots.")
                values.append(f"{mean * scale:.{digits}f}")
            csv_rows.append([scene, label, *values])
            tex_label = "No-$Y$" if method == "no_y" else label
            tex_rows.append(" & ".join([scene, tex_label, *values]) + r"\\")
    write_csv(folder / "table_4.csv",
              ["Scenario", "Method", *(v[0].replace("150", f"{duration:g}") for v in Y_FIELDS)], csv_rows)
    seeds = len({r["seed"] for r in results})
    latex = r"""\begin{table}[!t]
""" + rf"\caption{{Energy-Feedback Ablation ({duration:g} min)}}" + r"""
\label{tab:y_feedback}
\centering\footnotesize
\setlength{\tabcolsep}{1.4pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrrr@{}}
\toprule
Scenario & Method & E05(\%) & On-time$_p$(\%) & $E_{150}$(J) & $\bar Y_f$(J/sensor)\\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{\columnwidth}\footnotesize
""" + f"Means over {seeds} paired seeds. " + r"""E05: original event window;
On-time$_p$: requests arriving in later bursts. Energy sums eight sensors;
$\bar Y_f$: terminal mean energy debt per sensor.
\end{minipage}
\end{table}
"""
    latex = latex.replace("E_{150}", f"E_{{{duration:g}}}")
    (folder / "table_4.tex").write_text(latex, encoding="utf-8")


def write_tables(results, output):
    folder = output / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    main = [r for r in results if r["experiment"] == "main"]
    energy = [r for r in results if r["experiment"] == "energy"]
    if main:
        table_main(main, folder)
    if energy:
        table_energy(energy, folder)
