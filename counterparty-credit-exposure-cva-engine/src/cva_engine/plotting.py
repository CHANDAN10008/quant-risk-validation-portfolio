"""Small dependency-free SVG charts for generated analysis reports."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PALETTE = {
    "netted_uncollateralized": "#2166ac",
    "unnetted_uncollateralized": "#d9822b",
    "netted_collateralized": "#238b45",
    "unnetted_collateralized": "#c43c39",
}


def _label(text: str, x: float, y: float, size: int = 12, anchor: str = "middle",
           color: str = "#263238", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"'
        f' font-family="Arial, sans-serif" font-size="{size}" fill="{color}"{transform}>'
        f'{escape(text)}</text>'
    )


def _write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def _line_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, Any]],
    y_zero: bool = False,
) -> None:
    width, height = 1120, 590
    left, top, bottom, right = 92.0, 76.0, 500.0, 800.0
    all_x = np.concatenate([np.asarray(item["x"], dtype=float) for item in series])
    all_y = np.concatenate([np.asarray(item["y"], dtype=float) for item in series])
    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    if xmax <= xmin:
        xmax = xmin + 1.0
    if y_zero:
        ymin = min(0.0, ymin)
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.06
    if not y_zero:
        ymin -= pad
    ymax += pad

    def px(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - ymin) / (ymax - ymin) * (bottom - top)

    body = [_label(title, width / 2, 34, 20, color="#172b4d")]
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = py(value)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e6e9ed"/>')
        body.append(_label(f"{value:,.3g}", left - 10, y + 4, 11, anchor="end"))
    for i in range(6):
        value = xmin + (xmax - xmin) * i / 5
        x = px(value)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#f1f2f4"/>')
        body.append(_label(f"{value:,.3g}", x, bottom + 22, 11))
    body.extend(
        [
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#52606d"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#52606d"/>',
            _label(x_label, (left + right) / 2, 558, 13),
            _label(y_label, 22, (top + bottom) / 2, 13, rotate=-90),
        ]
    )
    for item in series:
        x_values = np.asarray(item["x"], dtype=float)
        y_values = np.asarray(item["y"], dtype=float)
        points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(x_values, y_values))
        dash = ' stroke-dasharray="7 5"' if item.get("dash") else ""
        body.append(
            f'<polyline points="{points}" fill="none" stroke="{item["color"]}" '
            f'stroke-width="2.2"{dash}/>'
        )
        if item.get("markers"):
            for x, y in zip(x_values, y_values):
                body.append(
                    f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" '
                    f'fill="{item["color"]}"/>'
                )
    legend_x, legend_y = 830.0, 86.0
    for index, item in enumerate(series):
        y = legend_y + index * 28
        dash = ' stroke-dasharray="7 5"' if item.get("dash") else ""
        body.append(
            f'<line x1="{legend_x}" y1="{y:.1f}" x2="{legend_x + 27}" y2="{y:.1f}" '
            f'stroke="{item["color"]}" stroke-width="2.2"{dash}/>'
        )
        body.append(_label(item["label"], legend_x + 36, y + 4, 11, anchor="start"))
    _write_svg(path, width, height, body)


def _bar_chart(path: Path, summary: pd.DataFrame) -> None:
    width, height = 900, 540
    left, top, bottom, right = 80.0, 74.0, 420.0, 850.0
    values = summary["CVA"].to_numpy(dtype=float) / 1e3
    maximum = max(float(np.max(values)), 1.0) * 1.12
    body = [_label("Unilateral CVA by netting and collateral scenario", width / 2, 36, 19)]
    for i in range(6):
        value = maximum * i / 5
        y = bottom - value / maximum * (bottom - top)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e6e9ed"/>')
        body.append(_label(f"{value:,.3g}", left - 10, y + 4, 11, anchor="end"))
    body.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#52606d"/>')
    step = (right - left) / max(len(summary), 1)
    bar_width = step * 0.52
    for i, row in summary.reset_index(drop=True).iterrows():
        x = left + step * (i + 0.5)
        value = max(float(row["CVA"]) / 1e3, 0.0)
        y = bottom - value / maximum * (bottom - top)
        scenario = str(row["scenario"])
        color = PALETTE.get(scenario, "#2166ac")
        body.append(
            f'<rect x="{x - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bottom - y:.1f}" fill="{color}"/>'
        )
        body.append(_label(f"{value:,.1f}", x, y - 9, 11))
        body.append(_label(scenario.replace("_", " "), x, bottom + 38, 10))
    body.append(_label("CVA (USD thousands)", 23, (top + bottom) / 2, 12, rotate=-90))
    _write_svg(path, width, height, body)


def _sensitivity_chart(path: Path, sensitivity: pd.DataFrame) -> None:
    width, height = 1100, 900
    body = [_label("Parameter sensitivity: collateralized netted CVA", width / 2, 32, 20)]
    parameters = list(sensitivity["parameter"].drop_duplicates())
    panel_w, panel_h = 500.0, 195.0
    for pidx, parameter in enumerate(parameters):
        col, row_idx = pidx % 2, pidx // 2
        x0 = 70.0 + col * 525.0
        y0 = 75.0 + row_idx * 205.0
        x1, y1 = x0 + panel_w - 55.0, y0 + panel_h - 42.0
        group = sensitivity[sensitivity["parameter"] == parameter].sort_values("value")
        xs = group["value"].to_numpy(dtype=float)
        ys = group["netted_collateralized_cva"].to_numpy(dtype=float) / 1e3
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = min(0.0, float(ys.min())), max(float(ys.max()), 1.0)
        if xmax <= xmin:
            xmax = xmin + 1.0
        ymax *= 1.08
        def px(x: float) -> float:
            return x0 + 42.0 + (x - xmin) / (xmax - xmin) * (x1 - x0 - 55.0)
        def py(y: float) -> float:
            return y1 - y / ymax * (y1 - y0 - 30.0)
        body.append(_label(parameter.replace("_", " "), x0 + panel_w / 2 - 25, y0, 14, color="#172b4d"))
        body.append(f'<line x1="{x0 + 42:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#52606d"/>')
        body.append(f'<line x1="{x0 + 42:.1f}" y1="{y0 + 18:.1f}" x2="{x0 + 42:.1f}" y2="{y1:.1f}" stroke="#52606d"/>')
        body.append(_label(f"{ymax:.3g}", x0 + 34, y0 + 22, 9, anchor="end"))
        body.append(_label("0", x0 + 34, y1 + 3, 9, anchor="end"))
        points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        body.append(f'<polyline points="{points}" fill="none" stroke="#238b45" stroke-width="2.2"/>')
        for x, y in zip(xs, ys):
            body.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="#238b45"/>')
            body.append(_label(f"{x:g}", px(x), y1 + 17, 9))
        body.append(_label("Input value", (x0 + x1) / 2, y1 + 34, 9))
        body.append(_label("CVA (USD thousands)", x0 + 8, (y0 + y1) / 2, 9, rotate=-90))
    _write_svg(path, width, height, body)


def write_analysis_plots(
    profiles: dict[str, pd.DataFrame],
    scenario_summary: pd.DataFrame,
    convergence: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output_dir: Path,
) -> None:
    colors = PALETTE
    exposure_series: list[dict[str, Any]] = []
    for name, profile in profiles.items():
        exposure_series.append(
            {
                "label": name.replace("_", " ") + " EE",
                "x": profile["time_years"],
                "y": profile["EE"] / 1e6,
                "color": colors[name],
            }
        )
        pfe_column = next(column for column in profile.columns if column.startswith("PFE_"))
        exposure_series.append(
            {
                "label": name.replace("_", " ") + " PFE",
                "x": profile["time_years"],
                "y": profile[pfe_column] / 1e6,
                "color": colors[name],
                "dash": True,
            }
        )
    _line_chart(
        output_dir / "exposure_profiles.svg",
        "Expected exposure and 97.5% PFE",
        "Time (years)",
        "Exposure (USD millions)",
        exposure_series,
        y_zero=True,
    )
    _bar_chart(output_dir / "cva_scenarios.svg", scenario_summary)
    counts = convergence["path_count"].to_numpy(dtype=float)
    _line_chart(
        output_dir / "pricing_convergence.svg",
        "Swap valuation convergence",
        "Number of paths",
        "Swap value (USD millions)",
        [
            {
                "label": "Monte Carlo estimate",
                "x": counts,
                "y": convergence["monte_carlo_swap_value"] / 1e6,
                "color": "#2166ac",
                "markers": True,
            },
            {
                "label": "Analytic curve value",
                "x": counts,
                "y": np.full_like(counts, convergence["analytic_swap_value"].iloc[0] / 1e6),
                "color": "#c43c39",
                "dash": True,
            },
        ],
    )
    _sensitivity_chart(output_dir / "parameter_sensitivity.svg", sensitivity)
