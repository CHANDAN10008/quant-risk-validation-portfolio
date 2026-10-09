"""Summary tables, metadata, and compact backtest plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import christoffersen_independence, kupiec_pof, traffic_light


def summarize(forecasts: pd.DataFrame, *, confidence: float) -> pd.DataFrame:
    tail = 1.0 - confidence
    rows = []
    for model in ("historical", "parametric"):
        exception_series = forecasts[f"{model}_exception"].to_numpy(dtype=bool)
        pof = kupiec_pof(exception_series, tail)
        ind = christoffersen_independence(exception_series)
        rows.append(
            {
                "model": model,
                "confidence": confidence,
                "backtest_days": len(forecasts),
                "exceptions": pof["exceptions"],
                "exception_rate": pof["exception_rate"],
                "average_var": forecasts[f"{model}_var"].mean(),
                "kupiec_lr": pof["lr_stat"],
                "kupiec_p_value": pof["p_value"],
                "christoffersen_ind_lr": ind["lr_stat"],
                "christoffersen_ind_p_value": ind["p_value"],
                "traffic_light_250d": traffic_light(exception_series[-250:], confidence=confidence),
            }
        )
    return pd.DataFrame(rows)


def write_run(
    forecasts: pd.DataFrame,
    output_dir: str | Path,
    *,
    confidence: float,
    lookback: int,
    data_type: str,
    seed: int | None,
    input_file: str | None,
) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(forecasts, confidence=confidence)
    forecasts.to_csv(out / "forecasts.csv", index_label="date", float_format="%.8f")
    summary.to_csv(out / "backtest_summary.csv", index=False, float_format="%.8f")
    metadata = {
        "data_type": data_type,
        "seed": seed,
        "input_file": input_file,
        "input_observations": int(len(forecasts) + lookback),
        "backtest_observations": int(len(forecasts)),
        "lookback": lookback,
        "confidence": confidence,
        "return_units": "decimal simple returns; VaR as positive loss fraction",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _plot(forecasts, out / "var_backtest.svg", confidence=confidence)
    return summary


def _plot(forecasts: pd.DataFrame, path: Path, *, confidence: float) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt is not None:
        fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
        ax.plot(forecasts.index, -forecasts["return"] * 100, color="#25324a", lw=0.8, alpha=0.65, label="Realized loss")
        ax.plot(forecasts.index, forecasts["historical_var"] * 100, color="#187f72", lw=1.2, label="Historical VaR")
        ax.plot(forecasts.index, forecasts["parametric_var"] * 100, color="#d98936", lw=1.2, label="Parametric VaR")
        for model, color in (("historical", "#187f72"), ("parametric", "#d98936")):
            breach = forecasts[f"{model}_exception"].astype(bool)
            ax.scatter(forecasts.index[breach], (-forecasts.loc[breach, "return"] * 100), s=18, color=color, edgecolor="white", linewidth=0.35, zorder=3)
        ax.axhline(0, color="#9aa4b2", lw=0.7)
        ax.set_title(f"Rolling one-day {confidence:.1%} VaR backtest")
        ax.set_ylabel("Loss / VaR (% of portfolio value)")
        ax.set_xlabel("Forecast date")
        ax.grid(True, color="#d9dee7", lw=0.55, alpha=0.8)
        ax.legend(frameon=False, ncol=3, loc="upper left")
        fig.savefig(path, format="svg")
        plt.close(fig)
        return

    width, height = 1200, 520
    left, right, top, bottom = 76, 22, 54, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    realized = -forecasts["return"].to_numpy(dtype=float) * 100
    historical = forecasts["historical_var"].to_numpy(dtype=float) * 100
    parametric = forecasts["parametric_var"].to_numpy(dtype=float) * 100
    lower = min(0.0, float(np.min(realized)), float(np.min(historical)), float(np.min(parametric)))
    upper = max(float(np.max(realized)), float(np.max(historical)), float(np.max(parametric)))
    span = max(upper - lower, 1e-9)
    lower -= span * 0.06
    upper += span * 0.08

    def xy(i: int, value: float) -> tuple[float, float]:
        x = left + plot_w * i / max(len(realized) - 1, 1)
        y = top + plot_h * (upper - value) / (upper - lower)
        return x, y

    def polyline(values: np.ndarray, color: str, stroke_width: float, opacity: float = 1.0) -> str:
        points = " ".join(f"{x:.1f},{y:.1f}" for i, val in enumerate(values) for x, y in [xy(i, float(val))])
        return f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}" points="{points}" />'

    grid: list[str] = []
    for tick in np.linspace(lower, upper, 6):
        y = xy(0, float(tick))[1]
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d9dee7" stroke-width="1" />')
        grid.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#536176">{tick:.1f}%</text>')
    breach_marks: list[str] = []
    for model, color in (("historical", "#187f72"), ("parametric", "#d98936")):
        mask = forecasts[f"{model}_exception"].to_numpy(dtype=bool)
        for i in np.flatnonzero(mask):
            x, y = xy(int(i), float(realized[i]))
            breach_marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" stroke="#fff" stroke-width="1" />')
    dates = pd.DatetimeIndex(forecasts.index)
    labels = [dates[0].strftime("%Y-%m-%d"), dates[len(dates)//2].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")]
    x_positions = [left, left + plot_w/2, left + plot_w]
    date_marks = [f'<text x="{x:.1f}" y="{height-25}" text-anchor="middle" font-size="12" fill="#536176">{label}</text>' for x, label in zip(x_positions, labels)]
    legend = (
        '<line x1="90" y1="26" x2="116" y2="26" stroke="#25324a" stroke-width="2"/><text x="122" y="30" font-size="12">Realized loss</text>'
        '<line x1="270" y1="26" x2="296" y2="26" stroke="#187f72" stroke-width="2"/><text x="302" y="30" font-size="12">Historical VaR</text>'
        '<line x1="455" y1="26" x2="481" y2="26" stroke="#d98936" stroke-width="2"/><text x="487" y="30" font-size="12">Parametric VaR</text>'
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width/2}" y="18" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" font-weight="600" fill="#25324a">Rolling one-day {confidence:.1%} VaR backtest</text>
<g font-family="Arial,sans-serif">{legend}{''.join(grid)}
{polyline(realized, '#25324a', 1.0, 0.65)}{polyline(historical, '#187f72', 1.4)}{polyline(parametric, '#d98936', 1.4)}{''.join(breach_marks)}{''.join(date_marks)}
<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-size="12" fill="#536176">Loss / VaR (% of portfolio value)</text>
</g></svg>'''
    path.write_text(svg, encoding="utf-8")
