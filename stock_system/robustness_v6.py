from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_v6
from strategy_v6 import DEFAULTS

OUTDIR = Path("stock_system/results")


def metric_row(name: str, result: dict, group: str, params: dict | None = None) -> dict:
    s = result["summary"]
    return {
        "group": group,
        "test": name,
        "cagr_pct": s["cagr_pct"],
        "return_pct": s["return_pct"],
        "max_drawdown_pct": s["max_drawdown_pct"],
        "sharpe": s["sharpe"],
        "calmar": s["calmar"],
        "spy_cagr_pct": s["benchmark_spy_cagr_pct"],
        "spy_max_drawdown_pct": s["benchmark_spy_max_drawdown_pct"],
        "rebalances": s["rebalances"],
        "turnover_events": s["turnover_events"],
        "params": json.dumps(params or {}, sort_keys=True),
        "start": s["data_start"],
        "end": s["data_end"],
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Fetch market data once; all robustness runs reuse the exact same dataset.
    raw = {s: run_v6.fetch_yahoo(s) for s in run_v6.UNIVERSE}
    rows: list[dict] = []

    baseline = run_v6.run_backtest(raw_data=raw)
    rows.append(metric_row("baseline_audited", baseline, "baseline"))

    # Cost stress: same logic, harsher execution assumptions.
    for mult in [2, 3, 5]:
        p = {
            "fee_rate": DEFAULTS["fee_rate"] * mult,
            "slippage_rate": DEFAULTS["slippage_rate"] * mult,
        }
        r = run_v6.run_backtest(params=p, raw_data=raw)
        rows.append(metric_row(f"cost_{mult}x", r, "cost", p))

    # Parameter neighbourhood tests. Change one structural parameter at a time,
    # then several combined variants. This is stability testing, not optimization.
    variants = [
        ("sma_slow_180", {"sma_slow": 180}),
        ("sma_slow_220", {"sma_slow": 220}),
        ("sma_fast_40", {"sma_fast": 40}),
        ("sma_fast_60", {"sma_fast": 60}),
        ("mom_fast_42", {"mom_fast": 42}),
        ("mom_fast_84", {"mom_fast": 84}),
        ("mom_mid_105", {"mom_mid": 105}),
        ("mom_mid_147", {"mom_mid": 147}),
        ("breadth_strong_055", {"breadth_strong": 0.55}),
        ("breadth_strong_065", {"breadth_strong": 0.65}),
        ("breadth_weak_030", {"breadth_weak": 0.30}),
        ("breadth_weak_040", {"breadth_weak": 0.40}),
        ("top_n_1", {"top_n": 1}),
        ("top_n_3", {"top_n": 3}),
        ("bear_spy_010", {"bear_spy": 0.10}),
        ("bear_spy_030", {"bear_spy": 0.30}),
        ("overlay_less", {"bull_spy": 0.70, "bull_qqq": 0.15, "bull_sat": 0.15}),
        ("overlay_more", {"bull_spy": 0.50, "bull_qqq": 0.25, "bull_sat": 0.25}),
        ("slow_conservative", {"sma_slow": 220, "mom_fast": 84, "breadth_strong": 0.65}),
        ("fast_aggressive", {"sma_slow": 180, "mom_fast": 42, "breadth_strong": 0.55}),
    ]
    for name, p in variants:
        r = run_v6.run_backtest(params=p, raw_data=raw)
        rows.append(metric_row(name, r, "parameter", p))

    # Time-slice tests. Indicators are computed on full history before slicing,
    # preserving warm-up and avoiding the split-reset problem.
    periods = [
        ("dev_2001_2015", "2001-01-01", "2015-12-31"),
        ("test_2016_2020", "2016-01-01", "2020-12-31"),
        ("holdout_2021_plus", "2021-01-01", None),
        ("post_gfc_2010_plus", "2010-01-01", None),
        ("recent_2018_plus", "2018-01-01", None),
    ]
    for name, start, end in periods:
        r = run_v6.run_backtest(raw_data=raw, start_override=start, end_override=end)
        rows.append(metric_row(name, r, "period", {"start": start, "end": end}))

    # Known stress windows. These are scenario slices, not independent OOS tests.
    stress = [
        ("dotcom_aftermath", "2001-01-01", "2003-12-31"),
        ("gfc", "2007-01-01", "2009-12-31"),
        ("covid", "2019-01-01", "2021-12-31"),
        ("inflation_bear", "2021-01-01", "2023-12-31"),
    ]
    for name, start, end in stress:
        r = run_v6.run_backtest(raw_data=raw, start_override=start, end_override=end)
        rows.append(metric_row(name, r, "stress_period", {"start": start, "end": end}))

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "v6_robustness.csv", index=False)

    param_df = df[df["group"] == "parameter"]
    cost_df = df[df["group"] == "cost"]
    period_df = df[df["group"] == "period"]
    summary = {
        "baseline": baseline["summary"],
        "parameter_tests": int(len(param_df)),
        "parameter_positive_cagr_pct": float((param_df["cagr_pct"] > 0).mean() * 100) if len(param_df) else 0.0,
        "parameter_beat_spy_cagr_pct": float((param_df["cagr_pct"] > param_df["spy_cagr_pct"]).mean() * 100) if len(param_df) else 0.0,
        "parameter_median_cagr_pct": float(param_df["cagr_pct"].median()) if len(param_df) else None,
        "parameter_worst_cagr_pct": float(param_df["cagr_pct"].min()) if len(param_df) else None,
        "parameter_median_max_drawdown_pct": float(param_df["max_drawdown_pct"].median()) if len(param_df) else None,
        "cost_tests_all_positive": bool((cost_df["cagr_pct"] > 0).all()) if len(cost_df) else False,
        "cost_5x_cagr_pct": float(cost_df.loc[cost_df["test"] == "cost_5x", "cagr_pct"].iloc[0]) if (cost_df["test"] == "cost_5x").any() else None,
        "period_tests_all_positive": bool((period_df["cagr_pct"] > 0).all()) if len(period_df) else False,
        "holdout_2021_plus_cagr_pct": float(df.loc[df["test"] == "holdout_2021_plus", "cagr_pct"].iloc[0]),
        "holdout_2021_plus_spy_cagr_pct": float(df.loc[df["test"] == "holdout_2021_plus", "spy_cagr_pct"].iloc[0]),
        "notes": [
            "V6 execution timing audited before robustness: signal at month-end close, fill next session open, pre-open sizing uses prior close only.",
            "Parameter variants are stability tests and must not be used to select a tuned winner after seeing these results.",
            "SPY benchmark uses raw Yahoo close in this research implementation; dividends are not included in either benchmark total-return claim.",
        ],
    }
    (OUTDIR / "v6_robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
