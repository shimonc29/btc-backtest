from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_v6

OUTDIR = Path("stock_system/results")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw = {s: run_v6.fetch_yahoo(s) for s in run_v6.UNIVERSE}

    # Fixed frozen V6 parameters. No re-optimization is allowed inside these windows.
    windows = [
        ("oos_2006_2010", "2006-01-01", "2010-12-31"),
        ("oos_2011_2015", "2011-01-01", "2015-12-31"),
        ("oos_2016_2020", "2016-01-01", "2020-12-31"),
        ("oos_2021_plus", "2021-01-01", None),
    ]

    rows = []
    skipped = []
    for name, start, end in windows:
        try:
            r = run_v6.run_backtest(raw_data=raw, start_override=start, end_override=end)
            s = r["summary"]
            rows.append({
                "window": name,
                "start": s["data_start"],
                "end": s["data_end"],
                "strategy_cagr_pct": s["cagr_pct"],
                "spy_cagr_pct": s["benchmark_spy_cagr_pct"],
                "alpha_cagr_pct": s["cagr_pct"] - s["benchmark_spy_cagr_pct"],
                "strategy_max_drawdown_pct": s["max_drawdown_pct"],
                "spy_max_drawdown_pct": s["benchmark_spy_max_drawdown_pct"],
                "strategy_sharpe": s["sharpe"],
                "rebalances": s["rebalances"],
            })
        except RuntimeError as e:
            skipped.append({"window": name, "start": start, "end": end, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "v6_walkforward.csv", index=False)

    summary = {
        "method": "Sequential OOS consistency test with frozen V6 parameters; no tuning between windows",
        "price_basis": "Yahoo adjusted OHLC / adjusted SPY benchmark",
        "windows_run": int(len(df)),
        "windows_skipped": skipped,
        "all_windows_positive_cagr": bool((df["strategy_cagr_pct"] > 0).all()) if len(df) else False,
        "windows_beating_spy": int((df["alpha_cagr_pct"] > 0).sum()) if len(df) else 0,
        "windows_beating_spy_pct": float((df["alpha_cagr_pct"] > 0).mean() * 100) if len(df) else 0.0,
        "median_strategy_cagr_pct": float(df["strategy_cagr_pct"].median()) if len(df) else None,
        "median_spy_cagr_pct": float(df["spy_cagr_pct"].median()) if len(df) else None,
        "median_alpha_cagr_pct": float(df["alpha_cagr_pct"].median()) if len(df) else None,
        "worst_window_cagr_pct": float(df["strategy_cagr_pct"].min()) if len(df) else None,
        "worst_window_alpha_cagr_pct": float(df["alpha_cagr_pct"].min()) if len(df) else None,
        "median_strategy_max_drawdown_pct": float(df["strategy_max_drawdown_pct"].median()) if len(df) else None,
        "median_spy_max_drawdown_pct": float(df["spy_max_drawdown_pct"].median()) if len(df) else None,
        "windows": rows,
    }
    (OUTDIR / "v6_walkforward_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
