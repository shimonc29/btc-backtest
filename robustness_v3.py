import argparse
import itertools
import pandas as pd
from backtest_v3 import backtest_v3, load_csv


def with_time(df):
    cols = {c.lower(): c for c in df.columns}
    key = cols.get('open time') or cols.get('timestamp') or cols.get('datetime')
    if not key:
        raise ValueError("CSV needs Open time/timestamp/datetime")
    out = df.copy()
    out['_ts'] = pd.to_datetime(out[key], errors='coerce')
    return out.dropna(subset=['_ts']).sort_values('_ts').reset_index(drop=True)


def slice_period(df, start=None, end=None):
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df['_ts'] >= pd.Timestamp(start)
    if end:
        mask &= df['_ts'] < pd.Timestamp(end)
    return df.loc[mask].copy()


def run_grid(df, equity, label, fee_rate, slippage_rate):
    rows = []
    for breakout, stop_atr, trail_atr in itertools.product(
        [15, 20, 25, 30], [1.5, 2.0, 2.5], [2.5, 3.0, 3.5]
    ):
        _, s = backtest_v3(
            df,
            initial_equity=equity,
            breakout_lookback=breakout,
            initial_stop_atr=stop_atr,
            trail_atr=trail_atr,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        rows.append({
            'period': label,
            'breakout_lookback': breakout,
            'initial_stop_atr': stop_atr,
            'trail_atr': trail_atr,
            'fee_rate': fee_rate,
            'slippage_rate': slippage_rate,
            **s,
        })
    return pd.DataFrame(rows)


def summarize(grid, name):
    valid = grid[grid['trades'] >= 10].copy()
    if valid.empty:
        return {'set': name, 'combos': len(grid), 'valid_combos': 0}
    return {
        'set': name,
        'combos': len(grid),
        'valid_combos': len(valid),
        'positive_return_pct': (valid['return_pct'] > 0).mean() * 100,
        'pf_above_1_pct': (valid['profit_factor'] > 1).mean() * 100,
        'pf_above_1_2_pct': (valid['profit_factor'] > 1.2).mean() * 100,
        'median_return_pct': valid['return_pct'].median(),
        'median_profit_factor': valid['profit_factor'].median(),
        'median_avg_r': valid['avg_r'].median(),
        'median_max_dd_pct': valid['max_drawdown_pct'].median(),
        'worst_return_pct': valid['return_pct'].min(),
        'best_return_pct': valid['return_pct'].max(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--equity', type=float, default=2000.0)
    args = p.parse_args()

    raw = with_time(load_csv(args.csv))
    dev = slice_period(raw, '2018-01-01', '2025-01-01')
    oos = slice_period(raw, '2025-01-01', None)

    # Parameter robustness is judged on DEV only. OOS is reported as an audit,
    # not used to select the winning parameter combination.
    dev_base = run_grid(dev, args.equity, 'DEV_BASE_COST', 0.001, 0.0002)
    dev_stress = run_grid(dev, args.equity, 'DEV_2X_COST', 0.002, 0.0004)
    oos_audit = run_grid(oos, args.equity, 'OOS_AUDIT_BASE_COST', 0.001, 0.0002)

    all_rows = pd.concat([dev_base, dev_stress, oos_audit], ignore_index=True)
    all_rows.to_csv('v3_robustness_grid.csv', index=False)

    summary = pd.DataFrame([
        summarize(dev_base, 'DEV_BASE_COST'),
        summarize(dev_stress, 'DEV_2X_COST'),
        summarize(oos_audit, 'OOS_AUDIT_BASE_COST'),
    ])
    summary.to_csv('v3_robustness_summary.csv', index=False)

    baseline = all_rows[
        (all_rows.breakout_lookback == 20) &
        (all_rows.initial_stop_atr == 2.0) &
        (all_rows.trail_atr == 3.0)
    ].copy()
    baseline.to_csv('v3_robustness_baseline.csv', index=False)

    print('\n=== V3 ROBUSTNESS SUMMARY ===')
    print(summary.to_string(index=False))
    print('\n=== FROZEN BASELINE 20/2.0/3.0 ===')
    cols = ['period','trades','return_pct','profit_factor','avg_r','max_drawdown_pct','final_equity']
    print(baseline[cols].to_string(index=False))


if __name__ == '__main__':
    main()
