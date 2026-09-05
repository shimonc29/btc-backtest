import argparse
import pandas as pd
from backtest_v3 import backtest_v3, load_csv


def period_slice(df, start=None, end=None):
    cols = {c.lower(): c for c in df.columns}
    if 'open time' in cols:
        ts = pd.to_datetime(df[cols['open time']], errors='coerce')
    elif 'timestamp' in cols:
        ts = pd.to_datetime(df[cols['timestamp']], errors='coerce')
    elif 'datetime' in cols:
        ts = pd.to_datetime(df[cols['datetime']], errors='coerce')
    else:
        raise ValueError("CSV needs 'Open time', 'timestamp', or 'datetime'")

    mask = pd.Series(True, index=df.index)
    if start:
        mask &= ts >= pd.Timestamp(start)
    if end:
        mask &= ts < pd.Timestamp(end)
    return df.loc[mask].copy()


def buy_hold_return_pct(df):
    if len(df) < 2:
        return float('nan')
    return (df.iloc[-1]['close'] / df.iloc[0]['open'] - 1) * 100


def run_period(name, df, equity):
    trades, stats = backtest_v3(df, initial_equity=equity)
    return {
        'period': name,
        'bars': len(df),
        'trades': stats['trades'],
        'win_rate_pct': stats['win_rate_pct'],
        'profit_factor': stats['profit_factor'],
        'avg_r': stats['avg_r'],
        'return_pct': stats['return_pct'],
        'max_drawdown_pct': stats['max_drawdown_pct'],
        'max_consecutive_losses': stats['max_consecutive_losses'],
        'final_equity': stats['final_equity'],
        'buy_hold_return_pct': buy_hold_return_pct(df),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--equity', type=float, default=2000.0)
    args = p.parse_args()

    df = load_csv(args.csv)

    # Keep 2025+ protected as out-of-sample. Do not tune V3 using it.
    periods = [
        ('2018-2020 DEV', '2018-01-01', '2021-01-01'),
        ('2021-2022 DEV', '2021-01-01', '2023-01-01'),
        ('2023-2024 DEV', '2023-01-01', '2025-01-01'),
        ('2025+ OUT-OF-SAMPLE', '2025-01-01', None),
        ('FULL', None, None),
    ]

    rows = []
    for name, start, end in periods:
        part = period_slice(df, start, end) if (start or end) else df
        if len(part) > 250:
            rows.append(run_period(name, part, args.equity))

    out = pd.DataFrame(rows)
    pd.set_option('display.max_columns', None)
    print(out.to_string(index=False))
    out.to_csv('research_v3_summary.csv', index=False)


if __name__ == '__main__':
    main()
