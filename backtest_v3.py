import argparse
import pandas as pd
import numpy as np


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def atr(df, period=14):
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def prepare(df, breakout_lookback=20):
    df = df.copy()
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    df['atr14'] = atr(df, 14)
    df['atr_pct'] = df['atr14'] / df['close']
    df['ema200_prev12'] = df['ema200'].shift(12)
    df['breakout'] = df['high'].rolling(breakout_lookback).max().shift(1)
    df['atr_q40'] = df['atr_pct'].rolling(100).quantile(0.40)
    df['atr_q90'] = df['atr_pct'].rolling(100).quantile(0.90)
    return df


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {'open', 'high', 'low', 'close'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    return df


def backtest_v3(
    df,
    initial_equity=2000.0,
    risk_pct=0.005,
    breakout_lookback=20,
    initial_stop_atr=2.0,
    trail_atr=3.0,
    time_stop_bars=30,
    fee_rate=0.001,
    slippage_rate=0.0002,
):
    df = prepare(df, breakout_lookback=breakout_lookback).reset_index(drop=True)
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    trades = []
    i = max(220, breakout_lookback + 200)

    while i < len(df) - 1:
        row = df.iloc[i]
        needed = [row['atr14'], row['breakout'], row['atr_q40'], row['atr_q90'], row['ema200_prev12']]
        if not all(np.isfinite(x) for x in needed):
            i += 1
            continue

        trend_ok = row['close'] > row['ema200'] and row['ema50'] > row['ema200'] and row['ema200'] > row['ema200_prev12']
        breakout_ok = row['close'] > row['breakout']
        volatility_ok = row['atr_q40'] <= row['atr_pct'] <= row['atr_q90']
        if not (trend_ok and breakout_ok and volatility_ok):
            i += 1
            continue

        entry_i = i + 1
        entry = df.iloc[entry_i]['open'] * (1 + slippage_rate)
        stop_distance = initial_stop_atr * row['atr14']
        if stop_distance <= 0:
            i += 1
            continue

        initial_stop = entry - stop_distance
        trail_stop = initial_stop
        risk_dollars = equity * risk_pct
        qty = risk_dollars / stop_distance
        notional = qty * entry
        if notional > equity:
            qty = equity / entry
            notional = qty * entry
            risk_dollars = qty * stop_distance

        entry_fee = notional * fee_rate
        exit_i = None
        exit_price = None
        outcome = None
        highest_close = entry
        last_i = min(entry_i + time_stop_bars - 1, len(df) - 1)

        for j in range(entry_i, last_i + 1):
            bar = df.iloc[j]
            if j > entry_i:
                prev = df.iloc[j - 1]
                highest_close = max(highest_close, prev['close'])
                if np.isfinite(prev['atr14']):
                    trail_stop = max(trail_stop, highest_close - trail_atr * prev['atr14'])

            if bar['low'] <= trail_stop:
                exit_i = j
                exit_price = trail_stop * (1 - slippage_rate)
                outcome = 'TRAIL_STOP' if trail_stop > initial_stop else 'INITIAL_STOP'
                break

            bars_held = j - entry_i + 1
            if bars_held >= 2 and bar['close'] < bar['ema20']:
                exit_i = j
                exit_price = bar['close'] * (1 - slippage_rate)
                outcome = 'EMA20_EXIT'
                break

        if exit_i is None:
            exit_i = last_i
            exit_price = df.iloc[exit_i]['close'] * (1 - slippage_rate)
            outcome = 'TIME'

        exit_notional = qty * exit_price
        exit_fee = exit_notional * fee_rate
        gross_pnl = qty * (exit_price - entry)
        net_pnl = gross_pnl - entry_fee - exit_fee
        r_multiple = net_pnl / risk_dollars if risk_dollars > 0 else 0.0

        equity_before = equity
        equity += net_pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak else 0)

        trades.append({
            'signal_index': i, 'entry_index': entry_i, 'exit_index': exit_i,
            'entry': entry, 'exit': exit_price, 'initial_stop': initial_stop,
            'final_trail_stop': trail_stop, 'qty': qty, 'notional': notional,
            'entry_fee': entry_fee, 'exit_fee': exit_fee, 'outcome': outcome,
            'net_pnl': net_pnl, 'r_multiple': r_multiple,
            'equity_before': equity_before, 'equity_after': equity,
        })
        i = exit_i + 1

    t = pd.DataFrame(trades)
    if t.empty:
        return t, {'initial_equity':initial_equity,'final_equity':equity,'return_pct':0.0,'trades':0,'win_rate_pct':0.0,'profit_factor':np.nan,'avg_r':np.nan,'max_drawdown_pct':0.0,'max_consecutive_losses':0,'total_fees_est':0.0}

    wins = t[t.net_pnl > 0]
    losses = t[t.net_pnl <= 0]
    gp = wins.net_pnl.sum(); gl = -losses.net_pnl.sum()
    pf = gp / gl if gl > 0 else np.inf
    cur = max_losses = 0
    for is_loss in (t.net_pnl <= 0):
        cur = cur + 1 if is_loss else 0
        max_losses = max(max_losses, cur)

    stats = {
        'initial_equity': initial_equity,
        'final_equity': equity,
        'return_pct': (equity / initial_equity - 1) * 100,
        'trades': len(t),
        'win_rate_pct': (t.net_pnl > 0).mean() * 100,
        'profit_factor': pf,
        'avg_r': t.r_multiple.mean(),
        'max_drawdown_pct': max_dd * 100,
        'max_consecutive_losses': max_losses,
        'total_fees_est': float((t['entry_fee'] + t['exit_fee']).sum()),
    }
    return t, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--equity', type=float, default=2000.0)
    args = p.parse_args()
    df = load_csv(args.csv)
    trades, stats = backtest_v3(df, initial_equity=args.equity)
    print('\n=== BTC Momentum Breakout 4H V3 ===')
    for k, v in stats.items(): print(f'{k}: {v}')
    trades.to_csv('btc_momentum_v3_trades.csv', index=False)


if __name__ == '__main__':
    main()
