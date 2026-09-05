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


def prepare(df):
    df = df.copy()
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    df['atr14'] = atr(df, 14)
    df['ema200_prev6'] = df['ema200'].shift(6)
    return df


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {'open','high','low','close'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    return df


def backtest_v2(
    df,
    initial_equity=2000.0,
    risk_pct=0.005,
    atr_mult=1.25,
    reward_r=2.5,
    time_stop_bars=12,
    fee_rate=0.001,
    slippage_rate=0.0002,
):
    """BTC Pullback 4H V2.

    Changes from V1:
    - Stronger trend filter: EMA20 > EMA50 > EMA200 and EMA200 rising.
    - Pullback must touch EMA20 in prior 3 bars but all closes stay above EMA50.
    - Trigger must reclaim EMA20 and close above previous high.
    - Slightly tighter ATR stop and wider reward target.
    - Same conservative same-bar stop-first rule.
    """
    df = prepare(df).reset_index(drop=True)
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    trades = []
    i = 210

    while i < len(df) - 1:
        row = df.iloc[i]
        prev = df.iloc[i-1]
        recent = df.iloc[i-3:i]

        trend_ok = (
            row['close'] > row['ema200'] and
            row['ema20'] > row['ema50'] > row['ema200'] and
            row['ema200'] > row['ema200_prev6']
        )
        pullback_ok = (
            (recent['low'] <= recent['ema20']).any() and
            (recent['close'] >= recent['ema50']).all()
        )
        trigger_ok = row['close'] > prev['high'] and row['close'] > row['ema20']

        if not (trend_ok and pullback_ok and trigger_ok and np.isfinite(row['atr14'])):
            i += 1
            continue

        entry_i = i + 1
        entry = df.iloc[entry_i]['open'] * (1 + slippage_rate)
        stop_distance = atr_mult * row['atr14']
        if stop_distance <= 0:
            i += 1
            continue

        stop = entry - stop_distance
        target = entry + reward_r * stop_distance
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
        last_i = min(entry_i + time_stop_bars - 1, len(df)-1)

        for j in range(entry_i, last_i + 1):
            bar = df.iloc[j]
            hit_stop = bar['low'] <= stop
            hit_target = bar['high'] >= target
            if hit_stop and hit_target:
                exit_i, exit_price, outcome = j, stop * (1-slippage_rate), 'SL_same_bar'
                break
            if hit_stop:
                exit_i, exit_price, outcome = j, stop * (1-slippage_rate), 'SL'
                break
            if hit_target:
                exit_i, exit_price, outcome = j, target * (1-slippage_rate), 'TP'
                break

        if exit_i is None:
            exit_i = last_i
            exit_price = df.iloc[exit_i]['close'] * (1-slippage_rate)
            outcome = 'TIME'

        exit_notional = qty * exit_price
        exit_fee = exit_notional * fee_rate
        gross_pnl = qty * (exit_price - entry)
        net_pnl = gross_pnl - entry_fee - exit_fee
        r_multiple = net_pnl / risk_dollars if risk_dollars > 0 else 0.0

        equity_before = equity
        equity += net_pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak-equity)/peak if peak else 0)

        trades.append({
            'signal_index': i,
            'entry_index': entry_i,
            'exit_index': exit_i,
            'entry': entry,
            'stop': stop,
            'target': target,
            'qty': qty,
            'notional': notional,
            'outcome': outcome,
            'net_pnl': net_pnl,
            'r_multiple': r_multiple,
            'equity_before': equity_before,
            'equity_after': equity,
        })
        i = exit_i + 1

    t = pd.DataFrame(trades)
    if t.empty:
        return t, {'initial_equity':initial_equity,'final_equity':equity,'return_pct':0.0,'trades':0,'win_rate_pct':0.0,'profit_factor':np.nan,'avg_r':np.nan,'max_drawdown_pct':0.0,'max_consecutive_losses':0,'total_fees_est':0.0}

    wins = t[t.net_pnl > 0]
    losses = t[t.net_pnl <= 0]
    gp = wins.net_pnl.sum()
    gl = -losses.net_pnl.sum()
    pf = gp/gl if gl > 0 else np.inf
    cur = max_losses = 0
    for is_loss in (t.net_pnl <= 0):
        cur = cur + 1 if is_loss else 0
        max_losses = max(max_losses, cur)

    stats = {
        'initial_equity': initial_equity,
        'final_equity': equity,
        'return_pct': (equity/initial_equity - 1)*100,
        'trades': len(t),
        'win_rate_pct': (t.net_pnl > 0).mean()*100,
        'profit_factor': pf,
        'avg_r': t.r_multiple.mean(),
        'max_drawdown_pct': max_dd*100,
        'max_consecutive_losses': max_losses,
        'total_fees_est': float((t['notional']*fee_rate + t['qty']*t['entry']*fee_rate).sum()),
    }
    return t, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--equity', type=float, default=2000.0)
    args = p.parse_args()
    df = load_csv(args.csv)
    trades, stats = backtest_v2(df, initial_equity=args.equity)
    print('\n=== BTC Pullback 4H V2 ===')
    for k,v in stats.items(): print(f'{k}: {v}')
    trades.to_csv('btc_pullback_v2_trades.csv', index=False)


if __name__ == '__main__':
    main()
