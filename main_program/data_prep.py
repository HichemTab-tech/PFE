import numpy as np
import pandas as pd

def extract_time_params(df, devices, usage_threshold=0.05):
    """
    Adds to each device:
      - alpha, beta (sec)
      - LOT        (sec)
      - max_delay  (sec)
      - m          (historical median start hour)
    """
    wake_h, sleep_h = 5, 2
    if wake_h <= sleep_h:
        window = list(range(wake_h, sleep_h + 1))
    else:
        window = list(range(wake_h, 24)) + list(range(0, sleep_h + 1))

    df = df.copy()
    df['date'] = df.index.date
    hours = df.index.hour

    results = {}
    for dev, params in devices.items():
        # column & active mask
        col = next(c for c in df.columns if dev in c)
        if wake_h <= sleep_h:
            mask = (hours >= wake_h) & (hours <= sleep_h)
        else:
            mask = (hours >= wake_h) | (hours <= sleep_h)
        usage = df.loc[mask, [col, 'date']]

        # dynamic threshold
        thresh = usage_threshold * usage[col].max()
        samp   = (usage.index[1] - usage.index[0]).total_seconds() if len(usage)>1 else 0

        # daily totals → LOT
        daily = (
            usage
            .assign(on=(usage[col] > thresh).astype(int))
            .groupby('date')['on']
            .sum()
            .multiply(samp)
        )
        lot_s = float(daily.median()) if not daily.empty else 0

        # historical median start hour m_d
        hourly_sum = usage.groupby(usage.index.hour)[col].sum()
        if hourly_sum.sum() > 0:
            m_d = (hourly_sum.index.to_numpy() * hourly_sum.values).sum() / hourly_sum.sum()
        else:
            m_d = np.mean(window)

        # α / β
        hourly = usage.groupby(usage.index.hour)[col].mean().reindex(window, fill_value=0)
        cum    = hourly.cumsum() / (hourly.sum() if hourly.sum()>0 else 1)
        rel_a  = int(np.searchsorted(cum.values, usage_threshold))
        rel_b  = len(cum) - int(np.searchsorted(cum[::-1].values, usage_threshold)) - 1
        alpha_s = window[rel_a] * 3600
        beta_s  = (window[rel_b] + 1) * 3600

        results[dev] = {
            **params,
            'alpha':    alpha_s,
            'beta':     beta_s,
            'LOT':      lot_s,
            'max_delay':lot_s,
            'm':        m_d,
        }

    return results