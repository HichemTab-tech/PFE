import numpy as np
import pandas as pd  # Ensure pandas is imported

# --- New Constants ---
SLOT_DURATION_MIN = 15
SLOTS_PER_HOUR = 60 // SLOT_DURATION_MIN  # 2
SLOTS_PER_DAY = 24 * SLOTS_PER_HOUR  # 48


# ---------------------

def extract_time_params(df, devices, usage_threshold=0.05):
    # Original wake_h, sleep_h are in hours
    wake_h, sleep_h = 5, 2
    # Convert them to the corresponding 30-minute slot indices
    wake_slot = wake_h * SLOTS_PER_HOUR
    # sleep_h=2 means up to 2:59 AM. So it includes slots corresponding to hour 2.
    # The last slot of hour 2 is slot 2*2 + (60//30 - 1) = 4 + 1 = 5.
    sleep_slot_inclusive = sleep_h * SLOTS_PER_HOUR + (SLOTS_PER_HOUR - 1)

    # Determine window in terms of 30-min slots
    if wake_slot <= sleep_slot_inclusive:
        # e.g., 10 (5am) to 5 (2am next day)
        # This part of the logic handles cases where the window doesn't wrap around
        window_slots = list(range(wake_slot, sleep_slot_inclusive + 1))
    else:
        # Window wraps around midnight (e.g., 5am to 2am next day)
        window_slots = list(range(wake_slot, SLOTS_PER_DAY)) + list(range(0, sleep_slot_inclusive + 1))

    df = df.copy()
    df['date'] = df.index.date
    # New 'slot' column: maps datetime to 30-min slot index (0-47)
    df['slot'] = df.index.hour * SLOTS_PER_HOUR + df.index.minute // SLOT_DURATION_MIN

    results = {}
    for dev, params in devices.items():
        col = next(c for c in df.columns if dev in c)

        # Apply mask based on the 'slot' column
        if wake_slot <= sleep_slot_inclusive:
            mask = (df['slot'] >= wake_slot) & (df['slot'] <= sleep_slot_inclusive)
        else:
            mask = (df['slot'] >= wake_slot) | (df['slot'] <= sleep_slot_inclusive)

        usage = df.loc[mask, [col, 'date', 'slot']]  # Include 'slot' for grouping

        # Dynamic threshold
        thresh = usage_threshold * usage[col].max()
        samp = (usage.index[1] - usage.index[0]).total_seconds() if len(usage) > 1 else 0

        # daily totals → LOT (still in seconds)
        daily = (
            usage
            .assign(on=(usage[col] > thresh).astype(int))
            .groupby('date')['on']
            .sum()
            .multiply(samp)
        )
        lot_s = float(daily.median()) if not daily.empty else 0

        # historical median start slot m_slot
        # Group by the new 'slot' column
        slot_sum = usage.groupby(usage['slot'])[col].sum()
        if slot_sum.sum() > 0:
            m_d_slot = (slot_sum.index.to_numpy() * slot_sum.values).sum() / slot_sum.sum()
        else:
            m_d_slot = np.mean(window_slots)  # Default if no usage

        # α / β in slots
        # Group by 'slot' for cumulative usage
        slot_avg = usage.groupby(usage['slot'])[col].mean().reindex(window_slots, fill_value=0)
        cum = slot_avg.cumsum() / (slot_avg.sum() if slot_avg.sum() > 0 else 1)

        # Use searchsorted to find slot indices for alpha and beta
        rel_a_idx = int(np.searchsorted(cum.values, usage_threshold))
        rel_b_idx = len(cum) - int(np.searchsorted(cum[::-1].values, usage_threshold)) - 1

        alpha_slot = window_slots[rel_a_idx]
        beta_slot = window_slots[rel_b_idx]

        results[dev] = {
            **params,
            'alpha_slot': alpha_slot,
            'beta_slot': beta_slot,
            'LOT': lot_s,  # LOT remains in seconds
            'max_delay': lot_s,
            'm_slot': m_d_slot,
        }

    return results