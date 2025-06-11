# data_prep.py
import numpy as np

# --- New Constants ---
SLOT_DURATION_MIN = 15
SLOTS_PER_HOUR = 60 // SLOT_DURATION_MIN  # THIS SHOULD BE 4 FOR 15-MIN SLOTS
SLOTS_PER_DAY = 24 * SLOTS_PER_HOUR      # 96
# ---------------------

def extract_time_params(df_day, devices, usage_threshold=0.05):
    """
    MODIFIED FOR SINGLE-DAY OPTIMIZATION:
    Analyzes a single day's data (df_day) to find which devices were active,
    their total runtime (LOT), and their original start time on that day.
    Historical comfort windows (alpha, beta) and median start (m_slot) are no longer calculated.
    """
    df_day = df_day.copy()
    # The 'slot' column is calculated in the calling function (generate_planning)
    if 'slot' not in df_day.columns:
         df_day['slot'] = df_day.index.hour * SLOTS_PER_HOUR + df_day.index.minute // SLOT_DURATION_MIN

    daily_tasks = {}
    # The data is sampled at 1-minute intervals.
    samp_period_seconds = 60

    for dev, params in devices.items():
        col = next((c for c in df_day.columns if dev in c), None)
        if not col:
            continue

        # Use the day's maximum usage to set a dynamic threshold
        max_power_on_day = df_day[col].max()
        if max_power_on_day == 0:
            continue  # Device was not used at all on this day.

        thresh = usage_threshold * max_power_on_day
        is_on = df_day[col] > thresh

        # Calculate total duration (Length Of Task) in seconds for the day
        lot_s = float(is_on.sum() * samp_period_seconds)

        if lot_s > 0:
            # Find the timestamp of the first "on" minute and get its corresponding slot
            first_on_index = is_on.idxmax()
            actual_start_slot = df_day.loc[first_on_index, 'slot']

            # The new 'params' for a device are its original params + LOT and actual start slot
            daily_tasks[dev] = {
                **params,
                'LOT': lot_s,
                'actual_start_slot': int(actual_start_slot),
            }

    return daily_tasks