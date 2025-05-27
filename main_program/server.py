import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

from data_import import load_csv_data, load_devices
from data_prep import extract_time_params  # Will use the updated version
from solvers import SolverFactory

# Also apply the fix for csa_solver.py's run method as discussed previously.

app = FastAPI()

# --- Allow CORS for frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- New Constants ---
SLOT_DURATION_MIN = 15
SLOTS_PER_HOUR = 60 // SLOT_DURATION_MIN  # 2
SLOTS_PER_DAY = 24 * SLOTS_PER_HOUR  # 48
# ---------------------

# --- Load data once on server startup ---
df = load_csv_data("HomeC.csv")
devices = load_devices("devices_with_w.json", "HomeC.csv")  # Initial load with max power


def build_price_profile():
    # Define hourly prices as before
    prices_hourly = [0.10] * 7 + [0.20] * 4 + [0.15] * 6 + [0.22] * 2 + [0.10] * 5  # 24 elements
    # Duplicate for each 30-min slot within the hour
    prices_30min = []
    for p_h in prices_hourly:
        prices_30min.extend([p_h] * SLOTS_PER_HOUR)
    return pd.Series(prices_30min, index=range(SLOTS_PER_DAY))


def build_load_profile(df_day, devices):
    total = df_day["use [kW]"]
    smart = df_day[[c for c in df_day.columns if any(d in c for d in devices)]].sum(axis=1)
    baseline = total - smart
    # Group by the new 30-min slot index
    load_profile_slots = baseline.groupby(
        df_day.index.hour * SLOTS_PER_HOUR + df_day.index.minute // SLOT_DURATION_MIN).mean()
    return load_profile_slots


# --- Request/Response models ---
class DeviceParams(BaseModel):
    w: float
    lambda_: float

class PlanningRequest(BaseModel):
    date: str
    custom_params: Optional[Dict[str, DeviceParams]] = None
    algorithm: Optional[str] = 'csa'
    start_hour: Optional[int] = 0

# --- Request/Response models (no changes needed here, as the values are just numbers) ---

class PlanningResponse(BaseModel):
    devices: Dict[str, DeviceParams]
    default_planning: Dict[str, int]
    optimized_planning: Dict[str, int]
    default_cost: float
    optimized_cost: float
    default_consumption_real: Dict[int, float]
    default_consumption: Dict[int, float]
    optimized_consumption: Dict[int, float]

# --- API Endpoints ---

@app.post("/planning", response_model=PlanningResponse)
async def generate_planning(req: PlanningRequest):
    target = pd.to_datetime(req.date).date()
    df_day = df[df.index.date == target]
    if df_day.empty:
        raise HTTPException(status_code=404, detail=f"No data for date {target}")

    effective_devices = devices.copy()  # Use a copy so original global 'devices' isn't modified

    # Filter by start_hour. This assumes req.start_hour is a whole hour.
    # The `extract_time_params` will then work on this filtered data.
    df_filtered = df_day[df_day.index.hour >= req.start_hour]

    # Add 'slot' column for consistency with data_prep and later grouping
    df_day['slot'] = df_day.index.hour * SLOTS_PER_HOUR + df_day.index.minute // SLOT_DURATION_MIN
    df_filtered['slot'] = df_filtered.index.hour * SLOTS_PER_HOUR + df_filtered.index.minute // SLOT_DURATION_MIN

    params = extract_time_params(df_filtered, effective_devices)
    price_profile = build_price_profile()
    load_profile = build_load_profile(df_filtered, effective_devices)

    # --- UPDATED DEVICE POWER CALCULATION (On-Period Mean) ---
    # Define a threshold for what constitutes "active" usage
    usage_threshold_power = 0.05
    for device_name, current_params in effective_devices.items():
        col = next((c for c in df_day.columns if device_name in c), None)
        if col is None:
            # Fallback if device column not found, use a default
            current_params['power'] = 1.0
        else:
            # Calculate the power only when the device is actively drawing power
            max_daily_power = df_day[col].max()  # Max power observed for THIS device on THIS day

            # Filter for readings above a threshold based on the day's max power
            active_readings = df_day[df_day[col] > usage_threshold_power * max_daily_power][col]

            if not active_readings.empty:
                current_params['power'] = active_readings.mean()  # Average power during active periods
            else:
                # If no active readings (e.g., device not used today),
                # fall back to the max power observed over all history (loaded by load_devices)
                # or a sensible default.
                current_params['power'] = current_params.get('power', 1.0)
                # -----------------------------------------------------------

    # Default planning (basic m_slot)
    default_schedule = {d: int(round(p['m_slot'])) for d, p in params.items()}

    # unpack (now with _slot suffix for start times, LOT is still seconds)
    α = {d: params_['alpha_slot'] for d, params_ in params.items()}
    β = {d: params_['beta_slot'] for d, params_ in params.items()}
    LOT_s = {d: params_['LOT'] for d, params_ in params.items()}  # LOT is still in seconds
    P = {d: effective_devices[d]['power'] for d in effective_devices}  # Use the updated 'power'
    W = {d: effective_devices[d]['w'] for d in effective_devices}
    L = {d: effective_devices[d]['lambda'] for d in effective_devices}
    M = {d: params_['m_slot'] for d, params_ in params.items()}

    # valid slots with wrap-around
    valid_slots = {}
    for d in effective_devices:  # Iterate over the devices actually being considered
        a, b = α[d], β[d]
        if a <= b:
            valid_slots[d] = list(range(a, b + 1))
        else:
            valid_slots[d] = list(range(a, SLOTS_PER_DAY)) + list(range(0, b + 1))

    # Fitness function (solver will receive slot indices)
    def fitness(n_):
        # Calculate cost for device d: Energy (kWh) * Price ($/kWh)
        # Energy = Power (kW) * Duration (hours)
        # Duration in hours = LOT (seconds) / 3600
        # The cost assumes the entire energy cost is incurred at the price of the *start slot*.
        # This is a common simplification for solvers; a more detailed cost is calculated later.
        cost = sum(W[d] * (P[d] * LOT_s[d] / 3600.0) * price_profile.loc[n_[d]] for d in n_)

        # Comfort penalty based on deviation from median slot
        comfort = sum(L[d] * W[d] * (n_[d] - M[d]) ** 2 for d in n_)
        return cost + comfort

    # Select and run solver
    solver = SolverFactory(req.algorithm, fitness=fitness, params={
        'α': α,
        'β': β,
        'LOT': LOT_s,
        'P': P,
        'W': W,
        'L': L,
        'M': M,
        'valid_hours': valid_slots  # Now contains slot indices
    })

    # IMPORTANT: The solver.run method expects a list of device names as its first argument
    optimized_schedule = solver.run(list(effective_devices.keys()), seed=42)

    # Cost estimations (now operating on 30-min slots)
    def estimate_cost(schedule):
        device_cost = 0.0
        # 1) Cost for scheduled devices
        for d, start_slot in schedule.items():  # h_slot is now a slot index
            power_kw = P[d]
            duration_s = LOT_s[d]

            # This is a simplification: assuming cost is entire duration * price of start slot
            # For a more accurate cost: sum(P[d] * (fraction of slot) * price_profile[current_slot] for each slot device is active)
            # However, for total cost, it's simpler: total_energy * average_price_during_run
            # Let's keep it simple for now, total energy * start slot price.
            # OR better, if we want detailed:
            current_time_in_s = 0  # seconds from start of device operation
            for s_offset in range(SLOTS_PER_DAY):  # Iterate through possible slots
                effective_slot = (start_slot + s_offset) % SLOTS_PER_DAY

                # Determine seconds device is ON in this slot
                slot_start_s_from_device_start = s_offset * (SLOT_DURATION_MIN * 60)
                slot_end_s_from_device_start = slot_start_s_from_device_start + (SLOT_DURATION_MIN * 60)

                # If this slot overlaps with device operation
                if slot_start_s_from_device_start < duration_s:
                    overlap_s = min(duration_s - slot_start_s_from_device_start, (SLOT_DURATION_MIN * 60))
                    energy_in_slot_kwh = power_kw * (overlap_s / 3600.0)
                    device_cost += energy_in_slot_kwh * price_profile[effective_slot]
                else:
                    break  # Device has finished operation

        # 2) Baseline household load cost (now slot-by-slot)
        # Ensure load_profile is indexed for all 48 slots and filled with 0 where no data
        full_load_profile = load_profile.reindex(range(SLOTS_PER_DAY), fill_value=0)
        baseline_cost = sum(
            full_load_profile[s] * price_profile[s]
            for s in range(SLOTS_PER_DAY)
        )
        return device_cost + baseline_cost

    # Consumption estimations (fractional slots)
    def estimate_consumption(schedule):
        # Initialize for all 30-min slots in a day
        hourly_consumption = {s: 0.0 for s in range(SLOTS_PER_DAY)}

        for d, start_slot in schedule.items():
            power_kw = P[d]
            duration_s = LOT_s[d]  # Duration in seconds
            remaining_s = duration_s
            current_slot = start_slot

            # Spread the device’s energy use across slots
            while remaining_s > 0:
                slot_duration_s = (SLOT_DURATION_MIN * 60)  # Full duration of one slot in seconds

                # How much of the current slot does the device consume?
                # It's the minimum of remaining_s and the full slot duration.
                duration_in_current_slot_s = min(remaining_s, slot_duration_s)

                # Add consumption for this fraction of the slot (converted to kWh)
                hourly_consumption[current_slot % SLOTS_PER_DAY] += power_kw * (duration_in_current_slot_s / 3600.0)

                remaining_s -= duration_in_current_slot_s
                current_slot += 1

        # Add baseline load to simulated consumption
        full_load_profile = load_profile.reindex(range(SLOTS_PER_DAY), fill_value=0)
        for s in range(SLOTS_PER_DAY):
            hourly_consumption[s] += full_load_profile[s]

        # Round to 2 decimals
        return {s: round(kW, 2) for s, kW in hourly_consumption.items()}

    # New function: estimate_real_consumption
    def estimate_real_consumption():
        # Group by 30-min slot
        slot_consumption = df_day.groupby('slot')['use [kW]'].mean()
        # Reindex to ensure all 48 slots are present, fill missing with 0
        slot_consumption = slot_consumption.reindex(range(SLOTS_PER_DAY), fill_value=0)
        return {slot: round(kW, 2) for slot, kW in slot_consumption.items()}

    default_cost = estimate_cost(default_schedule)
    optimized_cost = estimate_cost(optimized_schedule)

    default_consumption_real = estimate_real_consumption()  # This is the actual historical load
    default_consumption = estimate_consumption(default_schedule)
    optimized_consumption = estimate_consumption(optimized_schedule)

    # Build response
    devices_info = {d: DeviceParams(w=v['w'], lambda_=v['lambda']) for d, v in effective_devices.items()}

    return PlanningResponse(
        devices=devices_info,
        default_planning=default_schedule,  # These are now slot indices (0-47)
        optimized_planning=optimized_schedule,  # These are now slot indices (0-47)
        default_cost=default_cost,
        optimized_cost=optimized_cost,
        default_consumption_real=default_consumption_real,  # The actual historical consumption
        default_consumption=default_consumption,  # The simulated default consumption
        optimized_consumption=optimized_consumption,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
