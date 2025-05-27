import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
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


# --- MODIFIED: build_load_profile to return individual device profiles ---
def build_load_profile(df_day: pd.DataFrame, devices: Dict[str, Any]) -> (pd.Series, Dict[str, pd.Series]):
    """
    Builds the baseline load profile and individual device load profiles for the target day.

    Returns:
        tuple: (baseline_profile_kW, device_profiles_kW)
            baseline_profile_kW: pd.Series mapping slot (0-47) -> avg kW for non-smart load
            device_profiles_kW: Dict[device_name, pd.Series] mapping slot (0-47) -> avg kW for that device
    """
    # Ensure 'slot' column is available for grouping
    # This line calculates the 30-minute slot index (0 to 47) for each row (each minute)
    # based on its hour and minute.
    # Example: 05:00-05:29 -> slot 10 (5*2), 05:30-05:59 -> slot 11 (5*2+1)
    df_day['slot'] = df_day.index.hour * SLOTS_PER_HOUR + df_day.index.minute // SLOT_DURATION_MIN

    # Get the total measured power consumption for each minute of the target day.
    # This is the "real" total load column from your CSV.
    total = df_day["use [kW]"]

    # Identify smart device columns
    # This creates a list of column names from df_day that correspond to your defined smart devices.
    # Example: ["Dishwasher [kW]", "Microwave [kW]", ...]
    smart_device_cols = [c for c in df_day.columns if any(d in c for d in devices)]

    # Calculate the sum of power consumption of *all* identified smart devices for each minute.
    # This is the real, raw combined consumption of the smart devices on the target day.
    smart_total = df_day[smart_device_cols].sum(axis=1)

    # Calculate the baseline load for each minute by subtracting the sum of smart device loads
    # from the total household load. This assumes that 'use [kW]' includes all loads,
    # and the smart device columns are components of that total.
    baseline = total - smart_total

    # Group the calculated baseline load by 30-minute slot and take the mean (average) power for each slot.
    # This gives you an average baseline consumption profile for the day.
    baseline_profile_kW = baseline.groupby(df_day['slot']).mean()

    # Reindex the baseline profile to ensure it has entries for all 48 slots (0-47).
    # If a slot had no data (e.g., due to filtering or missing data), it will be filled with 0.
    baseline_profile_kW = baseline_profile_kW.reindex(range(SLOTS_PER_DAY), fill_value=0)

    # Initialize an empty dictionary to store the individual power profiles for each smart device.
    device_profiles_kW = {}

    # Loop through each smart device name provided in your `devices` configuration.
    for dev_name in devices.keys():
        # Find the actual column name in the DataFrame for the current device.
        col = next((c for c in smart_device_cols if dev_name in c), None)
        if col:
            # If the device's column is found:
            # Group that device's raw consumption by 30-minute slot and take the mean.
            # This gives you the average power profile for that specific device for the day.
            dev_profile = df_day[col].groupby(df_day['slot']).mean()
            # Reindex to ensure all 48 slots, filling missing with 0 (meaning device was off or no data).
            device_profiles_kW[dev_name] = dev_profile.reindex(range(SLOTS_PER_DAY), fill_value=0)
        else:
            # Fallback if a device from your JSON isn't found in the CSV (shouldn't happen often).
            device_profiles_kW[dev_name] = pd.Series([0.0] * SLOTS_PER_DAY, index=range(SLOTS_PER_DAY))

    # Return the calculated baseline profile and the dictionary of individual device profiles.
    return baseline_profile_kW, device_profiles_kW


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

    # MODIFIED: Get both baseline and individual device profiles
    baseline_load_profile, device_load_profiles = build_load_profile(df_day, effective_devices)

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

    # --- MODIFIED: Cost estimations using device_load_profiles for accuracy ---
    def estimate_cost(schedule, baseline_profile, device_profiles, price_profile):
        device_cost = 0.0
        # 1) Cost for scheduled devices
        for d, start_slot in schedule.items():
            dev_profile = device_profiles[d]

            # Determine how many slots the device's typical LOT (in seconds) spans.
            # We use ceil because even a partial slot means the device is 'on' in that slot.
            num_slots_for_LOT = int(np.ceil(LOT_s[d] / (SLOT_DURATION_MIN * 60.0)))

            # Sum up energy for each slot the device is active based on its profile
            for offset in range(num_slots_for_LOT):
                current_slot = (start_slot + offset) % SLOTS_PER_DAY

                # Get the average power for this device in this historical slot
                power_in_slot_kw = dev_profile.get(current_slot, 0.0)

                # Energy = power (kW) * duration of slot (hours)
                energy_kwh_in_slot = power_in_slot_kw * (SLOT_DURATION_MIN / 60.0)
                device_cost += energy_kwh_in_slot * price_profile[current_slot]

        # 2) Baseline household load cost (now slot-by-slot)
        # baseline_profile is already reindexed to all 48 slots with fill_value=0
        baseline_cost = sum(
            baseline_profile[s] * price_profile[s] * (SLOT_DURATION_MIN / 60.0)
            for s in range(SLOTS_PER_DAY)
        )
        return device_cost + baseline_cost

    # --- MODIFIED: Consumption estimations using device_load_profiles for accuracy ---
    def estimate_consumption(schedule, baseline_profile, device_profiles):
        # Initialize with baseline consumption for all slots
        # Ensure it's a mutable dict for additions
        hourly_consumption = baseline_profile.reindex(range(SLOTS_PER_DAY), fill_value=0).to_dict()

        for d, start_slot in schedule.items():
            dev_profile = device_profiles[d]
            num_slots_for_LOT = int(np.ceil(LOT_s[d] / (SLOT_DURATION_MIN * 60.0)))

            # Spread the device’s average power from its profile across its scheduled slots
            for offset in range(num_slots_for_LOT):
                current_slot = (start_slot + offset) % SLOTS_PER_DAY

                # Add the average power from its historical profile for this slot
                # If the device wasn't active at this particular slot in its history, its profile entry will be 0.
                hourly_consumption[current_slot] += dev_profile.get(current_slot, 0.0)

        return {s: kW for s, kW in hourly_consumption.items()}

    # New function: estimate_real_consumption
    def estimate_real_consumption():
        # Group by 30-min slot
        slot_consumption = df_day.groupby('slot')['use [kW]'].mean()
        # Reindex to ensure all 48 slots are present, fill missing with 0
        slot_consumption = slot_consumption.reindex(range(SLOTS_PER_DAY), fill_value=0)
        return {slot: kW for slot, kW in slot_consumption.items()}

    # Pass the new profiles to the estimation functions
    default_cost = estimate_cost(default_schedule, baseline_load_profile, device_load_profiles, price_profile)
    optimized_cost = estimate_cost(optimized_schedule, baseline_load_profile, device_load_profiles, price_profile)

    default_consumption_real = estimate_real_consumption()
    default_consumption = estimate_consumption(default_schedule, baseline_load_profile, device_load_profiles)
    optimized_consumption = estimate_consumption(optimized_schedule, baseline_load_profile, device_load_profiles)

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
