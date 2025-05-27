# server.py
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Import constants directly from data_prep for consistency
from data_import import load_csv_data, load_devices
from data_prep import extract_time_params, SLOT_DURATION_MIN, SLOTS_PER_HOUR, SLOTS_PER_DAY
from solvers import SolverFactory

app = FastAPI()

# --- Allow CORS for frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Ensure constants are also defined here for clarity, though imported from data_prep ---
# Re-defining constants in server.py is less ideal; importing them is better
# SLOT_DURATION_MIN, SLOTS_PER_HOUR, SLOTS_PER_DAY are now imported directly.

# --- Load data once on server startup ---
df = load_csv_data("HomeC.csv")
devices = load_devices("devices_with_w.json", "HomeC.csv")


def build_price_profile():
    prices_hourly = [0.10] * 7 + [0.20] * 4 + [0.15] * 6 + [0.22] * 2 + [0.10] * 5
    prices_slotted = []
    for p_h in prices_hourly:
        prices_slotted.extend([p_h] * SLOTS_PER_HOUR)
    return pd.Series(prices_slotted, index=range(SLOTS_PER_DAY))


def build_load_profile(df_day: pd.DataFrame, devices: Dict[str, Any]) -> (pd.Series, Dict[str, pd.Series]):
    df_day['slot'] = df_day.index.hour * SLOTS_PER_HOUR + df_day.index.minute // SLOT_DURATION_MIN

    total = df_day["use [kW]"]
    smart_device_cols = [c for c in df_day.columns if any(d in c for d in devices)]
    smart_total = df_day[smart_device_cols].sum(axis=1)
    baseline = total - smart_total

    baseline_profile_kW = baseline.groupby(df_day['slot']).mean()
    baseline_profile_kW = baseline_profile_kW.reindex(range(SLOTS_PER_DAY), fill_value=0)

    device_profiles_kW = {}
    for dev_name in devices.keys():
        col = next((c for c in smart_device_cols if dev_name in c), None)
        if col:
            dev_profile = df_day[col].groupby(df_day['slot']).mean()
            device_profiles_kW[dev_name] = dev_profile.reindex(range(SLOTS_PER_DAY), fill_value=0)
        else:
            device_profiles_kW[dev_name] = pd.Series([0.0] * SLOTS_PER_DAY, index=range(SLOTS_PER_DAY))

    return baseline_profile_kW, device_profiles_kW


# --- Request/Response models (no changes needed) ---
class DeviceParams(BaseModel):
    w: float
    lambda_: float


class PlanningRequest(BaseModel):
    date: str
    custom_params: Optional[Dict[str, DeviceParams]] = None
    algorithm: Optional[str] = 'csa'
    start_hour: Optional[int] = 0


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

    effective_devices = devices.copy()

    start_slot_filter = req.start_hour * SLOTS_PER_HOUR
    df_filtered = df_day[
        df_day.index.hour * SLOTS_PER_HOUR + df_day.index.minute // SLOT_DURATION_MIN >= start_slot_filter]

    df_day['slot'] = df_day.index.hour * SLOTS_PER_HOUR + df_day.index.minute // SLOT_DURATION_MIN
    df_filtered['slot'] = df_filtered.index.hour * SLOTS_PER_HOUR + df_filtered.index.minute // SLOT_DURATION_MIN

    params = extract_time_params(df_filtered, effective_devices)
    price_profile = build_price_profile()
    baseline_load_profile, device_load_profiles = build_load_profile(df_day, effective_devices)

    # --- DEVICE POWER CALCULATION (On-Period Mean) for Fitness function (P) ---
    usage_threshold_power = 0.05
    for device_name, current_params in effective_devices.items():
        col = next((c for c in df_day.columns if device_name in c), None)
        if col is None:
            current_params['power'] = 1.0
        else:
            max_daily_power = df_day[col].max()
            active_readings = df_day[df_day[col] > usage_threshold_power * max_daily_power][col]
            if not active_readings.empty:
                current_params['power'] = active_readings.mean()
            else:
                current_params['power'] = current_params.get('power',
                                                             1.0)  # Fallback to historical max if no activity on target day

    # Default planning (basic m_slot) - This is for display purposes, not for 'default_consumption' calculation
    default_schedule_for_display = {d: int(round(p['m_slot'])) for d, p in params.items()}

    # Unpack parameters for solver and estimation functions
    α = {d: params_['alpha_slot'] for d, params_ in params.items()}
    β = {d: params_['beta_slot'] for d, params_ in params.items()}
    LOT_s = {d: params_['LOT'] for d, params_ in params.items()}
    P = {d: effective_devices[d]['power'] for d in effective_devices}
    W = {d: effective_devices[d]['w'] for d in effective_devices}
    L = {d: effective_devices[d]['lambda'] for d in effective_devices}
    M = {d: params_['m_slot'] for d, params_ in params.items()}

    valid_slots = {}
    for d in effective_devices:
        a, b = α[d], β[d]
        if a <= b:
            valid_slots[d] = list(range(a, b + 1))
        else:
            valid_slots[d] = list(range(a, SLOTS_PER_DAY)) + list(range(0, b + 1))

    # Fitness function (solver will receive slot indices)
    def fitness(n_):
        # This cost is for the solver's optimization, using simplified P and LOT_s
        cost = sum(W[d] * (P[d] * LOT_s[d] / 3600.0) * price_profile.loc[n_[d]] for d in n_)
        comfort = sum(L[d] * W[d] * (n_[d] - M[d]) ** 2 for d in n_)
        return cost + comfort

    # Select and run solver
    solver = SolverFactory(req.algorithm, fitness=fitness, params={
        'α': α, 'β': β, 'LOT': LOT_s, 'P': P, 'W': W, 'L': L, 'M': M, 'valid_hours': valid_slots
    })
    optimized_schedule = solver.run(list(effective_devices.keys()), seed=42)

    # --- NEW: Function to calculate consumption from actual historical profiles ---
    def calculate_actual_consumption_from_profiles(baseline_profile: pd.Series,
                                                   device_profiles: Dict[str, pd.Series]) -> Dict[int, float]:
        # Initialize with baseline consumption for all slots
        consumption = baseline_profile.reindex(range(SLOTS_PER_DAY), fill_value=0).to_dict()

        # Add each device's *actual historical profile* for the day
        for dev_name, dev_profile in device_profiles.items():
            for slot, value in dev_profile.items():
                consumption[slot] += value

        return {s: round(kW, 2) for s, kW in consumption.items()}

    # --- NEW: Function to calculate cost from actual historical profiles ---
    def calculate_actual_cost_from_profiles(baseline_profile: pd.Series, device_profiles: Dict[str, pd.Series],
                                            price_profile: pd.Series) -> float:
        cost = 0.0
        # Cost from baseline
        cost += sum(
            baseline_profile[s] * price_profile[s] * (SLOT_DURATION_MIN / 60.0)
            for s in range(SLOTS_PER_DAY)
        )
        # Cost from individual devices' actual historical profiles
        for dev_name, dev_profile in device_profiles.items():
            for slot, value in dev_profile.items():
                cost += value * price_profile[slot] * (SLOT_DURATION_MIN / 60.0)
        return cost

    # --- Function to simulate consumption based on a generated schedule ---
    def simulate_consumption(schedule: Dict[str, int], baseline_profile: pd.Series,
                             device_profiles: Dict[str, pd.Series], lot_seconds_map: Dict[str, float]) -> Dict[
        int, float]:
        hourly_consumption = baseline_profile.reindex(range(SLOTS_PER_DAY), fill_value=0).to_dict()

        for d, start_slot in schedule.items():
            dev_profile = device_profiles[d]
            # Use the LOT_s from params for the simulation duration
            num_slots_for_LOT = int(np.ceil(lot_seconds_map[d] / (SLOT_DURATION_MIN * 60.0)))

            for offset in range(num_slots_for_LOT):
                current_slot = (start_slot + offset) % SLOTS_PER_DAY
                hourly_consumption[current_slot] += dev_profile.get(current_slot, 0.0)

        return {s: round(kW, 2) for s, kW in hourly_consumption.items()}

    # --- Function to simulate cost based on a generated schedule ---
    def simulate_cost(schedule: Dict[str, int], baseline_profile: pd.Series, price_profile: pd.Series,
                      lot_seconds_map: Dict[str, float], single_device_power_map: Dict[str, float]) -> float:
        total_scheduled_device_cost = 0.0
        for d, start_slot in schedule.items():
            power_kw = single_device_power_map[d]
            duration_s = lot_seconds_map[d]

            total_scheduled_device_cost += power_kw * (duration_s / 3600.0) * price_profile.loc[start_slot]

        baseline_cost = sum(
            baseline_profile[s] * price_profile[s] * (SLOT_DURATION_MIN / 60.0)
            for s in range(SLOTS_PER_DAY)
        )
        return total_scheduled_device_cost + baseline_cost

    # --- FINAL CALCULATIONS FOR RESPONSE ---
    # default_consumption_real: The raw, actual historical consumption for the day
    default_consumption_real = df_day.groupby('slot')['use [kW]'].mean().reindex(range(SLOTS_PER_DAY),
                                                                                 fill_value=0).to_dict()

    # default_consumption: NOW CALCULATED TO BE IDENTICAL TO default_consumption_real
    default_consumption = calculate_actual_consumption_from_profiles(baseline_load_profile, device_load_profiles)

    # default_cost: NOW CALCULATED TO BE THE ACTUAL HISTORICAL COST for the day
    default_cost = calculate_actual_cost_from_profiles(baseline_load_profile, device_load_profiles, price_profile)

    # optimized_consumption: This is the simulated consumption based on the optimizer's schedule
    optimized_consumption = simulate_consumption(optimized_schedule, baseline_load_profile, device_load_profiles, LOT_s)

    # optimized_cost: This is the simulated cost based on the optimizer's schedule
    optimized_cost = simulate_cost(optimized_schedule, baseline_load_profile, price_profile, LOT_s, P)

    # Build response
    devices_info = {d: DeviceParams(w=v['w'], lambda_=v['lambda']) for d, v in effective_devices.items()}

    return PlanningResponse(
        devices=devices_info,
        default_planning=default_schedule_for_display,  # Still shows the historical median start times
        optimized_planning=optimized_schedule,
        default_cost=default_cost,
        optimized_cost=optimized_cost,
        default_consumption_real=default_consumption_real,
        default_consumption=default_consumption,  # This should now be a near-perfect match to default_consumption_real
        optimized_consumption=optimized_consumption,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)