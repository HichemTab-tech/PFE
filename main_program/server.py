# server.py
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Import constants directly from data_prep for consistency (NO LOCAL REDEFINITION)
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
    slot_duration_min: int = SLOT_DURATION_MIN
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

    # --- REVISED DEVICE POWER CALCULATION (P_for_solver) ---
    # This ensures that P_for_solver[d] * LOT_s[d] / 3600.0 exactly equals the actual total energy of device d
    # observed on the target day, maintaining energy conservation between default and optimized simulations.
    P_for_solver = {}
    for device_name, current_params in effective_devices.items():
        # 1. Calculate the actual total energy (kWh) consumed by this device on the target day
        actual_total_energy_kwh_on_target_day = sum(
            device_load_profiles[device_name][s] * (SLOT_DURATION_MIN / 60.0)
            for s in range(SLOTS_PER_DAY)
        )

        # 2. Get the median duration (LOT_s) for this device from the pre-processing params
        duration_s = params[device_name]['LOT']

        # 3. Calculate the 'effective' power (P) for the solver.
        # This P is such that (P * duration_s / 3600) == actual_total_energy_kwh
        if duration_s > 0:
            P_for_solver[device_name] = (actual_total_energy_kwh_on_target_day * 3600.0) / duration_s
        else:
            P_for_solver[device_name] = 0.0 # If device has no duration, its power is 0

        # Update the effective_devices dict's 'power' attribute for external consistency if needed
        current_params['power'] = P_for_solver[device_name]
    # --------------------------------------------------------

    # Default planning (basic m_slot) - This is for display purposes, not for 'default_consumption' calculation
    default_schedule_for_display = {d: int(round(p['m_slot'])) for d, p in params.items()}

    # Unpack parameters for solver and estimation functions
    α = {d: params_['alpha_slot'] for d, params_ in params.items()}
    β = {d: params_['beta_slot'] for d, params_ in params.items()}
    LOT_s = {d: params_['LOT'] for d, params_ in params.items()} # This LOT_s is the median duration from data_prep
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
        total_cost = 0.0
        for d, start_slot in n_.items():
            power_kw = P_for_solver[d] # Use P_for_solver for consistency
            duration_s = LOT_s[d]

            # Cost calculation logic remains the same (summing energy * price over duration)
            num_slots_for_LOT = int(np.ceil(duration_s / (SLOT_DURATION_MIN * 60.0)))
            device_energy_cost_d = 0.0
            for offset in range(num_slots_for_LOT):
                current_slot = (start_slot + offset) % SLOTS_PER_DAY
                seconds_in_current_slot = min(duration_s - offset * (SLOT_DURATION_MIN * 60.0), SLOT_DURATION_MIN * 60.0)
                energy_kwh_in_slot = power_kw * (seconds_in_current_slot / 3600.0)
                cost_in_slot = energy_kwh_in_slot * price_profile[current_slot]
                device_energy_cost_d += cost_in_slot

            total_cost += W[d] * device_energy_cost_d

        comfort = sum(L[d] * W[d] * (n_[d] - M[d]) ** 2 for d in n_)
        return total_cost + comfort

    # Select and run solver
    solver = SolverFactory(req.algorithm, fitness=fitness, params={
        'α': α, 'β': β, 'LOT': LOT_s, 'P': P_for_solver, 'W': W, 'L': L, 'M': M, 'valid_hours': valid_slots # Pass P_for_solver
    })
    optimized_schedule = solver.run(list(effective_devices.keys()), seed=42)

    # --- Consumption and Cost Calculation Functions (No major logic changes needed here) ---
    def calculate_actual_consumption_from_profiles(baseline_profile: pd.Series,
                                                   device_profiles: Dict[str, pd.Series]) -> Dict[int, float]:
        consumption = baseline_profile.reindex(range(SLOTS_PER_DAY), fill_value=0).to_dict()
        for dev_name, dev_profile in device_profiles.items():
            for slot, value in dev_profile.items():
                consumption[slot] += value
        return {s: round(kW, 2) for s, kW in consumption.items()}

    def calculate_actual_cost_from_profiles(baseline_profile: pd.Series, device_profiles: Dict[str, pd.Series],
                                            price_profile: pd.Series) -> float:
        cost = 0.0
        cost += sum(
            baseline_profile[s] * price_profile[s] * (SLOT_DURATION_MIN / 60.0)
            for s in range(SLOTS_PER_DAY)
        )
        for dev_name, dev_profile in device_profiles.items():
            for slot, value in dev_profile.items():
                cost += value * price_profile[slot] * (SLOT_DURATION_MIN / 60.0)
        return cost

    def simulate_consumption(schedule: Dict[str, int], baseline_profile: pd.Series,
                             lot_seconds_map: Dict[str, float], effective_power_map: Dict[str, float]) -> Dict[
        int, float]:
        consumption = baseline_profile.reindex(range(SLOTS_PER_DAY), fill_value=0).to_dict()
        slot_duration_seconds = SLOT_DURATION_MIN * 60.0

        for d, start_slot in schedule.items():
            power_kw = effective_power_map[d] # This is P_for_solver for the optimized path
            duration_s = lot_seconds_map[d] # This is LOT_s for the optimized path

            remaining_duration_s = duration_s
            current_slot_offset = 0

            while remaining_duration_s > 0:
                current_slot_index = (start_slot + current_slot_offset) % SLOTS_PER_DAY
                seconds_in_this_slot = min(remaining_duration_s, slot_duration_seconds)
                power_contribution_to_slot = power_kw * (seconds_in_this_slot / slot_duration_seconds)
                consumption[current_slot_index] += power_contribution_to_slot

                remaining_duration_s -= seconds_in_this_slot
                current_slot_offset += 1
        return {s: round(kW, 2) for s, kW in consumption.items()}

    def simulate_cost(schedule: Dict[str, int], baseline_profile: pd.Series, price_profile: pd.Series,
                      lot_seconds_map: Dict[str, float], effective_power_map: Dict[str, float]) -> float:
        total_scheduled_device_cost = 0.0
        slot_duration_seconds = SLOT_DURATION_MIN * 60.0

        for d, start_slot in schedule.items():
            power_kw = effective_power_map[d] # This is P_for_solver for the optimized path
            duration_s = lot_seconds_map[d] # This is LOT_s for the optimized path

            remaining_duration_s = duration_s
            current_slot_offset = 0

            while remaining_duration_s > 0:
                current_slot_index = (start_slot + current_slot_offset) % SLOTS_PER_DAY
                seconds_in_this_slot = min(remaining_duration_s, slot_duration_seconds)

                energy_kwh_in_slot = power_kw * (seconds_in_this_slot / 3600.0)
                cost_in_slot = energy_kwh_in_slot * price_profile[current_slot_index]
                total_scheduled_device_cost += cost_in_slot

                remaining_duration_s -= seconds_in_this_slot
                current_slot_offset += 1

        baseline_cost = sum(
            baseline_profile[s] * price_profile[s] * (SLOT_DURATION_MIN / 60.0)
            for s in range(SLOTS_PER_DAY)
        )
        return total_scheduled_device_cost + baseline_cost

    # --- FINAL CALCULATIONS FOR RESPONSE ---
    # default_consumption_real: The raw, actual historical consumption for the day
    default_consumption_real = df_day.groupby('slot')['use [kW]'].mean().reindex(range(SLOTS_PER_DAY),
                                                                                 fill_value=0).to_dict()

    # default_consumption: This is designed to be IDENTICAL to default_consumption_real for baseline comparison.
    default_consumption = calculate_actual_consumption_from_profiles(baseline_load_profile, device_load_profiles)

    # default_cost: This is the actual historical cost for the day.
    default_cost = calculate_actual_cost_from_profiles(baseline_load_profile, device_load_profiles, price_profile)

    # optimized_consumption: This is the simulated consumption based on the optimizer's schedule.
    # It now uses P_for_solver, which ensures total energy conservation.
    optimized_consumption = simulate_consumption(optimized_schedule, baseline_load_profile, LOT_s, P_for_solver)

    # optimized_cost: This is the simulated cost based on the optimizer's schedule.
    # It now uses P_for_solver, which ensures total energy conservation.
    optimized_cost = simulate_cost(optimized_schedule, baseline_load_profile, price_profile, LOT_s, P_for_solver)

    # Build response
    devices_info = {d: DeviceParams(w=v['w'], lambda_=v['lambda']) for d, v in effective_devices.items()}

    return PlanningResponse(
        devices=devices_info,
        default_planning=default_schedule_for_display,
        optimized_planning=optimized_schedule,
        default_cost=default_cost,
        optimized_cost=optimized_cost,
        default_consumption_real=default_consumption_real,
        default_consumption=default_consumption,
        optimized_consumption=optimized_consumption,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)