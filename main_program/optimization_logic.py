from typing import Callable
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from data_import import load_csv_data, load_devices
# Import constants directly from data_prep for consistency
from data_prep import SLOT_DURATION_MIN, SLOTS_PER_HOUR, SLOTS_PER_DAY, extract_time_params
from main_program.solvers import SolverFactory

# --- Global Data Loading (Load once when module is imported) ---
_df = load_csv_data("HomeC.csv")
_devices = load_devices("devices_with_w.json", "HomeC.csv")


# -------------------------------------------------------------


def build_price_profile():
    """Build a price profile for the day with hourly variations."""
    prices_hourly = [0.10] * 7 + [0.20] * 4 + [0.15] * 6 + [0.22] * 2 + [0.10] * 5
    prices_slotted = []
    for p_h in prices_hourly:
        prices_slotted.extend([p_h] * SLOTS_PER_HOUR)
    return pd.Series(prices_slotted, index=range(SLOTS_PER_DAY))


def build_load_profile(df_day: pd.DataFrame, devices: Dict[str, Any]) -> (pd.Series, Dict[str, pd.Series]):
    """Build baseline and device-specific load profiles from daily data."""
    # df_day is expected to already have the 'slot' column now from generate_planning
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


def calculate_device_power_for_solver(device_profiles: Dict[str, pd.Series],
                                      effective_devices: Dict[str, Any],
                                      params: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate effective power values for each device to use in the solver.
    This ensures energy conservation between default and optimized simulations.
    """
    P_for_solver = {}
    for device_name, current_params in effective_devices.items():
        # 1. Calculate the actual total energy (kWh) consumed by this device on the target day
        actual_total_energy_kwh_on_target_day = sum(
            device_profiles[device_name][s] * (SLOT_DURATION_MIN / 60.0)
            for s in range(SLOTS_PER_DAY)
        )

        # 2. Get the median duration (LOT_s) for this device from the pre-processing params
        duration_s = params[device_name]['LOT']

        # 3. Calculate the 'effective' power (P) for the solver.
        # This P is such that (P * duration_s / 3600) == actual_total_energy_kwh
        if duration_s > 0:
            P_for_solver[device_name] = (actual_total_energy_kwh_on_target_day * 3600.0) / duration_s
        else:
            P_for_solver[device_name] = 0.0  # If the device has no duration, its power is 0

        # Update the effective_devices dict's 'power' attribute for external consistency if needed
        current_params['power'] = P_for_solver[device_name]

    return P_for_solver


# simulate_consumption function (moved to be accessible by fitness function)
def simulate_consumption(schedule: Dict[str, int], baseline_profile: pd.Series,
                         lot_seconds_map: Dict[str, float],
                         effective_power_map: Dict[str, float]) -> Dict[int, float]:
    """Simulate consumption based on a given schedule."""
    consumption = baseline_profile.reindex(range(SLOTS_PER_DAY), fill_value=0).to_dict()
    slot_duration_seconds = SLOT_DURATION_MIN * 60.0

    for d, start_slot in schedule.items():
        power_kw = effective_power_map[d]
        duration_s = lot_seconds_map[d]

        remaining_duration_s = duration_s
        current_slot_offset = 0

        while remaining_duration_s > 0:
            current_slot_index = (start_slot + current_slot_offset) % SLOTS_PER_DAY
            seconds_in_this_slot = min(remaining_duration_s, slot_duration_seconds)
            # Calculate power contribution based on the fraction of the slot the device is ON
            power_contribution_to_slot = power_kw * (seconds_in_this_slot / slot_duration_seconds)
            consumption[current_slot_index] += power_contribution_to_slot

            remaining_duration_s -= seconds_in_this_slot
            current_slot_offset += 1
    return {s: round(kW, 2) for s, kW in consumption.items()}


# MODIFIED: Added max_peak_kW_threshold and gamma to create_fitness_function
def create_fitness_function(P_for_solver: Dict[str, float], LOT_s: Dict[str, float],
                            W: Dict[str, float], L: Dict[str, float], M: Dict[str, float],
                            price_profile: pd.Series,
                            baseline_profile: pd.Series,  # Baseline needed for peak calculation
                            max_peak_kW_threshold: float,  # Peak threshold
                            gamma: float  # Peak penalty coefficient (global)
                            ) -> Callable:
    """Create and return a fitness function for the optimization solver."""

    def fitness(n_):
        total_energy_cost = 0.0
        # Calculate energy cost
        for d, start_slot in n_.items():
            power_kw = P_for_solver[d]
            duration_s = LOT_s[d]

            num_slots_for_LOT = int(np.ceil(duration_s / (SLOT_DURATION_MIN * 60.0)))
            device_energy_cost_d = 0.0
            for offset in range(num_slots_for_LOT):
                current_slot = (start_slot + offset) % SLOTS_PER_DAY
                seconds_in_current_slot = min(duration_s - offset * (SLOT_DURATION_MIN * 60.0),
                                              SLOT_DURATION_MIN * 60.0)
                energy_kwh_in_slot = power_kw * (seconds_in_current_slot / 3600.0)
                cost_in_slot = energy_kwh_in_slot * price_profile[current_slot]
                device_energy_cost_d += cost_in_slot

            total_energy_cost += W[d] * device_energy_cost_d

        # Calculate comfort penalty
        comfort_penalty = sum(L[d] * W[d] * (n_[d] - M[d]) ** 2 for d in n_)

        # Calculate peak consumption penalty
        # Simulate the total consumption profile for the current schedule `n_`
        current_simulated_consumption = simulate_consumption(n_, baseline_profile, LOT_s, P_for_solver)

        observed_peak_kW = max(current_simulated_consumption.values()) if current_simulated_consumption else 0.0

        peak_penalty_value = 0.0
        if observed_peak_kW > max_peak_kW_threshold:
            # Quadratic penalty for exceeding the peak threshold
            peak_penalty_value = gamma * (observed_peak_kW - max_peak_kW_threshold) ** 2

        # --- DEBUG PRINTING (UNCOMMENT THIS BLOCK TO SEE FITNESS COMPONENTS) ---
        # print(f"\n--- Fitness Calc for Schedule ---")
        # print(f"  Schedule: {n_}")
        # print(f"  Energy Cost (weighted): {total_energy_cost:.4f}") # More precision for debug
        # print(f"  Comfort Penalty (weighted): {comfort_penalty:.4f}") # More precision for debug
        # print(f"  Observed Peak: {observed_peak_kW:.4f} kW (Threshold: {max_peak_kW_threshold:.4f} kW)")
        # print(f"  Peak Penalty Value: {peak_penalty_value:.4f} (Gamma: {gamma})") # More precision for debug
        # print(f"  Total Fitness: {total_energy_cost + comfort_penalty + peak_penalty_value:.4f}")
        # -------------------------------------------------------------------

        return total_energy_cost + comfort_penalty + peak_penalty_value

    return fitness


def calculate_actual_consumption_from_profiles(baseline_profile: pd.Series,
                                               device_profiles: Dict[str, pd.Series]) -> Dict[int, float]:
    """Calculate actual consumption from baseline and device profiles."""
    consumption = baseline_profile.reindex(range(SLOTS_PER_DAY), fill_value=0).to_dict()
    for dev_name, dev_profile in device_profiles.items():
        for slot, value in dev_profile.items():
            consumption[slot] += value
    return {s: round(kW, 2) for s, kW in consumption.items()}


def calculate_actual_cost_from_profiles(baseline_profile: pd.Series,
                                        device_profiles: Dict[str, pd.Series],
                                        price_profile: pd.Series) -> float:
    """Calculate actual cost from baseline and device profiles."""
    cost = 0.0
    cost += sum(
        baseline_profile[s] * price_profile[s] * (SLOT_DURATION_MIN / 60.0)
        for s in range(SLOTS_PER_DAY)
    )
    for dev_name, dev_profile in device_profiles.items():
        for slot, value in dev_profile.items():
            cost += value * price_profile[slot] * (SLOT_DURATION_MIN / 60.0)
    return cost


def simulate_cost(schedule: Dict[str, int], baseline_profile: pd.Series,
                  price_profile: pd.Series,
                  lot_seconds_map: Dict[str, float],
                  effective_power_map: Dict[str, float]) -> float:
    """Simulate cost based on a given schedule."""
    total_scheduled_device_cost = 0.0
    slot_duration_seconds = SLOT_DURATION_MIN * 60.0

    for d, start_slot in schedule.items():
        power_kw = effective_power_map[d]
        duration_s = lot_seconds_map[d]

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


def get_valid_slots(α: Dict[str, int], β: Dict[str, int], devices: Dict[str, Any]) -> Dict[str, list]:
    """Get valid slots for each device based on alpha and beta parameters."""
    valid_slots = {}
    for d in devices:
        a, b = α[d], β[d]
        if a <= b:
            valid_slots[d] = list(range(a, b + 1))
        else:
            valid_slots[d] = list(range(a, SLOTS_PER_DAY)) + list(range(0, b + 1))
    return valid_slots


# --- Request/Response models (no changes) ---
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
    default_peak_kW: float
    optimized_peak_kW: float


# MODIFIED: Added max_peak_kW_threshold and gamma to generate_planning
def generate_planning(date="2016-01-05", start_hour=0, algorithm='csa', max_iter=100,
                      max_peak_kW_threshold: float = 3.0,  # Default peak threshold
                      gamma: float = 100.0  # Default gamma coefficient for peak penalty
                      ):
    print(f"Calculating planning data for {date} with {algorithm.upper()}...")
    print(f"Peak threshold: {max_peak_kW_threshold} kW, Gamma (Peak Penalty Coeff): {gamma}")

    global _df, _devices
    df = _df
    devices = _devices

    target = pd.to_datetime(date).date()
    df_day = df[df.index.date == target]
    if df_day.empty:
        raise ValueError(f"No data for date {target}")

    # --- FIX for KeyError: 'slot' ---
    # Ensure 'slot' column is added to df_day here, before any groupby operations.
    df_day = df_day.copy()  # Operate on a copy to prevent SettingWithCopyWarning if df_day was a slice
    df_day['slot'] = df_day.index.hour * SLOTS_PER_HOUR + df_day.index.minute // SLOT_DURATION_MIN
    # --------------------------------

    effective_devices = devices.copy()

    params = extract_time_params(df_day, effective_devices)  # extract_time_params creates its own copy, this is fine.

    price_profile = build_price_profile()
    baseline_load_profile, device_load_profiles = build_load_profile(df_day, effective_devices)

    P_for_solver = calculate_device_power_for_solver(device_load_profiles, effective_devices, params)
    print("\n--- Device Effective Powers (P_for_solver) ---")
    for d, p in P_for_solver.items():
        print(f"  {d}: {p:.4f} kW")
    print("---------------------------------------------")

    default_schedule_for_display = {d: int(round(p['m_slot'])) for d, p in params.items()}

    α = {d: params_['alpha_slot'] for d, params_ in params.items()}
    β = {d: params_['beta_slot'] for d, params_ in params.items()}
    LOT_s = {d: params_['LOT'] for d, params_ in params.items()}
    W = {d: effective_devices[d]['w'] for d in effective_devices}
    L = {d: effective_devices[d]['lambda'] for d in effective_devices}
    M = {d: params_['m_slot'] for d, params_ in params.items()}

    valid_slots = get_valid_slots(α, β, effective_devices)

    # Pass max_peak_kW_threshold, gamma, and baseline_load_profile to create_fitness_function
    fitness = create_fitness_function(P_for_solver, LOT_s, W, L, M, price_profile,
                                      baseline_load_profile, max_peak_kW_threshold, gamma)

    solver = SolverFactory(algorithm, fitness=fitness, params={
        'α': α, 'β': β, 'LOT': LOT_s, 'P': P_for_solver, 'W': W, 'L': L, 'M': M, 'valid_hours': valid_slots
    })
    optimized_schedule = solver.run(list(effective_devices.keys()), seed=42, max_iter=max_iter)

    # --- FINAL CALCULATIONS FOR RESPONSE ---
    default_consumption_real = df_day.groupby('slot')['use [kW]'].mean().reindex(range(SLOTS_PER_DAY),
                                                                                 fill_value=0).to_dict()

    default_consumption = calculate_actual_consumption_from_profiles(baseline_load_profile, device_load_profiles)
    default_peak_kW = max(default_consumption.values()) if default_consumption else 0.0

    default_cost = calculate_actual_cost_from_profiles(baseline_load_profile, device_load_profiles, price_profile)

    optimized_consumption = simulate_consumption(optimized_schedule, baseline_load_profile, LOT_s, P_for_solver)
    optimized_peak_kW = max(optimized_consumption.values()) if optimized_consumption else 0.0

    optimized_cost = simulate_cost(optimized_schedule, baseline_load_profile, price_profile, LOT_s, P_for_solver)

    devices_info = {d: DeviceParams(w=v['w'], lambda_=v['lambda']) for d, v in effective_devices.items()}

    return {
        "slot_duration_min": SLOT_DURATION_MIN,
        "devices_info": devices_info,
        "default_planning": default_schedule_for_display,
        "optimized_planning": optimized_schedule,
        "default_cost": default_cost,
        "optimized_cost": optimized_cost,
        "default_consumption_real": default_consumption_real,
        "default_consumption": default_consumption,
        "optimized_consumption": optimized_consumption,
        "price_profile": price_profile.to_dict(),
        "default_peak_kW": default_peak_kW,
        "optimized_peak_kW": optimized_peak_kW
    }