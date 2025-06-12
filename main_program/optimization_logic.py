import datetime  # Added for timedelta in extract_time_params
from typing import Callable
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from pydantic import BaseModel

from data_import import load_csv_data, load_devices
# Import constants directly from data_prep for consistency
from data_prep import SLOT_DURATION_MIN, SLOTS_PER_HOUR, SLOTS_PER_DAY, extract_time_params
# REMOVED: from solvers import SolverFactory
# ADDED: Direct solver imports
from solvers.csa_solver import CsaSolver
from solvers.ga_solver import GaSolver

# --- Global Data Loading (Load once when module is imported) ---
_df = load_csv_data("HomeC.csv")
_devices = load_devices("devices_with_w.json", "HomeC.csv")


# -------------------------------------------------------------


def build_price_profile():
    """Build a price profile for the day with hourly variations."""
    prices_hourly = (
            [1.2050] * 6  # 00:00–05:59 → Night
            + [2.1645] * 11  # 06:00–16:59 → Full
            + [8.1147] * 4  # 17:00–20:59 → Peak
            + [2.1645] * 1  # 21:00–21:59 → Full
            + [1.2050] * 2  # 22:00–23:59 → Night
    )
    prices_slotted = []
    for p_h in prices_hourly:
        prices_slotted.extend([p_h] * SLOTS_PER_HOUR)
    return pd.Series(prices_slotted, index=range(SLOTS_PER_DAY))


def build_load_profile(df_day: pd.DataFrame, devices: Dict[str, Any]) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """Build baseline and device-specific load profiles from daily data."""
    df_day = df_day.copy()  # Operate on a copy to avoid SettingWithCopyWarning
    df_day['slot'] = df_day.index.hour * SLOTS_PER_HOUR + df_day.index.minute // SLOT_DURATION_MIN

    # Get total usage and generation
    # =================================================================
    total_usage = df_day["use [kW]"]
    generation = df_day.get("gen [kW]", 0) # Use .get for safety if column is missing
    # =================================================================

    smart_device_cols = [c for c in df_day.columns if any(d in c for d in devices)]
    smart_total = df_day[smart_device_cols].sum(axis=1)

    # MODIFIED: Correct baseline calculation
    # Baseline is what the grid sees (usage) PLUS what was generated, MINUS smart devices.
    # This represents the "other" non-schedulable consumption.
    # =================================================================
    baseline = total_usage + generation - smart_total
    # =================================================================

    # --- FIX 1: Ensure baseline is never negative ---
    # This is still a good safety measure in case of other data errors.
    baseline = np.maximum(0, baseline)
    # -----------------------------------------------

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

        # 2. Get the duration (LOT_s) for this device from the daily task parameters
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


def create_fitness_function(P_for_solver: Dict[str, float], LOT_s: Dict[str, float],
                            W: Dict[str, float], L: Dict[str, float], M: Dict[str, float],
                            price_profile: pd.Series) -> Callable:
    """Create and return a fitness function for the optimization solver."""

    def fitness(n_: Dict[str, int]):  # Explicitly type n_ as Dict[str, int]
        total_cost = 0.0
        for d, start_slot in n_.items():
            power_kw = P_for_solver[d]
            duration_s = LOT_s[d]

            # Cost calculation logic
            num_slots_for_LOT = int(np.ceil(duration_s / (SLOT_DURATION_MIN * 60.0)))
            device_energy_cost_d = 0.0
            for offset in range(num_slots_for_LOT):
                current_slot = (start_slot + offset) % SLOTS_PER_DAY
                seconds_in_current_slot = min(duration_s - offset * (SLOT_DURATION_MIN * 60.0),
                                              SLOT_DURATION_MIN * 60.0)
                energy_kwh_in_slot = power_kw * (seconds_in_current_slot / 3600.0)
                cost_in_slot = energy_kwh_in_slot * price_profile[current_slot]
                device_energy_cost_d += cost_in_slot

            total_cost += W[d] * device_energy_cost_d

        # MODIFIED: Comfort penalty is disabled by setting L=0 for all devices. This term will be zero.
        comfort = sum(L[d] * W[d] * (n_[d] - M[d]) ** 2 for d in n_)
        return total_cost + comfort

    return fitness


def calculate_actual_consumption_from_profiles(baseline_profile: pd.Series,
                                               device_profiles: Dict[str, pd.Series]) -> Dict[int, float]:
    """Calculate actual consumption from baseline and device profiles."""
    consumption = baseline_profile.reindex(range(SLOTS_PER_DAY), fill_value=0).to_dict()
    for dev_name, dev_profile in device_profiles.items():
        for slot, value in dev_profile.items():
            consumption[slot] += value
    # No need to round here, let plot_total_consumption_results handle it if necessary
    return {s: kW for s, kW in consumption.items()}


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


def simulate_consumption(schedule: Dict[str, int], baseline_profile: pd.Series,
                         lot_seconds_map: Dict[str, float],
                         effective_power_map: Dict[str, float]) -> Dict[int, float]:
    """Simulate total consumption based on a given schedule."""
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
            power_contribution_to_slot = power_kw * (seconds_in_this_slot / slot_duration_seconds)
            consumption[current_slot_index] += power_contribution_to_slot

            remaining_duration_s -= seconds_in_this_slot
            current_slot_offset += 1
    # No need to round here, let plot_total_consumption_results handle it if necessary
    return {s: kW for s, kW in consumption.items()}


def simulate_cost(schedule: Dict[str, int], baseline_profile: pd.Series,
                  price_profile: pd.Series,
                  lot_seconds_map: Dict[str, float],
                  effective_power_map: Dict[str, float]) -> float:
    """Simulate total cost based on a given schedule."""
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


def simulate_individual_device_consumption(schedule: Dict[str, int],
                                           lot_seconds_map: Dict[str, float],
                                           effective_power_map: Dict[str, float]) -> Dict[str, Dict[int, float]]:
    """
    Simulate consumption for each device based on a given schedule, showing the
    effective power (P_for_solver) when the device is active.
    """
    device_consumptions = {dev: {s: 0.0 for s in range(SLOTS_PER_DAY)} for dev in schedule.keys()}
    slot_duration_seconds = SLOT_DURATION_MIN * 60.0

    for d, start_slot in schedule.items():
        power_kw = effective_power_map[d]
        duration_s = lot_seconds_map[d]

        remaining_duration_s = duration_s
        current_slot_offset = 0

        while remaining_duration_s > 0:
            current_slot_index = (start_slot + current_slot_offset) % SLOTS_PER_DAY

            device_consumptions[d][current_slot_index] = power_kw

            seconds_in_this_slot = min(remaining_duration_s, slot_duration_seconds)
            remaining_duration_s -= seconds_in_this_slot
            current_slot_offset += 1

    # No need to round here, let plotting function handle it if necessary
    return {dev: {s: kW for s, kW in profile.items()} for dev, profile in device_consumptions.items()}


# This function is no longer needed as valid slots are the full day.
# def get_valid_slots(α: Dict[str, int], β: Dict[str, int], devices: Dict[str, Any]) -> Dict[str, List[int]]:
#    ...


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
    fitness_history: List[float]  # New field
    device_parameters: Dict[str, Any]  # New field for plotting device ranges


def generate_planning(date: str = "2016-01-05", start_hour: int = 0, algorithm: str = 'csa', max_iter: int = 100,
                      picLimit: Optional[float] = None) -> Dict[str, Any]:
    print(f"Calculating planning data for {date} with {algorithm.upper()}...")

    # Access global dataframes
    global _df, _devices
    df = _df
    devices = _devices

    target = pd.to_datetime(date).date()
    df_day = df[df.index.date == target].copy()
    if df_day.empty:
        raise ValueError(f"No data for date {target}")

    df_day['slot'] = df_day.index.hour * SLOTS_PER_HOUR + df_day.index.minute // SLOT_DURATION_MIN

    # MODIFIED: Identify active devices and their tasks for the specific day.
    # The 'start_hour' parameter is no longer used for filtering.
    daily_tasks = extract_time_params(df_day, devices)
    devices_to_schedule = list(daily_tasks.keys())
    effective_devices = {dev: devices[dev] for dev in devices_to_schedule}

    # Use 'daily_tasks' as the new 'params' object
    params = daily_tasks

    price_profile = build_price_profile()
    baseline_load_profile, device_load_profiles = build_load_profile(df_day, devices)

    # ADD THIS DEBUG BLOCK
    # =================================================================
    if picLimit is not None:
        print("\n--- DEBUG: Baseline Load Analysis ---")
        peak_baseline_kw = baseline_load_profile.max()
        peak_baseline_slot = baseline_load_profile.idxmax()
        print(f"The maximum baseline load is {peak_baseline_kw:.2f} kW at slot {peak_baseline_slot}.")
        if peak_baseline_kw > picLimit:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(
                f"!!! CRITICAL WARNING: The baseline itself ({peak_baseline_kw:.2f} kW) is higher than your picLimit ({picLimit} kW).")
            print(f"!!! The optimizer CANNOT fix this peak.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # =================================================================

    P_for_solver = calculate_device_power_for_solver(device_load_profiles, effective_devices, params)

    # MODIFIED: The "default" schedule is the actual schedule from the target day.
    default_schedule_for_display = {d: p['actual_start_slot'] for d, p in params.items()}

    # MODIFIED: Set up parameters for a cost-only optimization over the full 24h day.
    LOT_s = {d: p['LOT'] for d, p in params.items()}
    W = {d: effective_devices[d]['w'] for d in effective_devices}
    L = {d: 0.0 for d in effective_devices}  # KEY CHANGE: Disables comfort penalty
    M = {d: p['actual_start_slot'] for d, p in params.items()}  # Used as anchor for immovable devices (W=0)

    # MODIFIED: The optimization window is the entire day for all active devices.
    # Historical comfort windows (alpha, beta) and forward-only constraints are removed.
    valid_slots = {d: list(range(SLOTS_PER_DAY)) for d in effective_devices}
    constrained_valid_slots = valid_slots  # Pass this to the repair function closure

    fitness = create_fitness_function(P_for_solver, LOT_s, W, L, M, price_profile)

    # --- NEW: Create a repair function for picLimit constraint ---
    repair_function = lambda s: s  # Default identity function
    if picLimit is not None:
        print(f"Enforcing peak consumption limit of {picLimit} kW.")

        def adjust_schedule_for_pic_limit(schedule: Dict[str, int]) -> Dict[str, int]:
            immovable_devices = {d: s for d, s in schedule.items() if W.get(d, 1) == 0}
            movable_schedule = {d: s for d, s in schedule.items() if W.get(d, 1) != 0}

            total_consumption = baseline_load_profile.copy().to_numpy()
            for device, start_slot in immovable_devices.items():
                power_kw = P_for_solver[device]
                duration_s = LOT_s[device]
                num_slots = int(np.ceil(duration_s / (SLOT_DURATION_MIN * 60.0)))
                if num_slots == 0: continue
                for offset in range(num_slots):
                    slot_idx = (start_slot + offset) % SLOTS_PER_DAY
                    total_consumption[slot_idx] += power_kw

            repaired_movable_schedule = {}
            devices_to_place = sorted(movable_schedule.keys(), key=lambda d: movable_schedule[d])

            for device in devices_to_place:
                power_kw = P_for_solver[device]
                duration_s = LOT_s[device]
                num_slots = int(np.ceil(duration_s / (SLOT_DURATION_MIN * 60.0)))

                if num_slots == 0:
                    repaired_movable_schedule[device] = movable_schedule[device]
                    continue

                # The repair function now searches within the full-day valid slots
                device_valid_slots = constrained_valid_slots[device]
                proposed_start_slot = movable_schedule[device]

                try:
                    start_index = device_valid_slots.index(proposed_start_slot)
                    slots_to_try = device_valid_slots[start_index:] + device_valid_slots[:start_index]
                except (ValueError, IndexError):
                    slots_to_try = device_valid_slots
                    if not slots_to_try:
                        repaired_movable_schedule[device] = proposed_start_slot
                        continue

                found_placement = False
                for try_slot in slots_to_try:
                    is_violation = False
                    for offset in range(num_slots):
                        slot_idx = (try_slot + offset) % SLOTS_PER_DAY
                        if total_consumption[slot_idx] + power_kw > picLimit:
                            is_violation = True
                            break
                    if not is_violation:
                        repaired_movable_schedule[device] = try_slot
                        for offset in range(num_slots):
                            slot_idx = (try_slot + offset) % SLOTS_PER_DAY
                            total_consumption[slot_idx] += power_kw
                        found_placement = True
                        break

                if not found_placement:
                    last_resort_slot = slots_to_try[-1] if slots_to_try else proposed_start_slot
                    repaired_movable_schedule[device] = last_resort_slot
                    for offset in range(num_slots):
                        slot_idx = (last_resort_slot + offset) % SLOTS_PER_DAY
                        total_consumption[slot_idx] += power_kw

            final_schedule = immovable_devices.copy()
            final_schedule.update(repaired_movable_schedule)
            return final_schedule

        repair_function = adjust_schedule_for_pic_limit

    # MODIFIED: Solver parameters are simplified for the new logic
    solver_params = {
        'LOT': LOT_s, 'P': P_for_solver, 'W': W, 'L': L, 'M': M,
        'valid_hours': valid_slots
    }

    if algorithm == 'csa':
        solver = CsaSolver(fitness=fitness, params=solver_params, repair_function=repair_function)
    elif algorithm == 'ga':
        solver = GaSolver(fitness=fitness, params=solver_params, repair_function=repair_function)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    optimized_schedule, fitness_history = solver.run(devices_to_schedule, seed=42, max_iter=max_iter)

    default_consumption_real = df_day.groupby('slot')['use [kW]'].mean().reindex(range(SLOTS_PER_DAY),
                                                                                 fill_value=0).to_dict()
    # "Default Simulated" is now based on the actual schedule, so it's the same as "Real".
    default_consumption = calculate_actual_consumption_from_profiles(baseline_load_profile, device_load_profiles)
    default_cost = calculate_actual_cost_from_profiles(baseline_load_profile, device_load_profiles, price_profile)
    optimized_consumption = simulate_consumption(optimized_schedule, baseline_load_profile, LOT_s, P_for_solver)
    optimized_cost = simulate_cost(optimized_schedule, baseline_load_profile, price_profile, LOT_s, P_for_solver)
    default_individual_consumption = simulate_individual_device_consumption(
        default_schedule_for_display, LOT_s, P_for_solver)
    optimized_individual_consumption = simulate_individual_device_consumption(
        optimized_schedule, LOT_s, P_for_solver)

    # MODIFIED: Device parameters for plotting no longer include historical data.
    device_plot_params = {d: {'actual_start_slot': p['actual_start_slot'],
                              'LOT': LOT_s[d], 'power_for_solver': P_for_solver[d],
                              'w': W[d], 'lambda': L[d]}  # L[d] is 0
                          for d, p in params.items()}

    print("Optimized Consumption Preview (first 5 slots):", list(optimized_consumption.items())[:5])

    return {
        "slot_duration_min": SLOT_DURATION_MIN,
        "devices_info": {d: DeviceParams(w=v['w'], lambda_=v['lambda']) for d, v in effective_devices.items()},
        "default_planning": default_schedule_for_display,
        "optimized_planning": optimized_schedule,
        "default_cost": default_cost,
        "optimized_cost": optimized_cost,
        "default_consumption_real": default_consumption_real,
        "default_consumption": default_consumption,
        "optimized_consumption": optimized_consumption,
        "price_profile": price_profile.to_dict(),
        "fitness_history": fitness_history,
        "device_parameters": device_plot_params,
        "default_individual_consumption": default_individual_consumption,
        "optimized_individual_consumption": optimized_individual_consumption,
        "picLimit": picLimit,
        "baseline_load": baseline_load_profile.to_dict(),  # <--- ADD THIS LINE
    }