import datetime  # Added for timedelta in extract_time_params
from typing import Callable
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from pydantic import BaseModel

from data_import import load_csv_data, load_devices
# Import constants directly from data_prep for consistency
from data_prep import SLOT_DURATION_MIN, SLOTS_PER_HOUR, SLOTS_PER_DAY, extract_time_params
from solvers import SolverFactory  # Corrected import path

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

    total = df_day["use [kW]"]
    smart_device_cols = [c for c in df_day.columns if any(d in c for d in devices)]
    smart_total = df_day[smart_device_cols].sum(axis=1)

    baseline = total - smart_total

    # --- FIX 1: Ensure baseline is never negative ---
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


def get_valid_slots(α: Dict[str, int], β: Dict[str, int], devices: Dict[str, Any]) -> Dict[str, List[int]]:
    """Get valid slots for each device based on alpha and beta parameters."""
    valid_slots = {}
    for d in devices:
        a, b = α[d], β[d]
        if a <= b:
            valid_slots[d] = list(range(a, b + 1))
        else:  # Window wraps around midnight
            valid_slots[d] = list(range(a, SLOTS_PER_DAY)) + list(range(0, b + 1))
    return valid_slots


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


def generate_planning(date: str = "2016-01-05", start_hour: int = 0, algorithm: str = 'csa', max_iter: int = 100) -> \
        Dict[str, Any]:
    print(f"Calculating planning data for {date} with {algorithm.upper()}...")

    # Access global dataframes
    global _df, _devices
    df = _df
    devices = _devices

    target = pd.to_datetime(date).date()
    df_day = df[df.index.date == target].copy()  # Ensure df_day is a copy to avoid SettingWithCopyWarning
    if df_day.empty:
        raise ValueError(f"No data for date {target}")

    # --- FIX 3: Ensure df_day has 'slot' column immediately ---
    df_day['slot'] = df_day.index.hour * SLOTS_PER_HOUR + df_day.index.minute // SLOT_DURATION_MIN
    # --------------------------------------------------------

    effective_devices = devices.copy()

    start_slot_filter = start_hour * SLOTS_PER_HOUR
    # This filter was for `df_filtered`, which was used in `extract_time_params`
    # If `extract_time_params` is supposed to use a filtered view, then `df_filtered` is still needed.
    # If `extract_time_params` just uses the 'window_slots' derived from `wake_h`/`sleep_h` (as it does now),
    # then `df_filtered` might not be strictly necessary for `extract_time_params` itself,
    # but the `df_day['slot']` assignment is definitely needed.
    df_filtered = df_day[
        df_day.index.hour * SLOTS_PER_HOUR + df_day.index.minute // SLOT_DURATION_MIN >= start_slot_filter].copy()

    params = extract_time_params(df_filtered, effective_devices)
    price_profile = build_price_profile()
    baseline_load_profile, device_load_profiles = build_load_profile(df_day,
                                                                     effective_devices)  # Use df_day for full historical context

    # Calculate effective power values for each device to use in the solver
    P_for_solver = calculate_device_power_for_solver(device_load_profiles, effective_devices, params)

    # Default planning (basic m_slot) - This is for display purposes, not for 'default_consumption' calculation
    default_schedule_for_display = {d: int(round(p['m_slot'])) for d, p in params.items()}

    # Unpack parameters for solver and estimation functions
    α = {d: params_['alpha_slot'] for d, params_ in params.items()}
    β = {d: params_['beta_slot'] for d, params_ in params.items()}
    LOT_s = {d: params_['LOT'] for d, params_ in params.items()}  # This LOT_s is the median duration from data_prep
    W = {d: effective_devices[d]['w'] for d in effective_devices}
    L = {d: effective_devices[d]['lambda'] for d in effective_devices}
    M = {d: params_['m_slot'] for d, params_ in params.items()}

    # Get valid slots for each device
    valid_slots = get_valid_slots(α, β, effective_devices)

    # Create fitness function for the solver
    fitness = create_fitness_function(P_for_solver, LOT_s, W, L, M, price_profile)

    # Select and run solver
    solver = SolverFactory(algorithm, fitness=fitness, params={
        'α': α, 'β': β, 'LOT': LOT_s, 'P': P_for_solver, 'W': W, 'L': L, 'M': M, 'valid_hours': valid_slots
    })
    optimized_schedule, fitness_history = solver.run(list(effective_devices.keys()), seed=42, max_iter=max_iter)

    # --- FINAL CALCULATIONS FOR RESPONSE ---
    # default_consumption_real: The raw, actual historical total consumption for the day
    default_consumption_real = df_day.groupby('slot')['use [kW]'].mean().reindex(range(SLOTS_PER_DAY),
                                                                                 fill_value=0).to_dict()

    # --- FIX 2: Default simulated consumption IS the actual historical consumption ---
    default_consumption = default_consumption_real  # Use the actual historical as the benchmark
    # --------------------------------------------------------------------------------

    # default_cost: This is the actual historical cost for the day.
    default_cost = calculate_actual_cost_from_profiles(baseline_load_profile, device_load_profiles, price_profile)

    # Optimized_consumption: This is the simulated total consumption based on the optimizer's schedule.
    # It uses P_for_solver and LOT_s, which ensure total energy conservation.
    optimized_consumption = simulate_consumption(optimized_schedule, baseline_load_profile, LOT_s, P_for_solver)

    # Optimized_cost: This is the simulated cost based on the optimizer's schedule.
    # It uses P_for_solver and LOT_s.
    optimized_cost = simulate_cost(optimized_schedule, baseline_load_profile, price_profile, LOT_s, P_for_solver)

    # Calculate individual device consumption profiles for plotting
    default_individual_consumption = simulate_individual_device_consumption(
        default_schedule_for_display, LOT_s, P_for_solver)
    optimized_individual_consumption = simulate_individual_device_consumption(
        optimized_schedule, LOT_s, P_for_solver)

    # Prepare device_parameters for plotting
    device_plot_params = {d: {'alpha_slot': p['alpha_slot'], 'beta_slot': p['beta_slot'], 'm_slot': p['m_slot'],
                              'LOT': LOT_s[d], 'power_for_solver': P_for_solver[d],
                              'w': W[d], 'lambda': L[d]}  # Include w and lambda for completeness
                          for d, p in params.items()}

    print("Optimized Consumption Preview (first 5 slots):", list(optimized_consumption.items())[:5])

    return {
        "slot_duration_min": SLOT_DURATION_MIN,
        "devices_info": {d: DeviceParams(w=v['w'], lambda_=v['lambda']) for d, v in effective_devices.items()},
        # Keeping the full device info for potential display
        "default_planning": default_schedule_for_display,
        "optimized_planning": optimized_schedule,
        "default_cost": default_cost,
        "optimized_cost": optimized_cost,
        "default_consumption_real": default_consumption_real,
        "default_consumption": default_consumption,  # Now should be equal to default_consumption_real
        "optimized_consumption": optimized_consumption,
        "price_profile": price_profile.to_dict(),  # Convert to dict for easier passing
        "fitness_history": fitness_history,  # Add fitness history
        "device_parameters": device_plot_params,  # Add detailed device parameters for plotting
        "default_individual_consumption": default_individual_consumption,
        "optimized_individual_consumption": optimized_individual_consumption,
    }