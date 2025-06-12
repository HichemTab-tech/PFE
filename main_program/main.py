import datetime
from typing import Dict, Any, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np  # Needed for np.ceil

# Corrected imports: assuming all .py files are in the same directory,
# and 'solvers' is a subdirectory.
from data_prep import SLOTS_PER_DAY, SLOTS_PER_HOUR, SLOT_DURATION_MIN
from optimization_logic import generate_planning  # Corrected import

date = "2016-01-05"  # Target date for the simulation


# --- Plotting Function (Original, slightly refactored) ---
def plot_total_consumption_results(results: Dict[str, Any]):
    slot_duration_min = results["slot_duration_min"]
    default_consumption_real = results["default_consumption_real"]
    default_consumption = results["default_consumption"]
    optimized_consumption = results["optimized_consumption"]
    baseline_load = results["baseline_load"]
    price_profile = pd.Series(results["price_profile"])

    x_labels = []
    for i in range(SLOTS_PER_DAY):
        hour = i // SLOTS_PER_HOUR
        minute = (i % SLOTS_PER_HOUR) * slot_duration_min
        x_labels.append(f"{hour:02d}:{minute:02d}")

    # Create a figure with a secondary y-axis for the price
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- Simplified Price Trace ---
    # Add the price profile as a simple line, NOT a shaded area
    fig.add_trace(
        go.Scatter(
            x=price_profile.index,
            y=price_profile.values,
            name='Price (DA/kWh)',
            line=dict(color='rgba(255, 165, 0, 0.5)', width=2, dash='dash')  # Changed to a dashed line
        ),
        secondary_y=True,
    )
    # --- END OF SIMPLIFIED TRACE ---

    # Add the consumption traces to the primary y-axis
    fig.add_trace(go.Scatter(
        x=list(baseline_load.keys()), y=list(baseline_load.values()),
        mode='lines', name='Uncontrollable Baseline Load',
        line=dict(color='rgba(128, 128, 128, 0.7)', width=2),  # Removed dash for clarity
        fill='tozeroy', fillcolor='rgba(211, 211, 211, 0.3)'
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=list(default_consumption_real.keys()), y=list(default_consumption_real.values()),
        mode='lines', name='Actual Historical Consumption', line=dict(color='darkgray', width=3)
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=list(default_consumption.keys()), y=list(default_consumption.values()),
        mode='lines', name='Default Simulated Consumption',
        line=dict(color='blue', width=2, dash='dot')
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=list(optimized_consumption.keys()), y=list(optimized_consumption.values()),
        mode='lines', name='Optimized Consumption', line=dict(color='red', width=2)
    ), secondary_y=False)

    picLimit = results.get("picLimit")
    if picLimit is not None:
        fig.add_hline(y=picLimit, line_dash="dash", line_color="firebrick",
                      annotation_text=f"Peak Limit ({picLimit} kW)",
                      annotation_position="bottom right")

    # Update layout for better aesthetics
    fig.update_layout(
        title_text=f"Total Energy Consumption and Price Profile for {datetime.date.fromisoformat(results['target_date_str'])}",
        xaxis_title="Time Slot",
        # =================================================================
        # CRITICAL PERFORMANCE FIX: Change the hover mode
        hovermode="closest",
        # =================================================================
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.2)'),
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, SLOTS_PER_DAY, SLOTS_PER_HOUR * 2)),
            ticktext=[x_labels[i] for i in range(0, SLOTS_PER_DAY, SLOTS_PER_HOUR * 2)],
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)'
        ),
        yaxis=dict(
            title_text="Power Consumption (kW)",
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)'
        ),
        yaxis2=dict(
            title_text="Price (DA/kWh)",
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            range=[0, price_profile.max() * 3]  # Gave it even more room
        )
    )
    fig.show()


# --- New Plotting Function: Fitness Evolution ---
def plot_fitness_evolution(fitness_history: List[float], algorithm_name: str):
    fig = go.Figure(data=go.Scatter(y=fitness_history, mode='lines', line=dict(color='green', width=2)))
    fig.update_layout(
        title_text=f"Fitness Evolution for {algorithm_name.upper()} Algorithm",
        xaxis_title="Iteration/Generation",
        yaxis_title="Fitness Value (Cost)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=40),
        yaxis=dict(type='log')  # Fitness can vary greatly, log scale can help
    )
    fig.show()


# --- New Plotting Function: Individual Device Schedules (Modified) ---
def plot_individual_device_schedules(
        device_parameters: Dict[str, Any],
        default_individual_consumption: Dict[str, Dict[int, float]],  # Pass the actual consumption profiles
        optimized_individual_consumption: Dict[str, Dict[int, float]]  # Pass the actual consumption profiles
):
    # MODIFIED: Plot only active devices for the day
    active_devices = {k: v for k, v in device_parameters.items() if v.get('LOT', 0) > 0}
    num_devices = len(active_devices)
    if num_devices == 0:
        print("No active devices to plot for this day.")
        return

    rows = int(np.ceil(num_devices / 2))  # Arrange in 2 columns

    fig = make_subplots(rows=rows, cols=2,
                        subplot_titles=list(active_devices.keys()),
                        vertical_spacing=0.08,
                        horizontal_spacing=0.05)

    x_labels = []  # HH:MM labels for axis ticks and hover
    for i in range(SLOTS_PER_DAY):
        hour = i // SLOTS_PER_HOUR
        minute = (i % SLOTS_PER_HOUR) * SLOT_DURATION_MIN
        x_labels.append(f"{hour:02d}:{minute:02d}")

    x_slot_indices = list(range(SLOTS_PER_DAY))

    # MODIFIED: Update dummy traces for legend; remove historical ones
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             marker=dict(symbol='star', size=10, color='purple'),
                             line=dict(color='purple', width=0),
                             name='Original Start Time', showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color="blue", width=2, dash='dot', shape='hv'),
                             name='Original Consumption', showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color="red", width=2, dash='solid', shape='hv'),
                             name='Optimized Consumption', showlegend=True), row=1, col=1)

    row_idx, col_idx = 1, 1
    for i, (dev_name, params) in enumerate(active_devices.items()):
        # MODIFIED: Use new parameters; no alpha, beta, or m_slot
        actual_start_slot = params['actual_start_slot']
        dev_power = params['power_for_solver']

        max_y_val = max(dev_power * 1.2, 0.1)

        # REMOVED: Shading for historical operating window (alpha to beta) is no longer relevant.

        # MODIFIED: Add a marker for the original start time on the target day.
        fig.add_trace(go.Scatter(
            x=[actual_start_slot], y=[max_y_val * 0.9], mode='markers',
            marker=dict(symbol='star', size=10, color='purple'),
            name='Original Start Time',
            customdata=[x_labels[actual_start_slot]],
            hovertemplate=f"Original Start: {dev_name}<br>Slot: %{{x}}<br>Time: %{{customdata}}<extra></extra>",
            showlegend=False
        ), row=row_idx, col=col_idx)

        # Plot Original Daily Consumption (using stepped line)
        default_profile_data = default_individual_consumption.get(dev_name, {s: 0.0 for s in x_slot_indices})
        default_y = [default_profile_data.get(s, 0.0) for s in x_slot_indices]

        fig.add_trace(go.Scatter(
            x=x_slot_indices, y=default_y, mode='lines',
            line=dict(color="blue", width=2, dash='dot', shape='hv'),
            name='Original Consumption',
            customdata=x_labels,
            hovertemplate=f"Original ON: {dev_name}<br>Slot: %{{x}}<br>Time: %{{customdata}}<br>Power: %{{y:.2f}} kW<extra></extra>",
            showlegend=False
        ), row=row_idx, col=col_idx)

        # Plot Optimized Simulated Consumption (using stepped line)
        optimized_profile_data = optimized_individual_consumption.get(dev_name, {s: 0.0 for s in x_slot_indices})
        optimized_y = [optimized_profile_data.get(s, 0.0) for s in x_slot_indices]

        fig.add_trace(go.Scatter(
            x=x_slot_indices, y=optimized_y, mode='lines',
            line=dict(color="red", width=2, dash='solid', shape='hv'),
            name='Optimized Consumption',
            customdata=x_labels,
            hovertemplate=f"Optimized ON: {dev_name}<br>Slot: %{{x}}<br>Time: %{{customdata}}<br>Power: %{{y:.2f}} kW<extra></extra>",
            showlegend=False
        ), row=row_idx, col=col_idx)

        fig.update_xaxes(
            title_text="Time Slot",
            tickmode='array',
            tickvals=list(range(0, SLOTS_PER_DAY, SLOTS_PER_HOUR * 2)),
            ticktext=[x_labels[k] for k in range(0, SLOTS_PER_DAY, SLOTS_PER_HOUR * 2)],
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)',
            row=row_idx, col=col_idx
        )
        fig.update_yaxes(title_text="Power (kW)", range=[0, max_y_val], row=row_idx, col=col_idx)

        col_idx += 1
        if col_idx > 2:
            col_idx = 1
            row_idx += 1

    fig.update_layout(
        title_text="Individual Device Schedules: Original vs. Optimized",
        height=400 * rows,
        showlegend=True,
        hovermode="closest",
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(255,255,255,0.7)'),
    )
    fig.show()


def main():
    try:
        # Pass max_iter and new picLimit to generate_planning
        planning_results = generate_planning(
            date,
            algorithm='csa',
            max_iter=200, # Increased for better convergence with larger search space
            picLimit=1.2
        )
        planning_results['target_date_str'] = date  # Add target date to results for plot title

        print("\n--- Planning Results ---")
        print(f"Original Cost: {planning_results['default_cost']:.2f} DA")
        print(f"Optimized Cost: {planning_results['optimized_cost']:.2f} DA")
        print(f"Cost Savings: {planning_results['default_cost'] - planning_results['optimized_cost']:.2f} DA")
        print(
            f"Percentage Savings: {((planning_results['default_cost'] - planning_results['optimized_cost']) / planning_results['default_cost']) * 100:.2f}%")

        print("\n--- Total Energy Consumed (kW-hours) ---")
        slot_duration_hours = SLOT_DURATION_MIN / 60.0
        total_energy_real = sum(planning_results['default_consumption_real'].values()) * slot_duration_hours
        total_energy_default_sim = sum(planning_results['default_consumption'].values()) * slot_duration_hours
        total_energy_optimized_sim = sum(planning_results['optimized_consumption'].values()) * slot_duration_hours

        print(f"Actual Total Energy on Day: {total_energy_real:.2f} kWh")
        print(f"Simulated Original Total Energy: {total_energy_default_sim:.2f} kWh")
        print(f"Simulated Optimized Total Energy: {total_energy_optimized_sim:.2f} kWh")

        print("\n--- Schedules ---")
        print("Original Schedule (Device -> Start Slot):")
        for d, s in planning_results['default_planning'].items():
            print(f"  {d}: Slot {s} ({s // SLOTS_PER_HOUR:02d}:{(s % SLOTS_PER_HOUR) * SLOT_DURATION_MIN:02d})")
        print("Optimized Schedule (Device -> Start Slot):")
        for d, s in planning_results['optimized_planning'].items():
            print(f"  {d}: Slot {s} ({s // SLOTS_PER_HOUR:02d}:{(s % SLOTS_PER_HOUR) * SLOT_DURATION_MIN:02d})")

        # Plot the results (total consumption)
        plot_total_consumption_results(planning_results)

        # # Plot fitness evolution
        # plot_fitness_evolution(planning_results['fitness_history'],
        #                        'csa')  # Assuming 'csa' algorithm is used for this run
        #
        # # Plot individual device schedules
        # plot_individual_device_schedules(
        #     planning_results['device_parameters'],
        #     planning_results['default_individual_consumption'],  # Pass individual consumption data
        #     planning_results['optimized_individual_consumption']  # Pass individual consumption data
        # )

    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")


main()