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
    price_profile = pd.Series(results["price_profile"])  # Convert back to Series for easy indexing

    # Create x-axis labels (time in HH:MM format)
    x_labels = []
    for i in range(SLOTS_PER_DAY):
        hour = i // SLOTS_PER_HOUR
        minute = (i % SLOTS_PER_HOUR) * slot_duration_min
        x_labels.append(f"{hour:02d}:{minute:02d}")

    fig = go.Figure()

    # Add shaded regions for price profile
    # Determine the max Y value for shading height
    max_consumption_value = max(
        max(default_consumption_real.values()),
        max(default_consumption.values()),
        max(optimized_consumption.values())
    ) * 1.1  # Add a little buffer

    # Add picLimit line if it exists
    picLimit = results.get("picLimit")
    if picLimit is not None:
        fig.add_hline(y=picLimit, line_dash="dash", line_color="firebrick",
                      annotation_text=f"Peak Limit ({picLimit} kW)",
                      annotation_position="bottom right")
        max_consumption_value = max(max_consumption_value, picLimit * 1.1)


    price_colors = {
        1.2050: "rgba(144, 238, 144, 0.2)",  # Light green
        2.1645: "rgba(255, 255, 0, 0.2)",  # Yellow
        8.1147: "rgba(255, 165, 0, 0.2)",  # Orange
        #0.22: "rgba(255, 99, 71, 0.2)"  # Tomato/Red
    }

    current_price = None
    start_slot_for_price = 0
    # Iterate through slots to find price changes and add shaded regions
    for i in range(SLOTS_PER_DAY):
        price_at_slot = price_profile.get(i)
        if current_price is None:
            current_price = price_at_slot
            start_slot_for_price = i
        elif price_at_slot != current_price:
            fig.add_shape(
                type="rect",
                x0=start_slot_for_price,
                x1=i,  # The shape ends before the current slot, as current slot has new price
                y0=0,
                y1=max_consumption_value,
                fillcolor=price_colors.get(current_price, "rgba(0,0,0,0.1)"),
                layer="below",
                line_width=0,
            )
            # Add annotation for the completed price segment
            fig.add_annotation(
                x=(start_slot_for_price + i) / 2,  # Horizontal center of the segment
                y=max_consumption_value * 0.5,  # Vertical center of the segment (adjust as needed)
                text=f"{current_price:.2f}",  # The price value for this segment
                showarrow=False,
                font=dict(
                    size=9,  # Small font size
                    color="rgba(0, 0, 0, 0.5)"  # Muted text color (e.g., semi-transparent black)
                ),
                textangle=0,  # Horizontal text
            )
            current_price = price_at_slot
            start_slot_for_price = i
    # Add the last segment
    if current_price is not None:
        fig.add_shape(
            type="rect",
            x0=start_slot_for_price,
            x1=SLOTS_PER_DAY,
            y0=0,
            y1=max_consumption_value,
            fillcolor=price_colors.get(current_price, "rgba(0,0,0,0.1)"),
            layer="below",
            line_width=0,
        )

    # Add consumption traces
    fig.add_trace(go.Scatter(x=list(default_consumption_real.keys()), y=list(default_consumption_real.values()),
                             mode='lines', name='Actual Historical Consumption', line=dict(color='lightgray', width=3)))
    fig.add_trace(go.Scatter(x=list(default_consumption.keys()), y=list(default_consumption.values()),
                             mode='lines', name='Default Simulated Consumption',
                             line=dict(color='blue', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=list(optimized_consumption.keys()), y=list(optimized_consumption.values()),
                             mode='lines', name='Optimized Consumption', line=dict(color='red', width=2)))

    # Update layout for better aesthetics
    fig.update_layout(
        title_text=f"Total Energy Consumption and Price Profile for {datetime.date.fromisoformat(results['target_date_str'])}",
        xaxis_title="Time Slot",
        yaxis_title="Power Consumption (kW)",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.2)'),
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, SLOTS_PER_DAY, SLOTS_PER_HOUR * 2)),  # Every 2 hours
            ticktext=[x_labels[i] for i in range(0, SLOTS_PER_DAY, SLOTS_PER_HOUR * 2)],
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)'
        ),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', range=[0, max_consumption_value])
    )
    fig.show()


# --- New Plotting Function: Fitness Evolution ---
def plot_fitness_evolution(fitness_history: List[float], algorithm_name: str):
    fig = go.Figure(data=go.Scatter(y=fitness_history, mode='lines', line=dict(color='green', width=2)))
    fig.update_layout(
        title_text=f"Fitness Evolution for {algorithm_name.upper()} Algorithm",
        xaxis_title="Iteration/Generation",
        yaxis_title="Fitness Value (Cost + Comfort)",
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
    num_devices = len(device_parameters)
    if num_devices == 0:
        print("No devices to plot.")
        return

    rows = int(np.ceil(num_devices / 2))  # Arrange in 2 columns

    fig = make_subplots(rows=rows, cols=2,
                        subplot_titles=list(device_parameters.keys()),
                        vertical_spacing=0.08,
                        horizontal_spacing=0.05)

    x_labels = []  # HH:MM labels for axis ticks and hover
    for i in range(SLOTS_PER_DAY):
        hour = i // SLOTS_PER_HOUR
        minute = (i % SLOTS_PER_HOUR) * SLOT_DURATION_MIN
        x_labels.append(f"{hour:02d}:{minute:02d}")

    # This list will be used for customdata for hover templates
    x_slot_indices = list(range(SLOTS_PER_DAY))

    # Add dummy traces for a combined legend across subplots
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             marker=dict(symbol='star', size=10, color='purple'),
                             line=dict(color='purple', width=0),
                             name='Median Historical Start', showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color="blue", width=2, dash='dot', shape='hv'),
                             name='Default Simulated Consumption', showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color="red", width=2, dash='solid', shape='hv'),
                             name='Optimized Simulated Consumption', showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color="rgba(173, 216, 230, 0.4)", width=10),
                             name='Historical Operation Window', showlegend=True), row=1, col=1)

    row_idx, col_idx = 1, 1
    for i, (dev_name, params) in enumerate(device_parameters.items()):
        alpha_slot = params['alpha_slot']
        beta_slot = params['beta_slot']
        m_slot = int(round(params['m_slot']))  # Median slot
        dev_power = params['power_for_solver']  # Effective power for the solver

        # Max Y for annotations and plot range
        max_y_val = max(dev_power * 1.2, 0.1)  # Ensure at least some height, even for 0 power devices

        # Shade the valid historical operating window (alpha to beta)
        if alpha_slot <= beta_slot:
            fig.add_shape(
                type="rect",
                x0=alpha_slot, x1=beta_slot + 1, y0=0, y1=max_y_val,
                fillcolor="rgba(173, 216, 230, 0.15)",  # Light blue shade
                line_width=0,
                row=row_idx, col=col_idx,
                name="Historical Window",
            )
        else:  # Window wraps around midnight
            fig.add_shape(type="rect", x0=alpha_slot, x1=SLOTS_PER_DAY, y0=0, y1=max_y_val,
                          fillcolor="rgba(173, 216, 230, 0.15)", line_width=0, row=row_idx, col=col_idx)
            fig.add_shape(type="rect", x0=0, x1=beta_slot + 1, y0=0, y1=max_y_val,
                          fillcolor="rgba(173, 216, 230, 0.15)", line_width=0, row=row_idx, col=col_idx)

        # Add a marker for m_slot (median historical start)
        fig.add_trace(go.Scatter(
            x=[m_slot], y=[max_y_val * 0.9], mode='markers',
            marker=dict(symbol='star', size=10, color='purple'),
            name='Median Historical Start',
            customdata=[x_labels[m_slot]],  # Single custom data point for the marker
            hovertemplate=f"Median Start: {dev_name}<br>Slot: %{{x}}<br>Time: %{{customdata}}<extra></extra>",
            showlegend=False
        ), row=row_idx, col=col_idx)

        # Plot Default Simulated Consumption (using stepped line)
        # The consumption data already contains 0s for off slots, so it's ready for plotting.
        default_profile_data = default_individual_consumption.get(dev_name, {s: 0.0 for s in x_slot_indices})
        default_y = [default_profile_data.get(s, 0.0) for s in x_slot_indices]  # Ensure all slots covered

        fig.add_trace(go.Scatter(
            x=x_slot_indices, y=default_y, mode='lines',
            line=dict(color="blue", width=2, dash='dot', shape='hv'),  # 'hv' for horizontal-vertical steps
            name='Default Simulated Consumption',
            customdata=x_labels,  # Pass HH:MM labels for all slots
            hovertemplate=f"Default ON: {dev_name}<br>Slot: %{{x}}<br>Time: %{{customdata}}<br>Power: %{{y:.2f}} kW<extra></extra>",
            showlegend=False
        ), row=row_idx, col=col_idx)

        # Plot Optimized Simulated Consumption (using stepped line)
        optimized_profile_data = optimized_individual_consumption.get(dev_name, {s: 0.0 for s in x_slot_indices})
        optimized_y = [optimized_profile_data.get(s, 0.0) for s in x_slot_indices]  # Ensure all slots covered

        fig.add_trace(go.Scatter(
            x=x_slot_indices, y=optimized_y, mode='lines',
            line=dict(color="red", width=2, dash='solid', shape='hv'),
            name='Optimized Simulated Consumption',
            customdata=x_labels,  # Pass HH:MM labels for all slots
            hovertemplate=f"Optimized ON: {dev_name}<br>Slot: %{{x}}<br>Time: %{{customdata}}<br>Power: %{{y:.2f}} kW<extra></extra>",
            showlegend=False
        ), row=row_idx, col=col_idx)

        # Update X-axis for each subplot
        fig.update_xaxes(
            title_text="Time Slot",
            tickmode='array',
            tickvals=list(range(0, SLOTS_PER_DAY, SLOTS_PER_HOUR * 2)),  # Every 2 hours
            ticktext=[x_labels[k] for k in range(0, SLOTS_PER_DAY, SLOTS_PER_HOUR * 2)],
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)',
            row=row_idx, col=col_idx
        )
        fig.update_yaxes(title_text="Power (kW)", range=[0, max_y_val], row=row_idx, col=col_idx)

        # Move to next subplot position
        col_idx += 1
        if col_idx > 2:
            col_idx = 1
            row_idx += 1

    fig.update_layout(
        title_text="Individual Device Schedules and Historical Operation Windows",
        height=400 * rows,  # Adjust height based on number of rows
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
            max_iter=100,
            picLimit=1.5  # NEW: Enforce a 3.5 kW peak consumption limit
        )
        planning_results['target_date_str'] = date  # Add target date to results for plot title

        print("\n--- Planning Results ---")
        print(f"Default Cost: {planning_results['default_cost']:.2f} DA")
        print(f"Optimized Cost: {planning_results['optimized_cost']:.2f} DA")
        print(f"Cost Savings: {planning_results['default_cost'] - planning_results['optimized_cost']:.2f} DA")
        print(
            f"Percentage Savings: {((planning_results['default_cost'] - planning_results['optimized_cost']) / planning_results['default_cost']) * 100:.2f}%")

        print("\n--- Total Energy Consumed (kW-hours) ---")
        slot_duration_hours = SLOT_DURATION_MIN / 60.0
        total_energy_real = sum(planning_results['default_consumption_real'].values()) * slot_duration_hours
        total_energy_default_sim = sum(planning_results['default_consumption'].values()) * slot_duration_hours
        total_energy_optimized_sim = sum(planning_results['optimized_consumption'].values()) * slot_duration_hours

        print(f"Actual Historical Total Energy: {total_energy_real:.2f} kWh")
        print(f"Default Simulated Total Energy: {total_energy_default_sim:.2f} kWh")
        print(f"Optimized Simulated Total Energy: {total_energy_optimized_sim:.2f} kWh")

        print("\n--- Schedules ---")
        print("Default Schedule (Device -> Start Slot):")
        for d, s in planning_results['default_planning'].items():
            print(f"  {d}: Slot {s} ({s // SLOTS_PER_HOUR:02d}:{(s % SLOTS_PER_HOUR) * SLOT_DURATION_MIN:02d})")
        print("Optimized Schedule (Device -> Start Slot):")
        for d, s in planning_results['optimized_planning'].items():
            print(f"  {d}: Slot {s} ({s // SLOTS_PER_HOUR:02d}:{(s % SLOTS_PER_HOUR) * SLOT_DURATION_MIN:02d})")

        # Plot the results (total consumption)
        plot_total_consumption_results(planning_results)

        # Plot fitness evolution
        plot_fitness_evolution(planning_results['fitness_history'],
                               'csa')  # Assuming 'csa' algorithm is used for this run

        # Plot individual device schedules
        plot_individual_device_schedules(
            planning_results['device_parameters'],
            planning_results['default_individual_consumption'],  # Pass individual consumption data
            planning_results['optimized_individual_consumption']  # Pass individual consumption data
        )

    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")


main()