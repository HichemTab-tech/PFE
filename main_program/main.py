import datetime
from typing import Dict, Any

import pandas as pd
import plotly.graph_objects as go

from main_program.data_prep import SLOTS_PER_DAY, SLOTS_PER_HOUR, SLOT_DURATION_MIN
from main_program.optimization_logic import generate_planning

date = "2016-01-05"  # Target date for the simulation

# --- GLOBAL PARAMETERS FOR PEAK CONTROL (CRITICAL: TUNE THESE!) ---
# 1. MAX_PEAK_KW_THRESHOLD: Set this SIGNIFICANTLY lower than your default historical peak
#    if you want to force reduction. Check the "Default Simulated Peak" value.
MAX_PEAK_KW_THRESHOLD = 2.5 # Example: If your historical peak is 4-5kW, 3.5kW is challenging.

# 2. GAMMA_PEAK_PENALTY: This coefficient needs to be LARGE enough for the peak penalty
#    to be numerically competitive with the energy cost and comfort penalty.
#    Start HIGH (e.g., 5000, 10000) and reduce if it's too aggressive.
GAMMA_PEAK_PENALTY = 5000.0 # Increased significantly. Tune this!

# 3. MAX_ITER: Number of iterations for the optimizer. More iterations = more exploration.
#    Often needed for complex multi-objective problems.
OPTIMIZER_MAX_ITER = 1000 # Increased iterations for better convergence
# -------------------------------------------------------------------


# --- Plotting Function (No functional changes, just for visualization) ---
def plot_results(results: Dict[str, Any]):
    slot_duration_min = results["slot_duration_min"]
    default_consumption_real = results["default_consumption_real"]
    default_consumption = results["default_consumption"]
    optimized_consumption = results["optimized_consumption"]
    price_profile = pd.Series(results["price_profile"])

    x_labels = []
    for i in range(SLOTS_PER_DAY):
        hour = i // SLOTS_PER_HOUR
        minute = (i % SLOTS_PER_HOUR) * slot_duration_min
        x_labels.append(f"{hour:02d}:{minute:02d}")

    fig = go.Figure()

    max_consumption_value = max(
        max(default_consumption_real.values()),
        max(default_consumption.values()),
        max(optimized_consumption.values()),
        MAX_PEAK_KW_THRESHOLD
    ) * 1.1

    price_colors = {
        0.10: "rgba(144, 238, 144, 0.2)",  # Light green
        0.15: "rgba(255, 255, 0, 0.2)",    # Yellow
        0.20: "rgba(255, 165, 0, 0.2)",    # Orange
        0.22: "rgba(255, 99, 71, 0.2)"     # Tomato/Red
    }

    current_price = None
    start_slot_for_price = 0
    for i in range(SLOTS_PER_DAY):
        price_at_slot = price_profile.get(i)
        if current_price is None:
            current_price = price_at_slot
            start_slot_for_price = i
        elif price_at_slot != current_price:
            fig.add_shape(
                type="rect", x0=start_slot_for_price, x1=i, y0=0, y1=max_consumption_value,
                fillcolor=price_colors.get(current_price, "rgba(0,0,0,0.1)"), layer="below", line_width=0,
            )
            fig.add_annotation(
                x=(start_slot_for_price + i) / 2, y=max_consumption_value * 0.5, text=f"${current_price:.2f}",
                showarrow=False, font=dict(size=9, color="rgba(0, 0, 0, 0.5)"), textangle=0,
            )
            current_price = price_at_slot
            start_slot_for_price = i
    if current_price is not None:
        fig.add_shape(
            type="rect", x0=start_slot_for_price, x1=SLOTS_PER_DAY, y0=0, y1=max_consumption_value,
            fillcolor=price_colors.get(current_price, "rgba(0,0,0,0.1)"), layer="below", line_width=0,
        )
        fig.add_annotation(
            x=(start_slot_for_price + SLOTS_PER_DAY) / 2, y=max_consumption_value * 0.5, text=f"${current_price:.2f}",
            showarrow=False, font=dict(size=9, color="rgba(0, 0, 0, 0.5)"), textangle=0,
        )

    fig.add_shape(
        type="line", x0=0, x1=SLOTS_PER_DAY - 1, y0=MAX_PEAK_KW_THRESHOLD, y1=MAX_PEAK_KW_THRESHOLD,
        line=dict(color="purple", width=2, dash="dash"), name=f"Peak Threshold ({MAX_PEAK_KW_THRESHOLD} kW)",
    )
    fig.add_annotation(
        x=SLOTS_PER_DAY / 2, y=MAX_PEAK_KW_THRESHOLD + max_consumption_value * 0.03,
        text=f"Peak Threshold: {MAX_PEAK_KW_THRESHOLD} kW", showarrow=False,
        font=dict(size=10, color="purple"), textangle=0
    )

    fig.add_trace(go.Scatter(x=list(default_consumption_real.keys()), y=list(default_consumption_real.values()),
                             mode='lines', name='Actual Historical Consumption', line=dict(color='lightgray', width=3)))
    fig.add_trace(go.Scatter(x=list(default_consumption.keys()), y=list(default_consumption.values()),
                             mode='lines', name='Default Simulated Consumption', line=dict(color='blue', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=list(optimized_consumption.keys()), y=list(optimized_consumption.values()),
                             mode='lines', name='Optimized Consumption', line=dict(color='red', width=2)))

    fig.update_layout(
        title_text=f"Energy Consumption and Price Profile for {datetime.date.fromisoformat(results['target_date_str'])}",
        xaxis_title="Time Slot", yaxis_title="Power Consumption (kW)", hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.2)'),
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(tickmode='array', tickvals=list(range(0, SLOTS_PER_DAY, SLOTS_PER_HOUR * 2)),
                   ticktext=[x_labels[i] for i in range(0, SLOTS_PER_DAY, SLOTS_PER_HOUR * 2)],
                   showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
    )
    fig.show()


def main():
    try:
        planning_results = generate_planning(date,
                                             max_peak_kW_threshold=MAX_PEAK_KW_THRESHOLD,
                                             gamma=GAMMA_PEAK_PENALTY,
                                             max_iter=OPTIMIZER_MAX_ITER # Use the new global variable
                                            )
        planning_results['target_date_str'] = date

        print("\n--- Planning Results ---")
        print(f"Default Cost: ${planning_results['default_cost']:.2f}")
        print(f"Optimized Cost: ${planning_results['optimized_cost']:.2f}")
        print(f"Cost Savings: ${planning_results['default_cost'] - planning_results['optimized_cost']:.2f}")
        print(f"Percentage Savings: {((planning_results['default_cost'] - planning_results['optimized_cost']) / planning_results['default_cost']) * 100:.2f}%")

        print("\n--- Total Energy Consumed (kW-hours) ---")
        slot_duration_hours = SLOT_DURATION_MIN / 60.0
        total_energy_real = sum(planning_results['default_consumption_real'].values()) * slot_duration_hours
        total_energy_default_sim = sum(planning_results['default_consumption'].values()) * slot_duration_hours
        total_energy_optimized_sim = sum(planning_results['optimized_consumption'].values()) * slot_duration_hours

        print(f"Actual Historical Total Energy: {total_energy_real:.2f} kWh")
        print(f"Default Simulated Total Energy: {total_energy_default_sim:.2f} kWh")
        print(f"Optimized Simulated Total Energy: {total_energy_optimized_sim:.2f} kWh")

        print("\n--- Peak Power Consumption (kW) ---")
        print(f"Default Simulated Peak: {planning_results['default_peak_kW']:.2f} kW")
        print(f"Optimized Simulated Peak: {planning_results['optimized_peak_kW']:.2f} kW")
        print(f"Peak Threshold: {MAX_PEAK_KW_THRESHOLD:.2f} kW")


        print("\n--- Schedules ---")
        print("Default Schedule (Device -> Start Slot):")
        for d, s in planning_results['default_planning'].items():
            print(f"  {d}: Slot {s} ({s // SLOTS_PER_HOUR:02d}:{(s % SLOTS_PER_HOUR) * SLOT_DURATION_MIN:02d})")
        print("Optimized Schedule (Device -> Start Slot):")
        for d, s in planning_results['optimized_planning'].items():
            print(f"  {d}: Slot {s} ({s // SLOTS_PER_HOUR:02d}:{(s % SLOTS_PER_HOUR) * SLOT_DURATION_MIN:02d})")

        plot_results(planning_results)

    except ValueError as e:
        print(f"Error: {e}")


main()