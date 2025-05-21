import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

from data_import import load_csv_data, load_devices
from data_prep import extract_time_params
from solvers import SolverFactory

app = FastAPI()

# --- Allow CORS for frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load data once on server startup ---
df = load_csv_data("HomeC.csv")
devices = load_devices("devices_with_w.json", "HomeC.csv")

def build_price_profile():
    return pd.Series(
        [0.10]*7 + [0.20]*4 + [0.15]*6 + [0.22]*2 + [0.10]*5,
        index=range(24)
    )

def build_load_profile(df_day, devices):
    # 1) Take the true total‐house power column
    #    (your CSV has "use [kW]" for total load)
    total = df_day["use [kW]"]

    # 2) If you want baseline _excluding_ your smart devices:
    smart = df_day[[c for c in df_day.columns if any(d in c for d in devices)]].sum(axis=1)
    baseline = total - smart
    lp = baseline.groupby(baseline.index.hour).mean()
    return lp

# --- Request/Response models ---
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

    # Filter by start_hour
    df_filtered = df_day[df_day.index.hour >= req.start_hour]

    params = extract_time_params(df_filtered, effective_devices)
    price_profile = build_price_profile()
    load_profile = build_load_profile(df_filtered, effective_devices)

    # Default planning (basic alpha hour)
    default_schedule = { d: int(round(p['m'])) for d,p in params.items() }

    device_power = {}
    for device in devices:
        # find the matching column name
        col = next((c for c in df_day.columns if device in c), None)
        if col is None:
            # fallback if you really don’t have per-device data
            device_power[device] = 1.0
        else:
            # use the mean draw for a realistic average kW
            device_power[device] = df_day[col].mean()

    # now inject these into your devices config:
    for d, p in effective_devices.items():
        p['power'] = device_power[d]





    # unpack
    α     = {d: params_['alpha']//3600        for d, params_ in params.items()}
    β     = {d: params_['beta']//3600 - 1     for d, params_ in params.items()}
    LOT_h = {d: params_['LOT']/3600           for d, params_ in params.items()}
    P     = {d: params_['power']              for d, params_ in params.items()}
    W     = {d: params_['w']                  for d, params_ in params.items()}
    L     = {d: params_['lambda']             for d, params_ in params.items()}
    M     = {d: params_['m']                  for d, params_ in params.items()}

    # valid hours with wrap-around
    valid_hours = {}
    for d in devices:
        a, b = α[d], β[d]
        valid_hours[d] = list(range(a, b+1)) if a<=b else list(range(a,24))+list(range(0,b+1))

    def fitness(n_):
        cost = sum(W[d] * P[d] * LOT_h[d] * price_profile.loc[n_[d]] for d in n_)
        comfort = sum(L[d] * W[d] * (n_[d] - M[d]) ** 2 for d in n_)
        return cost + comfort




    # Select and run solver
    solver = SolverFactory(req.algorithm, fitness=fitness, params={
        'α': α,
        'β': β,
        'LOT': LOT_h,
        'P': P,
        'W': W,
        'L': L,
        'M': M,
        'valid_hours': valid_hours
    })
    optimized_schedule = solver.run(params, seed=42)

    # Cost estimations
    def estimate_cost(schedule):
        device_cost = 0.0
        # 1) Cost for scheduled devices
        for d, h in schedule.items():
            power_kw = effective_devices[d].get('power', 1.0)  # kW draw
            duration_h = params[d]['LOT'] / 3600  # hours of operation
            device_cost += power_kw * duration_h * price_profile[h]  # kW·h × $/kWh

        # 2) Baseline household load cost (now hour-by-hour)
        baseline_cost = sum(
            load_profile[h] * price_profile[h]
            for h in load_profile.index
        )

        return device_cost + baseline_cost

    # Consumption estimations (fractional hours)
    def estimate_consumption(schedule):
        hourly_consumption = {h: 0.0 for h in range(24)}

        for d, start_hour in schedule.items():
            power_kw = effective_devices[d]['power']
            duration_h = params[d]['LOT'] / 3600.0  # fractional hours
            remaining = duration_h
            hour = start_hour

            # Spread the device’s energy use across hours
            while remaining > 0:
                dt = min(1.0, remaining)
                hourly_consumption[hour % 24] += power_kw * dt
                remaining -= dt
                hour += 1

        # Round to 2 decimals
        return {h: kW for h, kW in hourly_consumption.items()}

    # New function:
    def estimate_real_consumption():
        hourly_consumption = df_day.groupby('hour')['use [kW]'].mean()
        hourly_consumption = hourly_consumption.reindex(range(24), fill_value=0)
        return {hour: kW for hour, kW in hourly_consumption.items()}

    default_cost = estimate_cost(default_schedule)
    optimized_cost = estimate_cost(optimized_schedule)

    default_consumption_real = estimate_real_consumption()
    default_consumption = estimate_consumption(default_schedule)
    optimized_consumption = estimate_consumption(optimized_schedule)

    # Build response
    devices_info = {d: DeviceParams(w=v['w'], lambda_=v['lambda']) for d, v in effective_devices.items()}

    return PlanningResponse(
        devices=devices_info,
        default_planning=default_schedule,
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
