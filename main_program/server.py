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
    numeric = df_day.select_dtypes(include=['float64', 'int64']).columns
    smart = [c for c in numeric for d in devices if d in c]
    baseline = df_day[numeric].drop(columns=smart).sum(axis=1)
    lp = baseline.groupby(baseline.index.hour).mean().reindex(range(24), fill_value=0)
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
@app.get("/devices")
async def get_devices():
    return {d: {"w": v["w"], "lambda": v["lambda"]} for d, v in devices.items()}

@app.post("/planning", response_model=PlanningResponse)
async def generate_planning(req: PlanningRequest):
    target = pd.to_datetime(req.date).date()
    df_day = df[df.index.date == target]
    if df_day.empty:
        raise HTTPException(status_code=404, detail=f"No data for date {target}")

    # Use custom params if provided
    effective_devices = devices.copy()
    if req.custom_params:
        for name, params in req.custom_params.items():
            if name in effective_devices:
                effective_devices[name]["w"] = params.w
                effective_devices[name]["lambda"] = params.lambda_
                # If 'power' is not already there, set a default fake value (1.5kW)
                if "power" not in effective_devices[name]:
                    effective_devices[name]["power"] = 1.5

    # Filter by start_hour
    df_filtered = df_day[df_day.index.hour >= req.start_hour]

    params = extract_time_params(df_filtered, effective_devices)
    price_profile = build_price_profile()
    load_profile = build_load_profile(df_filtered, effective_devices)

    # Default planning (basic alpha hour)
    default_schedule = {d: p['alpha'] // 3600 for d, p in params.items()}






    # unpack
    α     = {d: params['alpha']//3600        for d, params in devices.items()}
    β     = {d: params['beta']//3600 - 1     for d, params in devices.items()}
    LOT_h = {d: params['LOT']/3600           for d, params in devices.items()}
    P     = {d: params['power']              for d, params in devices.items()}
    W     = {d: params['w']                  for d, params in devices.items()}
    L     = {d: params['lambda']             for d, params in devices.items()}
    M     = {d: params['m']                  for d, params in devices.items()}

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
    optimized_schedule = solver.run(params, price_profile, load_profile, seed=42)

    # Cost estimations
    def estimate_cost(schedule, load_profile, price_profile):
        cost = 0.0
        for d, h in schedule.items():
            lot_hours = params[d]['LOT'] / 3600  # LOT in hours
            hourly_cost = price_profile[h] * effective_devices[d]['lambda']
            cost += lot_hours * hourly_cost
        # Add baseline load cost
        cost += (load_profile[req.start_hour:].sum()) * price_profile[req.start_hour:].mean()
        return round(cost, 2)

    # Consumption estimations
    def estimate_consumption(schedule, effective_devices, params):
        hourly_consumption = {h: 0.0 for h in range(24)}

        for d, start_hour in schedule.items():
            lot = params[d]['LOT'] / 3600  # LOT in hours
            duration = int(round(lot))  # Round to integer hours (simplified)
            power = effective_devices[d].get('power', 1.5)

            # Mark hours where device consumes power
            for h in range(start_hour, start_hour + duration):
                if 0 <= h < 24:
                    hourly_consumption[h] += power

        # Round all values to 2 decimals
        hourly_consumption = {h: round(kW, 2) for h, kW in hourly_consumption.items()}

        return hourly_consumption

    # New function:
    def estimate_real_consumption(df_day):
        hourly_consumption = df_day.groupby(df_day.index.hour)['use [kW]'].mean()
        hourly_consumption = hourly_consumption.reindex(range(24), fill_value=0)
        return {hour: round(kW, 2) for hour, kW in hourly_consumption.items()}

    default_cost = estimate_cost(default_schedule, load_profile, price_profile)
    optimized_cost = estimate_cost(optimized_schedule, load_profile, price_profile)

    default_consumption_real = estimate_real_consumption(df_day)
    default_consumption = estimate_consumption(default_schedule, effective_devices, params)
    optimized_consumption = estimate_consumption(optimized_schedule, effective_devices, params)

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
