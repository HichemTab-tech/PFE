# server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import logic functions from optimization_logic
from optimization_logic import (
    generate_planning, PlanningResponse, PlanningRequest
)

app = FastAPI()

# --- Allow CORS for frontend ---
# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.post("/planning", response_model=PlanningResponse)
async def generate_planning_req(req: PlanningRequest):
    print(req)
    try:
        results = generate_planning(req.date, req.start_hour, req.algorithm)

        # return {
        #     "slot_duration_min": SLOT_DURATION_MIN,
        #     "devices_info": effective_devices,  # Keeping the full device info for potential display
        #     "default_planning": default_schedule_for_display,
        #     "optimized_planning": optimized_schedule,
        #     "default_cost": default_cost,
        #     "optimized_cost": optimized_cost,
        #     "default_consumption_real": default_consumption_real,
        #     "default_consumption": default_consumption,
        #     "optimized_consumption": optimized_consumption,
        #     "price_profile": price_profile.to_dict()  # Convert to dict for easier passing
        # }
        return PlanningResponse(
            devices=results['devices_info'],  # Changed from results.devices_info
            default_planning=results['default_planning'],  # Changed from results.default_planning
            optimized_planning=results['optimized_planning'],  # Changed from results.optimized_schedule
            default_cost=results['default_cost'],  # Changed from results.default_cost
            optimized_cost=results['optimized_cost'],  # Changed from results.optimized_cost
            default_consumption_real=results['default_consumption_real'],
            # Changed from results.default_consumption_real
            default_consumption=results['default_consumption'],  # Changed from results.default_consumption
            optimized_consumption=results['optimized_consumption'],  # Changed from results.optimized_consumption
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
