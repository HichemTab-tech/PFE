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
    try:
        return generate_planning(req.date, req.start_hour, req.algorithm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
