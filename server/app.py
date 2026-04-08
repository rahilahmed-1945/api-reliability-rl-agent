from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Dict
import logging

from server.environment import APIEnvironment
from models import APIAction   # ✅ IMPORTANT FIX

# -----------------------------
# APP CONFIG
# -----------------------------
app = FastAPI(
    title="API Reliability RL Agent",
    description="Decision intelligence system for API optimization",
    version="1.0.0"
)

# Initialize environment
env = APIEnvironment()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# REQUEST MODELS
# -----------------------------
class ResetRequest(BaseModel):
    difficulty: Literal["easy", "medium", "hard"] = "easy"


class ActionRequest(BaseModel):
    action: Dict[str, str]


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# RESET
# -----------------------------
@app.post("/reset")
def reset(req: ResetRequest):
    try:
        logger.info(f"Reset called: difficulty={req.difficulty}")

        obs = env.reset(difficulty=req.difficulty)

        return {
            "observation": obs.dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# STEP (🔥 FIXED)
# -----------------------------
@app.post("/step")
def step(req: ActionRequest):
    try:
        logger.info(f"Action received: {req.action}")

        # 🔥 FIX: convert dict → APIAction object
        action_obj = APIAction(**req.action)

        obs = env.step(action_obj)

        return {
            "observation": obs.dict(),
            "reward": obs.reward,
            "done": obs.done
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # -----------------------------
# MAIN ENTRYPOINT (🔥 REQUIRED)
# -----------------------------
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()