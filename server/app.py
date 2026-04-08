from fastapi import FastAPI
from pydantic import BaseModel
from server.environment import APIEnvironment

app = FastAPI()

# Initialize environment
env = APIEnvironment()


# -------- Request Models --------
class ResetRequest(BaseModel):
    difficulty: str = "easy"


class ActionRequest(BaseModel):
    action: dict


# -------- RESET --------
@app.post("/reset")
def reset(req: ResetRequest):
    obs = env.reset(difficulty=req.difficulty)
    return {
        "observation": obs.dict()
    }


# -------- STEP --------
@app.post("/step")
def step(req: ActionRequest):
    action = req.action
    obs = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": obs.reward,
        "done": obs.done
    }