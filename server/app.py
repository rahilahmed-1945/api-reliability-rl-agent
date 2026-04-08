from fastapi import FastAPI
from openenv.core.env_server import create_fastapi_app
from server.environment import APIEnvironment
from models import APIAction, APIObservation

# Create OpenEnv app
openenv_app = create_fastapi_app(
    APIEnvironment,
    action_cls=APIAction,
    observation_cls=APIObservation
)

# Main FastAPI app
app = FastAPI()

# 🔥 Mount OpenEnv at root
app.mount("/", openenv_app)