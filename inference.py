import requests
import time
import os
import random
from typing import List

from openai import OpenAI

# -----------------------------
# CONFIG (MANDATORY)
# -----------------------------
BASE_URL = os.getenv("BASE_URL", "http://0.0.0.0:8000")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
API_KEY = os.getenv("HF_TOKEN", "dummy")

TASK_NAME = "api_reliability"
BENCHMARK = "openenv_api_env"

MAX_STEPS = 10

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ✅ reproducibility
random.seed(42)


# -----------------------------
# LOGGING (STRICT FORMAT)
# -----------------------------
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# -----------------------------
# ENV CALLS
# -----------------------------
def reset_env(difficulty):
    return requests.post(f"{BASE_URL}/reset", json={"difficulty": difficulty}).json()


def step_env(action):
    return requests.post(
        f"{BASE_URL}/step",
        json={"action": {"action": action}}
    ).json()


# -----------------------------
# BASELINE AGENT (SIMPLE)
# -----------------------------
def agent(obs):
    o = obs["observation"]

    status = o["api_status"]
    latency = o["latency"]
    retries = o["retry_count"]
    load = o["system_load"]

    if status == "success" and latency < 120:
        return "accept"

    if status == "failed":
        return "retry" if retries < 2 else "switch_api"

    if latency > 150 or load == "high":
        return "use_cache"

    return "retry"


# -----------------------------
# RUN EPISODE
# -----------------------------
def run_episode(difficulty):

    log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    success = False

    try:
        obs = reset_env(difficulty)

        for step in range(1, MAX_STEPS + 1):

            action = agent(obs)

            try:
                result = step_env(action)

                reward = result["reward"]
                done = result["done"]
                error = "null"

            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, error)

            if done:
                success = result["observation"]["api_status"] == "success"
                break

            obs = result
            time.sleep(0.05)

    except Exception as e:
        print(f"[STEP] step=0 action=error reward=0.00 done=true error={str(e)}")

    # ✅ FINAL SCORE (0–1)
    score = sum(rewards) / len(rewards) if rewards else 0.0
    score = max(min(score, 1.0), 0.0)

    log_end(success, steps_taken, score, rewards)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    for difficulty in ["easy", "medium", "hard"]:
        run_episode(difficulty)