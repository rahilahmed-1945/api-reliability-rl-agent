import requests
import random
import time
import os
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL = os.getenv("BASE_URL", "http://0.0.0.0:8000")

client = OpenAI(
    api_key=os.getenv("HF_TOKEN", "dummy"),
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

random.seed(42)

# -----------------------------
# RESET ENV
# -----------------------------
def reset_env(difficulty="easy"):
    return requests.post(f"{BASE_URL}/reset", json={"difficulty": difficulty}).json()


# -----------------------------
# STEP ENV
# -----------------------------
def step_env(action):
    payload = {"action": {"action": action}}
    return requests.post(f"{BASE_URL}/step", json=payload).json()


# -----------------------------
# SCORE FUNCTION
# -----------------------------
def compute_score(total_reward):
    if total_reward >= 30:
        return 1.0
    elif total_reward >= 0:
        return 0.5
    else:
        return 0.0


# -----------------------------
# 🔥 LLM AGENT (REAL AI)
# -----------------------------
def llm_agent(obs):
    prompt = f"""
You are an API reliability decision agent.

State:
- status: {obs['observation']['api_status']}
- latency: {obs['observation']['latency']}
- retries: {obs['observation']['retry_count']}
- load: {obs['observation']['system_load']}

Choose ONE action from:
retry, switch_api, use_cache, return_error

Only output the action.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5
        )

        action = response.choices[0].message.content.strip().lower()

        if action not in ["retry", "switch_api", "use_cache", "return_error"]:
            return "retry"

        return action

    except:
        return "retry"


# -----------------------------
# RUN EPISODE
# -----------------------------
def run_episode(difficulty="easy"):
    print(f"[START] task={difficulty} env=api_env model={MODEL_NAME}")

    obs = reset_env(difficulty)
    total_reward = 0
    rewards_list = []
    step_count = 0
    done = False

    MAX_STEPS = 10

    while step_count < MAX_STEPS:
        step_count += 1

        try:
            action = llm_agent(obs)

            result = step_env(action)

            reward = result["reward"]
            done = result["done"]

            total_reward += reward
            rewards_list.append(round(reward, 2))

            print(f"[STEP] step={step_count} action={action} reward={round(reward,2)} done={str(done).lower()} error=null")

            if done:
                success = result["observation"]["api_status"] == "success"
                break

            obs = result

        except Exception as e:
            print(f"[STEP] step={step_count} action=error reward=0.00 done=true error={str(e)}")
            success = False
            break

        time.sleep(0.1)

    if not done:
        success = False

    score = compute_score(total_reward)
    rewards_str = ",".join([f"{r:.2f}" for r in rewards_list])

    print(f"[END] success={str(success).lower()} steps={step_count} score={score:.2f} rewards={rewards_str}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    for difficulty in ["easy", "medium", "hard"]:
        run_episode(difficulty)