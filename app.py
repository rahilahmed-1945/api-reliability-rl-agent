import gradio as gr
import requests
import os
import random
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

client = OpenAI(
    api_key=os.getenv("HF_TOKEN", "dummy"),
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

ACTIONS = ["accept", "retry", "switch_api", "use_cache", "return_error"]

Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.6

trained = False


# -----------------------------
# ENV
# -----------------------------
def reset_env(difficulty):
    return requests.post(f"{BASE_URL}/reset", json={"difficulty": difficulty}).json()

def step_env(action):
    return requests.post(f"{BASE_URL}/step", json={"action": {"action": action}}).json()


# -----------------------------
# STATE
# -----------------------------
def get_state(obs):
    return (
        obs["api_status"],
        int(obs["latency"] // 50),
        obs["system_load"]
    )


# -----------------------------
# AGENT
# -----------------------------
def agent(obs):
    state = get_state(obs)

    if state not in Q or random.random() < epsilon:
        return random.choice(ACTIONS)

    return max(Q[state], key=Q[state].get)


# -----------------------------
# UPDATE
# -----------------------------
def update_q(obs, action, reward, next_obs):
    reward = max(min(reward, 10), -10)

    s = get_state(obs)
    ns = get_state(next_obs)

    if s not in Q:
        Q[s] = {a: 0 for a in ACTIONS}
    if ns not in Q:
        Q[ns] = {a: 0 for a in ACTIONS}

    Q[s][action] += alpha * (
        reward + gamma * max(Q[ns].values()) - Q[s][action]
    )


# -----------------------------
# LABEL
# -----------------------------
def get_label(reward):
    if reward >= 8:
        return "GOOD"
    elif reward >= 1:
        return "OKAY"
    else:
        return "BAD"


def compute_score(reward):
    return 1 if reward >= 8 else (0.5 if reward >= 1 else 0)


# -----------------------------
# TRAIN (RUN ONCE)
# -----------------------------
def train_agent(difficulty):
    global trained

    EPISODES = 12
    MAX_STEPS = 3

    for _ in range(EPISODES):
        obs = reset_env(difficulty)["observation"]
        done = False

        while not done:
            action = agent(obs)
            res = step_env(action)

            next_obs = res["observation"]
            reward = res["reward"]

            update_q(obs, action, reward, next_obs)

            obs = next_obs
            done = res["done"]

    trained = True
    return f"Training done | States learned: {len(Q)}"


# -----------------------------
# RUN (FAST)
# -----------------------------
def run_agent(difficulty):
    if not trained:
        return ("Train first!", 0, 0, 0, "", 0, 0, "No model", False, 0)

    obs = reset_env(difficulty)["observation"]

    action = agent(obs)
    res = step_env(action)

    obs = res["observation"]
    reward = res["reward"]

    label = get_label(reward)

    return (
        obs["api_status"],
        round(obs["latency"], 2),
        obs["retry_count"],
        round(obs["api_cost"], 3),
        obs["system_load"],
        round(reward, 2),
        compute_score(reward),
        f"{label} - Agent decision",
        True,
        len(Q)
    )


# -----------------------------
# UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 API Reliability RL Agent (FINAL)")
    gr.Markdown("Train once → Run fast decisions")

    difficulty = gr.Dropdown(["easy", "medium", "hard"], value="easy")

    train_btn = gr.Button("🔥 Train Agent")
    run_btn = gr.Button("⚡ Run Agent")

    status = gr.Textbox(label="Status")

    api_status = gr.Textbox(label="API Status")
    latency = gr.Number(label="Latency")
    retry = gr.Number(label="Retries")

    cost = gr.Number(label="Cost")
    load = gr.Textbox(label="Load")

    reward = gr.Number(label="Reward")
    score = gr.Number(label="Score")

    explanation = gr.Textbox(label="Explanation")
    done = gr.Checkbox()

    states = gr.Number(label="States Learned")

    train_btn.click(train_agent, [difficulty], [status])

    run_btn.click(
        run_agent,
        [difficulty],
        [api_status, latency, retry, cost, load,
         reward, score, explanation, done, states]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)