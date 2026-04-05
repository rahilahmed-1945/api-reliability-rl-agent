import gradio as gr
import requests
import os
import random
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

ACTIONS = ["accept", "retry", "switch_api", "use_cache", "return_error"]

# Q-learning params
Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.4


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
# STATE ENCODING
# -----------------------------
def get_state(obs):
    return (
        obs["api_status"],
        int(obs["latency"] // 100),
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
# Q UPDATE
# -----------------------------
def update_q(obs, action, reward, next_obs):
    state = get_state(obs)
    next_state = get_state(next_obs)

    if state not in Q:
        Q[state] = {a: 0 for a in ACTIONS}
    if next_state not in Q:
        Q[next_state] = {a: 0 for a in ACTIONS}

    Q[state][action] += alpha * (
        reward + gamma * max(Q[next_state].values()) - Q[state][action]
    )


# -----------------------------
# SCORE + LABEL
# -----------------------------
def compute_score(reward):
    return 1.0 if reward >= 10 else (0.5 if reward >= 0 else 0.0)


def get_label(reward):
    return "GOOD" if reward >= 4 else "BAD"


# -----------------------------
# AI EXPLANATION
# -----------------------------
def explain_action(obs, action, label):
    try:
        prompt = f"""
Decision: {label}

State:
- status: {obs['api_status']}
- latency: {obs['latency']}
- load: {obs['system_load']}
- retries: {obs['retry_count']}

Explain in ONE short line.
"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=25,
            temperature=0.4
        )
        return f"{label} - {response.choices[0].message.content.strip()}"

    except:
        return f"{label} - Explanation unavailable"


# -----------------------------
# FAST RL LOOP (FIXED)
# -----------------------------
def run_step(difficulty, _):

    EPISODES = 5        # 🔥 reduced
    MAX_STEPS = 3       # 🔥 reduced

    last_obs = None
    last_reward = 0
    last_action = None

    for _ in range(EPISODES):

        obs = reset_env(difficulty)["observation"]
        done = False
        step_count = 0

        while not done and step_count < MAX_STEPS:
            action = agent(obs)

            res = step_env(action)
            next_obs = res["observation"]
            reward = res["reward"]

            update_q(obs, action, reward, next_obs)

            obs = next_obs
            done = res["done"]
            step_count += 1

            last_obs = obs
            last_reward = reward
            last_action = action

    states_learned = len(Q)

    score = compute_score(last_reward)
    label = get_label(last_reward)
    explanation = explain_action(last_obs, last_action, label)

    return (
        last_obs["api_status"],
        round(last_obs["latency"], 2),
        last_obs["retry_count"],
        round(last_obs["api_cost"], 3),
        last_obs["system_load"],
        round(last_reward, 2),
        score,
        explanation,
        True,
        states_learned
    )


# -----------------------------
# RESET
# -----------------------------
def manual_reset(difficulty):
    reset_env(difficulty)
    return "Environment reset"


# -----------------------------
# UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 API Reliability RL Agent")
    gr.Markdown("Fast RL training (optimized for Hugging Face)")

    difficulty = gr.Dropdown(["easy", "medium", "hard"], value="easy")

    run_btn = gr.Button("Run RL Training")
    reset_btn = gr.Button("Reset Environment")

    status_msg = gr.Textbox()

    api_status = gr.Textbox(label="API Status")
    latency = gr.Number(label="Latency")
    retry_count = gr.Number(label="Retries")

    api_cost = gr.Number(label="Cost")
    system_load = gr.Textbox(label="Load")

    reward = gr.Number(label="Reward")
    score = gr.Number(label="Score")

    explanation = gr.Textbox(label="Explanation")
    done = gr.Checkbox(label="Done")

    states = gr.Number(label="States Learned")

    run_btn.click(
        run_step,
        [difficulty, gr.Textbox(visible=False)],
        [api_status, latency, retry_count, api_cost, system_load,
         reward, score, explanation, done, states]
    )

    reset_btn.click(manual_reset, [difficulty], [status_msg])

demo.launch(server_name="0.0.0.0", server_port=7860)