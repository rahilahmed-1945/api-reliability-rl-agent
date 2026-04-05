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
epsilon = 0.2


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
# AGENT (ε-greedy)
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
    if reward >= 10:
        return 1.0
    elif reward >= 0:
        return 0.5
    else:
        return 0.0


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

Rules:
- latency < 100 ms = GOOD
- if status = success → DO NOT retry/switch/cache
- use_cache only when latency >150 ms
- retry only when failed
- switching API without failure is BAD

Explain in ONE short line.
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.4
        )

        return f"{label} - {response.choices[0].message.content.strip()}"

    except:
        return f"{label} - Explanation unavailable"


# -----------------------------
# RL EPISODE RUN
# -----------------------------
def run_step(difficulty, _):

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

    score = compute_score(reward)
    label = get_label(reward)
    explanation = explain_action(obs, action, label)

    return (
        obs["api_status"],
        round(obs["latency"], 2),
        obs["retry_count"],
        round(obs["api_cost"], 3),
        obs["system_load"],
        round(reward, 2),
        score,
        explanation,
        done
    )


# -----------------------------
# RESET BUTTON
# -----------------------------
def manual_reset(difficulty):
    reset_env(difficulty)
    return "Environment reset"


# -----------------------------
# UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 API Reliability RL Agent")
    gr.Markdown("Autonomous decision-making with reinforcement learning")

    with gr.Row():
        difficulty = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Difficulty")

    run_btn = gr.Button("Run RL Episode")
    reset_btn = gr.Button("Reset Environment")

    status_msg = gr.Textbox(label="Status")

    with gr.Row():
        api_status = gr.Textbox(label="API Status")
        latency = gr.Number(label="Latency (ms)")
        retry_count = gr.Number(label="Retry Count")

    with gr.Row():
        api_cost = gr.Number(label="API Cost")
        system_load = gr.Textbox(label="System Load")

    with gr.Row():
        reward = gr.Number(label="Reward")
        score = gr.Number(label="Score (0–1)")

    explanation = gr.Textbox(label="🤖 AI Explanation")
    done = gr.Checkbox(label="Done")

    run_btn.click(
        run_step,
        [difficulty, gr.Textbox(visible=False)],
        [
            api_status,
            latency,
            retry_count,
            api_cost,
            system_load,
            reward,
            score,
            explanation,
            done
        ]
    )

    reset_btn.click(manual_reset, [difficulty], [status_msg])

demo.launch(server_name="0.0.0.0", server_port=7860)