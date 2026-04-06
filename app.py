import gradio as gr
import requests
import os
import random
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")  # 🔥 fixed

client = OpenAI(
    api_key=os.getenv("HF_TOKEN", "dummy"),
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

ACTIONS = ["accept", "retry", "switch_api", "use_cache", "return_error"]

Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.7   # 🔥 HIGH exploration (fix repetition)


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
# STATE (MORE GRANULAR 🔥)
# -----------------------------
def get_state(obs):
    return (
        obs["api_status"],
        int(obs["latency"] // 50),   # 🔥 FIXED (more variation)
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
# SCORE
# -----------------------------
def compute_score(reward):
    return 1.0 if reward >= 10 else (0.5 if reward >= 0 else 0.0)


def get_label(reward):
    return "GOOD" if reward >= 4 else "BAD"


# -----------------------------
# EXPLANATION (NO FAIL FALLBACK)
# -----------------------------
def explain_action(obs, action, label):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"Explain briefly why action {action} is {label} given status={obs['api_status']}, latency={obs['latency']}, load={obs['system_load']}."
            }],
            max_tokens=20,
            temperature=0.3
        )
        return f"{label} - {response.choices[0].message.content.strip()}"
    except:
        return f"{label} - Based on latency, status and retries"


# -----------------------------
# MAIN RUN
# -----------------------------
def run_step(difficulty, mode, manual_action):

    EPISODES = 5
    MAX_STEPS = 3

    last_obs = None
    last_reward = 0
    last_action = None

    for _ in range(EPISODES):

        obs = reset_env(difficulty)["observation"]
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:

            if mode == "agent":
                action = agent(obs)
            else:
                action = manual_action

            res = step_env(action)
            next_obs = res["observation"]
            reward = res["reward"]

            update_q(obs, action, reward, next_obs)

            obs = next_obs
            done = res["done"]
            steps += 1

            last_obs = obs
            last_reward = reward
            last_action = action

    states = len(Q)

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
        states
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
    gr.Markdown("Manual vs RL Agent comparison")

    difficulty = gr.Dropdown(["easy", "medium", "hard"], value="easy")

    mode = gr.Dropdown(["agent", "manual"], value="agent", label="Mode")

    manual_action = gr.Dropdown(
        ACTIONS,
        value="accept",
        label="Manual Action"
    )

    run_btn = gr.Button("Run")
    reset_btn = gr.Button("Reset")

    status = gr.Textbox()

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

    run_btn.click(
        run_step,
        [difficulty, mode, manual_action],
        [api_status, latency, retry, cost, load, reward, score, explanation, done, states]
    )

    reset_btn.click(manual_reset, [difficulty], [status])

demo.launch(server_name="0.0.0.0", server_port=7860)