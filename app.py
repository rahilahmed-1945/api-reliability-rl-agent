import gradio as gr
import requests
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


# -----------------------------
# RESET ENV
# -----------------------------
def reset_env(difficulty):
    res = requests.post(f"{BASE_URL}/reset", json={"difficulty": difficulty})
    return res.json()


# -----------------------------
# STEP ENV
# -----------------------------
def step_env(action):
    payload = {"action": {"action": action}}
    res = requests.post(f"{BASE_URL}/step", json=payload)
    return res.json()


# -----------------------------
# SCORE FUNCTION
# -----------------------------
def compute_score(reward):
    if reward >= 10:
        return 1.0
    elif reward >= 0:
        return 0.5
    else:
        return 0.0


# -----------------------------
# LABEL (FIXED THRESHOLD)
# -----------------------------
def get_label(reward):
    return "GOOD" if reward >= 5 else "BAD"


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

Note:
- latency < 150 ms is GOOD
- success is GOOD

Explain briefly.
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.5
        )

        return f"{label} - {response.choices[0].message.content.strip()}"

    except Exception:
        return f"{label} - Explanation unavailable"


# -----------------------------
# MAIN FUNCTION (🔥 FIXED)
# -----------------------------
def run_step(difficulty, action):
    # 🔥 ALWAYS RESET → ensures new random environment
    reset_env(difficulty)

    step_response = step_env(action)

    obs = step_response["observation"]
    reward = step_response["reward"]

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
        step_response["done"]
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
    gr.Markdown("# 🚀 API Reliability RL Environment")
    gr.Markdown("Simulate intelligent decision-making under API failures")

    with gr.Row():
        difficulty = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Difficulty")
        action = gr.Dropdown(
            ["accept", "retry", "switch_api", "use_cache", "return_error"],
            value="accept",
            label="Action"
        )

    run_btn = gr.Button("Run Step")
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
        fn=run_step,
        inputs=[difficulty, action],
        outputs=[
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

    reset_btn.click(
        fn=manual_reset,
        inputs=[difficulty],
        outputs=[status_msg]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)