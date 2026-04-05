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
# GLOBAL STATE
# -----------------------------
env_initialized = False


# -----------------------------
# RESET ENV
# -----------------------------
def reset_env(difficulty):
    global env_initialized
    res = requests.post(f"{BASE_URL}/reset", json={"difficulty": difficulty})
    env_initialized = True
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
# 🤖 AI EXPLANATION (FINAL FIXED)
# -----------------------------
def explain_action(obs, action):
    try:
        prompt = f"""
You are an expert backend engineer.

Evaluate THIS API decision.

State:
- API status: {obs['api_status']}
- Latency: {obs['latency']} ms
- Retry count: {obs['retry_count']}
- System load: {obs['system_load']}

Action: {action}

STRICT RULES:
- Output ONLY ONE line
- Start with either GOOD or BAD
- Then a single short explanation
- Do NOT give multiple answers
- Do NOT continue after the explanation
- Use actual values in reasoning

Format strictly:
GOOD or BAD - explanation
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=25,        # 🔥 prevents overflow
            temperature=0.6,      # 🔥 controlled variation
            stop=["\n"]           # 🔥 stops extra output
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return "AI explanation unavailable"


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def run_step(difficulty, action):
    global env_initialized

    if not env_initialized:
        reset_env(difficulty)

    step_response = step_env(action)

    obs = step_response["observation"]
    reward = step_response["reward"]

    score = compute_score(reward)
    explanation = explain_action(obs, action)

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
# RESET BUTTON HANDLER
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
            ["retry", "switch_api", "use_cache", "return_error"],
            value="retry",
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