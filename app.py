import gradio as gr
import requests

BASE_URL = "http://localhost:8000"

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
    payload = {
        "action": {
            "action": action
        }
    }
    res = requests.post(f"{BASE_URL}/step", json=payload)
    return res.json()


# -----------------------------
# MAIN FUNCTION (UI LOGIC)
# -----------------------------
def run_step(difficulty, action):
    reset_response = reset_env(difficulty)

    step_response = step_env(action)

    obs = step_response["observation"]

    return (
        obs["api_status"],
        round(obs["latency"], 2),
        obs["retry_count"],
        round(obs["api_cost"], 3),
        obs["system_load"],
        round(step_response["reward"], 2),
        step_response["done"]
    )


# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
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

    with gr.Row():
        api_status = gr.Textbox(label="API Status")
        latency = gr.Number(label="Latency (ms)")
        retry_count = gr.Number(label="Retry Count")

    with gr.Row():
        api_cost = gr.Number(label="API Cost")
        system_load = gr.Textbox(label="System Load")

    with gr.Row():
        reward = gr.Number(label="Reward")
        done = gr.Checkbox(label="Done")

    run_btn.click(
        fn=run_step,
        inputs=[difficulty, action],
        outputs=[api_status, latency, retry_count, api_cost, system_load, reward, done]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)