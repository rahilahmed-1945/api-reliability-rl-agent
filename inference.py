import requests
import random
import time

BASE_URL = "http://127.0.0.1:8000"


# -----------------------------
# RESET ENVIRONMENT
# -----------------------------
def reset_env(difficulty="easy"):
    response = requests.post(f"{BASE_URL}/reset", json={"difficulty": difficulty})
    return response.json()


# -----------------------------
# STEP FUNCTION
# -----------------------------
def step_env(action):
    payload = {
        "action": {
            "action": action
        }
    }
    response = requests.post(f"{BASE_URL}/step", json=payload)
    return response.json()


# -----------------------------
# GRADER FUNCTION
# -----------------------------
def compute_score(total_reward):
    if total_reward >= 30:
        return 1.0
    elif total_reward >= 0:
        return 0.5
    else:
        return 0.0


# -----------------------------
# AGENTS
# -----------------------------
def random_agent():
    return random.choice(["retry", "switch_api", "use_cache", "return_error"])


def bad_agent():
    return "retry"


def heuristic_agent(obs):
    status = obs["observation"]["api_status"]
    retry_count = obs["observation"]["retry_count"]

    if status == "failed" and retry_count < 2:
        return "retry"
    elif status == "failed":
        return "switch_api"
    elif status == "slow":
        return "use_cache"
    else:
        return "return_error"


# -----------------------------
# RUN EPISODE
# -----------------------------
def run_episode(agent_name, difficulty="easy"):
    print(f"[START] task={difficulty} env=api_env model={agent_name}")

    obs = reset_env(difficulty)
    total_reward = 0
    step_count = 0
    done = False

    MAX_STEPS = 10  # prevents infinite loops

    while step_count < MAX_STEPS:
        step_count += 1

        try:
            if agent_name == "random":
                action = random_agent()
            elif agent_name == "bad":
                action = bad_agent()
            else:
                action = heuristic_agent(obs)

            result = step_env(action)

            reward = result["reward"]
            done = result["done"]

            total_reward += reward

            print(f"[STEP] step={step_count} action={action} reward={round(reward,2)} done={done} error=None")

            if done:
                success = result["observation"]["api_status"] == "success"
                break

            obs = result

        except Exception as e:
            print(f"[STEP] step={step_count} action=error reward=0 done=True error={str(e)}")
            success = False
            break

        time.sleep(0.2)

    # fallback if never done
    if not done:
        success = False

    # -----------------------------
    # FINAL SCORE
    # -----------------------------
    score = compute_score(total_reward)

    print(f"[END] success={success} steps={step_count} rewards={round(total_reward,2)} score={score}\n")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    for agent in ["random", "bad", "heuristic"]:
        run_episode(agent, difficulty="easy")
        