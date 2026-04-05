# 🚀 Cost-Aware API Reliability RL Environment

## 🧠 Overview

This project implements a **real-world reinforcement learning (RL) environment** that simulates API reliability challenges in backend systems.

Agents must make intelligent decisions under uncertainty to balance:

* ✅ Success rate
* ⏱️ Latency
* 💰 Cost

This models real-world scenarios like microservice failures, API retries, and fallback strategies.

---

## 🎯 Objective

Enable agents to learn optimal strategies for handling unreliable APIs using the OpenEnv framework.

---

## 🧩 State Space (Observation)

| Feature       | Description                 |
| ------------- | --------------------------- |
| `api_status`  | success / slow / failed     |
| `latency`     | Response time in ms         |
| `retry_count` | Number of retries performed |
| `api_cost`    | Cost of API usage           |
| `system_load` | low / medium / high         |

---

## ⚡ Action Space

| Action         | Description                       |
| -------------- | --------------------------------- |
| `retry`        | Retry the same API                |
| `switch_api`   | Switch to backup API              |
| `use_cache`    | Use cached response (fast, cheap) |
| `return_error` | Stop and return failure           |

---

## 🏆 Reward Function

The reward function models real-world trade-offs:

* +10 → successful response
* −0.01 × latency
* −2 × retry_count
* −3 × api_cost
* −8 → failure
* −2 → repeated same action penalty

👉 Encourages:

* efficient responses
* minimal retries
* cost optimization

---

## 🧪 Tasks (Difficulty Levels)

| Task   | Description                         |
| ------ | ----------------------------------- |
| Easy   | Low failure probability (~30%)      |
| Medium | Moderate failures (~50%)            |
| Hard   | High failure + system stress (~70%) |

---

## 🤖 Baseline Agents

| Agent     | Behavior                   |
| --------- | -------------------------- |
| Random    | Random actions             |
| Bad       | Always retries             |
| Heuristic | Smart rule-based decisions |

---

## 📊 Evaluation

* Rewards are computed per step
* Total reward is converted into a **score between 0.0 and 1.0**
* Score reflects overall performance

---

## 🧪 Inference Script

Run baseline evaluation:

```bash
python inference.py
```

Outputs structured logs:

```
[START] ...
[STEP] ...
[END] ...
```

---

## 🌐 API Usage

### Reset Environment

```
POST /reset
```

### Take Action

```
POST /step
```

### Get State

```
GET /state
```

---

## 🚀 Live Demo

👉 https://rahilahmed1945-api-reliability-rl-env.hf.space

---

## 🛠️ Tech Stack

* OpenEnv
* FastAPI
* Gradio
* Docker
* Hugging Face Spaces

---

## 📦 Local Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --reload
python app.py
```

---

## 🐳 Docker Setup

```bash
docker build -t api-env .
docker run -p 8000:8000 api-env
```

---

## 🧠 OpenEnv Compliance

* ✅ Typed Action / Observation / State models
* ✅ step(), reset(), state() implemented
* ✅ openenv.yaml included
* ✅ Dockerized deployment
* ✅ HF Space live

---

## 🎯 Key Highlights

* Real-world API reliability simulation
* Multi-objective reward design
* Deterministic evaluation via scoring
* Fully deployable RL environment

---

## 👥 Team

* PALETI SAI TARUN - saitarunpaleti@gmail.com 

* Rahil Ahmed - rahilahmed1305@gmail.com

* Ganesh Rayapati - nehapavanr@gmail.com