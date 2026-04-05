---
title: API Reliability RL Environment
emoji: 🚀
colorFrom: purple
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

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

## 🧠 Environment Dynamics

* API A → cheaper but less reliable
* API B → more reliable but higher cost
* Failures persist across steps (**temporal memory**)
* Repeated retries increase failure probability (**cascading effect**)
* Actions influence future system behavior

---

## 🏆 Reward Function

* +10 → successful response
* −0.01 × latency
* −scaled retry penalty (non-linear with retries)
* −3 × api_cost
* −8 → failure
* −1 → cache overuse penalty
* −2 → repeated action penalty

---

## 🧪 Tasks (Difficulty Levels)

| Task   | Description                      |
| ------ | -------------------------------- |
| Easy   | Low failure probability (~30%)   |
| Medium | Moderate failures (~50%)         |
| Hard   | High failure + cascading effects |

---

## 🤖 Baseline Agents

| Agent     | Behavior             |
| --------- | -------------------- |
| Random    | Random actions       |
| Bad       | Always retries       |
| Heuristic | Rule-based decisions |

---

## 📊 Evaluation

* Rewards are computed per step
* Total reward is converted into a **score between 0.0 and 1.0**
* Score reflects overall performance

---

## 🧪 Inference Script

```bash
python inference.py
```

---

## 🌐 API Usage

### Reset Environment

POST /reset

### Take Action

POST /step

### Get State

GET /state

---

## 🤖 AI Integration

The system uses an OpenAI-compatible API to generate **real-time explanations** for decisions, improving interpretability.

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
* OpenAI-compatible API

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
* State-aware stochastic environment
* Multi-objective reward optimization
* LLM-based interpretability
* Fully deployable RL environment

---

## 👥 Team

* PALETI SAI TARUN — [saitarunpaleti@gmail.com](mailto:saitarunpaleti@gmail.com)
* Rahil Ahmed — [rahilahmed1305@gmail.com](mailto:rahilahmed1305@gmail.com)
* Ganesh Rayapati — [nehapavanr@gmail.com](mailto:nehapavanr@gmail.com)
