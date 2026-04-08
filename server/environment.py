import random
import uuid

from openenv.core.env_server import Environment
from models import APIAction, APIObservation, APIState


class APIEnvironment(Environment):

    def __init__(self):
        self._state = APIState()
        self.current_status = "success"
        self.latency = 100
        self.cost = 0.01
        self.system_load = "low"
        self.difficulty = "easy"
        self.last_action = None

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self, seed=None, episode_id=None, difficulty="easy", **kwargs):

        self.difficulty = difficulty

        self._state = APIState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_api="A",
            retry_count=0,
            done=False
        )

        # Track total cost for scoring
        self._state.total_cost = 0.0

        # Randomize load
        self.system_load = random.choice(["low", "medium", "high"])

        status, latency, cost = self.simulate_api()

        self.current_status = status
        self.latency = latency
        self.cost = cost

        return APIObservation(
            done=False,
            reward=0.0,
            api_status=self.current_status,
            latency=self.latency,
            retry_count=self._state.retry_count,
            api_cost=self.cost,
            system_load=self.system_load,
            message=f"Task: {self.difficulty} | Environment reset"
        )

    # -----------------------------
    # SIMULATION
    # -----------------------------
    def simulate_api(self):

        if self.difficulty == "easy":
            fail_prob = 0.3
        elif self.difficulty == "medium":
            fail_prob = 0.5
        else:
            fail_prob = 0.7

        if self.system_load == "high":
            fail_prob += 0.1

        if self._state.current_api == "A":
            fail_prob += 0.1
        else:
            fail_prob -= 0.1

        fail_prob = max(0.0, min(1.0, fail_prob))

        rand = random.random()

        # API latency difference
        if self._state.current_api == "A":
            latency_range = (150, 400)
        else:
            latency_range = (50, 200)

        if rand < fail_prob:
            return "failed", random.uniform(*latency_range), random.uniform(0.01, 0.1)
        elif rand < fail_prob + 0.1:
            return "slow", random.uniform(200, 500), random.uniform(0.01, 0.1)
        else:
            return "success", random.uniform(50, 200), random.uniform(0.01, 0.1)

    # -----------------------------
    # TASK SCORING (0 → 1)
    # -----------------------------
    def compute_task_score(self):

        # Lower retries + cost + latency = better score
        score = 1.0

        score -= 0.1 * self._state.retry_count
        score -= 0.05 * self._state.total_cost
        score -= 0.001 * self.latency

        if self.current_status != "success":
            score -= 0.3

        return max(min(score, 1.0), 0.0)

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action: APIAction, timeout_s=None, **kwargs):

        self._state.step_count += 1

        prev_status = self.current_status
        prev_latency = self.latency

        # ---------------- ACTIONS ----------------
        if action.action == "accept":
            pass

        elif action.action == "retry":
            self._state.retry_count += 1
            self.current_status, self.latency, self.cost = self.simulate_api()

        elif action.action == "switch_api":
            self._state.current_api = "B" if self._state.current_api == "A" else "A"
            self.system_load = "medium"
            self._state.retry_count = max(0, self._state.retry_count - 1)
            self.current_status, self.latency, self.cost = self.simulate_api()

        elif action.action == "use_cache":
            success_prob = 0.9 if self.system_load != "high" else 0.6
            self.current_status = "success" if random.random() < success_prob else "failed"
            self.latency = random.uniform(10, 50)
            self.cost = 0.0

        elif action.action == "return_error":
            self.current_status = "failed"
            self.latency = 0
            self.cost = 0.0

        else:
            self.current_status, self.latency, self.cost = self.simulate_api()

        # Track total cost
        self._state.total_cost += self.cost

        # ---------------- REWARD (RAW) ----------------
        reward = 0

        reward += 8 if self.current_status == "success" else -8
        reward -= 0.02 * self.latency
        reward -= 5 * self.cost
        reward -= 2 * self._state.retry_count

        if prev_status == "success" and prev_latency < 120:
            reward += 5 if action.action == "accept" else -5

        if prev_status == "failed":
            reward += 4 if action.action in ["retry", "switch_api"] else -4

        if prev_status == "slow":
            reward += 3 if action.action in ["use_cache", "switch_api"] else -2

        if self.last_action == action.action:
            reward -= 3

        if self.current_status == "success" and self.latency < 80 and self.cost < 0.02:
            reward += 3

        # Normalize reward to 0–1
        reward = max(min((reward + 10) / 20, 1), 0)

        self.last_action = action.action

        # ---------------- DONE ----------------
        done = self._state.step_count >= 10 or self._state.retry_count >= 5

        # ---------------- TASK SCORE ----------------
        score = self.compute_task_score()

        return APIObservation(
            done=done,
            reward=score,  # ✅ FINAL SCORE (0–1)
            api_status=self.current_status,
            latency=self.latency,
            retry_count=self._state.retry_count,
            api_cost=self.cost,
            system_load=self.system_load,
            message=f"Task: {self.difficulty} | Score: {round(score, 2)} | Action: {action.action}"
        )

    # -----------------------------
    # STATE (REQUIRED)
    # -----------------------------
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value