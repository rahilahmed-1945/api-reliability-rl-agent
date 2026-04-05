import random
import uuid

from openenv.core.env_server import Environment
from models import APIAction, APIObservation, APIState


class APIEnvironment(Environment):

    def __init__(self):
        self.state = APIState()
        self.current_status = "success"
        self.latency = 100
        self.cost = 0.01
        self.system_load = "low"

        self.difficulty = "easy"
        self.last_action = None

    def reset(self, seed=None, episode_id=None, difficulty="easy", **kwargs):
        random.seed(seed)

        self.difficulty = difficulty

        self.state = APIState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_api="A",
            retry_count=0,
            done=False
        )

        self.system_load = random.choice(["low", "medium", "high"])

        self.current_status, self.latency, self.cost = self.simulate_api()

        return APIObservation(
            done=False,
            reward=None,
            api_status=self.current_status,
            latency=self.latency,
            retry_count=self.state.retry_count,
            api_cost=self.cost,
            system_load=self.system_load,
            message="Environment reset"
        )

    def simulate_api(self):
        if self.difficulty == "easy":
            fail_prob = 0.3
        elif self.difficulty == "medium":
            fail_prob = 0.5
        else:
            fail_prob = 0.7

        if self.system_load == "high":
            fail_prob += 0.1

        rand = random.random()

        if rand < fail_prob:
            status = "failed"
            latency = random.uniform(300, 800)
        elif rand < fail_prob + 0.1:
            status = "slow"
            latency = random.uniform(200, 500)
        else:
            status = "success"
            latency = random.uniform(50, 200)

        cost = random.uniform(0.01, 0.1)

        return status, latency, cost

    def step(self, action: APIAction, timeout_s=None, **kwargs):
        self.state.step_count += 1

        # ACTION EFFECTS
        if action.action == "retry":
            self.state.retry_count += 1

        elif action.action == "switch_api":
            self.state.current_api = "B" if self.state.current_api == "A" else "A"

        elif action.action == "use_cache":
            if random.random() < 0.7:
                self.current_status = "success"
            else:
                self.current_status = "failed"

            self.latency = random.uniform(10, 50)
            self.cost = 0.0

        elif action.action == "return_error":
            self.current_status = "failed"

        # SIMULATION
        self.current_status, self.latency, self.cost = self.simulate_api()

        # REWARD
        reward = 0

        if self.last_action == action.action:
            reward -= 2
        self.last_action = action.action

        if self.current_status == "success":
            reward += 10

        reward -= 0.01 * self.latency
        reward -= 2 * self.state.retry_count
        reward -= 3 * self.cost

        if self.current_status == "failed":
            reward -= 8

        # DONE CONDITION (IMPORTANT CHANGE)
        done = (
            self.state.step_count >= 5
            or self.state.retry_count >= 5
        )

        return APIObservation(
            done=done,
            reward=reward,
            api_status=self.current_status,
            latency=self.latency,
            retry_count=self.state.retry_count,
            api_cost=self.cost,
            system_load=self.system_load,
            message=f"Action taken: {action.action}"
        )

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value