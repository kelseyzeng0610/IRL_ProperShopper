"""
Baselines (Random, GreedyVector, BehaviorCloning) — drop-in for your IRL stack.

All three baselines consume the SAME state abstraction as your ExpertAgent:
    s = expert_agent.trans(game_state)

Call shape for all policies:
    action = policy.act(expert_agent, game_state, forbidden=[])

BehaviorCloningBaseline:
    bc.fit(demos, expert_agent)  # demos = [(game_state, action), ...]

Env invariants for comparability:
    Launch your env with:  --player_speed=0.25 --file=start-state.txt

If your action convention differs, update DIR_TO_ACTION once below.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# =============================================================
# Action mapping / motion helpers
# =============================================================

# Map (dx, dy) unit steps -> action ids.
# Default convention (edit if your env differs):
#   0: WEST  (0,-1)
#   1: EAST  (0, 1)
#   2: NORTH (-1,0)
#   3: SOUTH ( 1,0)
DIR_TO_ACTION: Dict[Tuple[int, int], int] = {
    (0, -1): 0,
    (0,  1): 1,
    (-1, 0): 2,
    (1,  0): 3,
}
ACTIONS: List[int] = list(DIR_TO_ACTION.values())


def round_quarter(x: float) -> float:
    """Round to nearest 0.25 to align with player_speed=0.25 discretization."""
    return float(np.round(x * 4) / 4)


def sgn_unit(v: float) -> int:
    """Map real value to unit step in {-1, 0, 1}."""
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def random_allowable(action_space: int, forbidden: Iterable[int]) -> int:
    """Uniformly sample an allowed action."""
    forbidden = set(forbidden)
    a = np.random.randint(0, action_space)
    while a in forbidden:
        a = np.random.randint(0, action_space)
    return int(a)


def step_to_action(step: Tuple[int, int]) -> Optional[int]:
    """Translate (dx, dy) to an action id using DIR_TO_ACTION."""
    return DIR_TO_ACTION.get(step)


def desired_step_from_goal(expert_agent, game_state: Dict[str, Any]) -> Tuple[int, int]:
    """
    Compute a 1-step (dx,dy) toward the current goal using the same axis
    abstraction used by ExpertAgent.trans():
      - x-only goals: move along x
      - y-only goals: move along y
      - otherwise: move along axis with larger |delta| (ties -> x)
    """
    obs = game_state["observation"]
    px, py = obs["players"][0]["position"]

    if not getattr(expert_agent, "goals", None):
        return (0, 0)

    g = expert_agent.goals[expert_agent.currentGoalIdx]
    gtype = g.get("type", "")
    gx, gy = g.get("position", (px, py))

    dx = round_quarter(gx - px)
    dy = round_quarter(gy - py)

    # Match your ExpertAgent.trans() goal abstractions:
    X_ONLY = {"WALKWAY", "SHELF_NAV", "EAST_WALKWAY"}
    Y_ONLY = {"AISLE", "LEAVE_COUNTERS", "COUNTER_NAV"}

    if gtype in X_ONLY:
        return (sgn_unit(dx), 0)
    if gtype in Y_ONLY:
        return (0, sgn_unit(dy))

    # 2D: choose the dominant axis
    if abs(dx) >= abs(dy):
        return (sgn_unit(dx), 0)
    return (0, sgn_unit(dy))



# Baseline 1: Random (trivial floor)


class RandomBaseline:
    """Uniform over actions, respecting a forbidden set."""

    def __init__(self, action_space: int):
        self.action_space = action_space

    def act(self, expert_agent, game_state: Dict[str, Any],
            forbidden: List[int] | None = None) -> int:
        return random_allowable(self.action_space, forbidden or [])



# Baseline 2: GreedyVector (axis-aligned toward goal)


class GreedyVectorBaseline:
    """
    Greedy one-step toward current goal using the same axis abstraction as ExpertAgent.trans().
    If the desired move is forbidden/blocked, tries orthogonal alternatives; if none, picks random allowed.
    """

    def __init__(self, action_space: int):
        self.action_space = action_space

    def act(self, expert_agent, game_state: Dict[str, Any],
            forbidden: List[int] | None = None) -> int:
        forbidden = forbidden or []
        step = desired_step_from_goal(expert_agent, game_state)
        a = step_to_action(step)
        if a is not None and a not in forbidden:
            return a

        # Try plausible alternatives if preferred move isn't allowed.
        sx, sy = step
        candidates: List[Tuple[int, int]]
        if sx != 0 and sy == 0:
            candidates = [(0, 1), (0, -1)]          # try along y if x is blocked
        elif sy != 0 and sx == 0:
            candidates = [(1, 0), (-1, 0)]          # try along x if y is blocked
        else:
            candidates = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for st in candidates:
            alt = step_to_action(st)
            if alt is not None and alt not in forbidden:
                return alt

        return random_allowable(self.action_space, forbidden)


# Baseline 3: Behavior Cloning (NumPy-only MLP)


class TinyMLP:
    """1-hidden-layer MLP (tanh + softmax) in NumPy for supervised imitation."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 lr: float = 1e-2, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(n_in, n_hidden) / max(1.0, math.sqrt(n_in))
        self.b1 = np.zeros((n_hidden,), dtype=float)
        self.W2 = rng.randn(n_hidden, n_out) / max(1.0, math.sqrt(n_hidden))
        self.b2 = np.zeros((n_out,), dtype=float)
        self.lr = float(lr)

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def _dtanh(y: np.ndarray) -> np.ndarray:
        return 1.0 - y * y

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_pre = x @ self.W1 + self.b1
        h = self._tanh(h_pre)
        o = h @ self.W2 + self.b2
        p = self._softmax(o)
        return h, o, p

    def step_ce(self, x: np.ndarray, y_idx: int) -> float:
        h, o, p = self.forward(x)
        grad_o = p.copy()
        grad_o[0, y_idx] -= 1.0

        # Update W2/b2
        self.W2 -= self.lr * (h.T @ grad_o)
        self.b2 -= self.lr * grad_o.ravel()

        # Backprop to hidden
        grad_h = (grad_o @ self.W2.T) * self._dtanh(h)

        # Update W1/b1
        self.W1 -= self.lr * (x.T @ grad_h)
        self.b1 -= self.lr * grad_h.ravel()

        # Cross-entropy loss
        loss = -math.log(max(1e-9, p[0, y_idx]))
        return float(loss)

    def predict(self, x: np.ndarray) -> int:
        _, _, p = self.forward(x)
        return int(np.argmax(p, axis=1)[0])


class BehaviorCloningBaseline:
    """
    Supervised imitation baseline:
      - Features derived *directly* from expert_agent.trans(game_state).
      - Adds a goal-type one-hot for better generalization across modes.
      - Trains TinyMLP on (state, action) pairs.
    """

    def __init__(self, action_space: int, hidden: int = 64, lr: float = 1e-2,
                 epochs: int = 5, seed: int = 0):
        self.action_space = int(action_space)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.model: Optional[TinyMLP] = None
        self.goal_types: List[str] = []

    # ---------- featurization ----------
    @staticmethod
    def _as_list(x: Any) -> List[float]:
        """
        Convert ExpertAgent.trans(state) output to numeric features:
          - int (direction) -> one-hot(4)
          - float           -> [value]
          - tuple/list      -> [values...]
        """
        if isinstance(x, (list, tuple)):
            return [float(v) for v in x]
        if isinstance(x, (int, np.integer)):
            onehot = [0.0, 0.0, 0.0, 0.0]
            i = int(x)
            if 0 <= i < 4:
                onehot[i] = 1.0
            return onehot
        return [float(x)]

    def _feat(self, expert_agent, game_state: Dict[str, Any]) -> np.ndarray:
        base = self._as_list(expert_agent.trans(game_state))
        gtype = ""
        if getattr(expert_agent, "goals", None):
            g = expert_agent.goals[expert_agent.currentGoalIdx]
            gtype = g.get("type", "")
        if gtype not in self.goal_types:
            self.goal_types.append(gtype)
        gt_idx = self.goal_types.index(gtype)
        gt_oh = [0.0] * max(1, len(self.goal_types))
        gt_oh[gt_idx] = 1.0
        x = np.asarray([*(base + gt_oh)], dtype=float)  # shape (1, D)
        return x

    # ---------- training ----------
    def fit(self, demos: List[Tuple[Dict[str, Any], int]], expert_agent) -> None:
        if not demos:
            raise ValueError("BehaviorCloningBaseline.fit: demos is empty")
        # Warm goal dictionary & infer input dim
        for gs, _ in demos:
            _ = self._feat(expert_agent, gs)
        x0 = self._feat(expert_agent, demos[0][0])
        self.model = TinyMLP(n_in=x0.shape[1], n_hidden=self.hidden,
                             n_out=self.action_space, lr=self.lr, seed=self.seed)
        rng = random.Random(self.seed)
        for ep in range(self.epochs):
            rng.shuffle(demos)
            losses: List[float] = []
            for gs, a in demos:
                x = self._feat(expert_agent, gs)
                losses.append(self.model.step_ce(x, int(a)))
            print(f"[BC] epoch {ep+1}/{self.epochs} loss={sum(losses)/max(1,len(losses)):.4f}")

    # ---------- policy ----------
    def act(self, expert_agent, game_state: Dict[str, Any],
            forbidden: List[int] | None = None) -> int:
        if self.model is None:
            return random_allowable(self.action_space, forbidden or [])
        a = self.model.predict(self._feat(expert_agent, game_state))
        if forbidden and a in set(forbidden):
            return random_allowable(self.action_space, forbidden)
        return a



# Optional: tiny eval harness & synthetic smoke test


class EnvClient:
    """
    Minimal wrapper for a Gym-like env (or swap with your socket client).
    Must implement:
      reset() -> (obs, info)
      step(a) -> (obs, r, done, trunc, info)
    """

    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, a: int):
        return self.env.step(a)


def evaluate_policy(policy, env_client: EnvClient, expert_agent,
                    episodes: int = 5,
                    forbids_fn: Optional[callable] = None) -> Dict[str, float]:
    rewards: List[float] = []
    lengths: List[int] = []
    for _ in range(episodes):
        obs, info = env_client.reset()
        done = False
        ep_r = 0.0
        t = 0
        while not done:
            forb = forbids_fn(obs) if forbids_fn else []
            a = policy.act(expert_agent, obs, forbidden=forb)
            obs, r, done, trunc, info = env_client.step(a)
            ep_r += float(r)
            t += 1
            if trunc:
                break
        rewards.append(ep_r)
        lengths.append(t)
    return {"avg_reward": float(np.mean(rewards)), "avg_len": float(np.mean(lengths))}


# ---- A tiny grid and "expert" for local sanity checks (optional) ----

@dataclass
class SyntheticGoal:
    type: str
    position: Tuple[float, float]


class SyntheticExpert:
    """Stand-in that matches ExpertAgent attributes used by baselines."""

    def __init__(self, goals: List[Dict[str, Any]]):
        self.goals = goals
        self.currentGoalIdx = 0

    def trans(self, game_state: Dict[str, Any]):
        # Emulates ExpertAgent.trans for two common goal modes
        px, py = game_state["observation"]["players"][0]["position"]
        g = self.goals[self.currentGoalIdx]
        gtype = g.get("type", "")
        gx, gy = g.get("position", (px, py))
        dx = round_quarter(gx - px)
        dy = round_quarter(gy - py)
        if gtype in {"WALKWAY", "SHELF_NAV", "EAST_WALKWAY"}:
            return dx  # x-only abstraction
        if gtype in {"AISLE", "LEAVE_COUNTERS", "COUNTER_NAV"}:
            return dy  # y-only abstraction
        return (dx, dy)


class DummyGridEnv:
    """
    Minimal 9x9 grid with a mid-row wall (two gaps). Player at (0,0). Reward +1 on goal,
    small step cost otherwise. This is only for smoke-testing the baselines.
    """

    def __init__(self, goal: Tuple[int, int] = (8, 8), seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.H = self.W = 9
        self.grid = np.zeros((self.H, self.W), dtype=int)
        # place a wall across row 4 with two openings
        for y in range(self.W):
            self.grid[4, y] = 1
        self.grid[4, 2] = 0
        self.grid[4, 6] = 0
        self.goal = goal
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        self.t = 0
        obs = {"observation": {"players": [{"position": (float(self.pos[0]), float(self.pos[1]))}]}}
        return obs, {}

    def _move(self, dx: int, dy: int):
        nx = int(np.clip(self.pos[0] + dx, 0, self.H - 1))
        ny = int(np.clip(self.pos[1] + dy, 0, self.W - 1))
        if self.grid[nx, ny] == 0:
            self.pos = (nx, ny)

    def step(self, a: int):
        delta = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}.get(int(a), (0, 0))
        self._move(*delta)
        r = -0.02
        done = False
        if self.pos == self.goal:
            r = 1.0
            done = True
        self.t += 1
        if self.t >= 200:
            done = True
        obs = {"observation": {"players": [{"position": (float(self.pos[0]), float(self.pos[1]))}]}}
        return obs, r, done, False, {}


def collect_demos(policy, env: DummyGridEnv, expert_agent: SyntheticExpert,
                  episodes: int = 8) -> List[Tuple[Dict[str, Any], int]]:
    demos: List[Tuple[Dict[str, Any], int]] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            a = policy.act(expert_agent, obs, forbidden=[])
            demos.append((obs, a))
            obs, r, done, trunc, info = env.step(a)
            if trunc:
                break
    return demos



# Main: quick smoke test (optional)

if __name__ == "__main__":
    # Synthetic goal that mimics an x-only (SHELF_NAV) segment
    goals = [{"type": "SHELF_NAV", "position": (8.0, 0.0)}]
    expert = SyntheticExpert(goals)
    env = DummyGridEnv(goal=(8, 0))
    client = EnvClient(env)

    action_space = 4

    # 1) Random baseline
    rand_pi = RandomBaseline(action_space)
    m_rand = evaluate_policy(rand_pi, client, expert, episodes=5)
    print("Random =>", m_rand)

    # 2) GreedyVector baseline
    greedy_pi = GreedyVectorBaseline(action_space)
    m_greedy = evaluate_policy(greedy_pi, client, expert, episodes=5)
    print("GreedyVector =>", m_greedy)

    # 3) Behavior Cloning baseline — train on Greedy demos, evaluate
    demos = collect_demos(greedy_pi, env, expert, episodes=6)
    bc_pi = BehaviorCloningBaseline(action_space, hidden=64, lr=5e-3, epochs=5)
    bc_pi.fit(demos, expert)
    m_bc = evaluate_policy(bc_pi, client, expert, episodes=5)
    print("BehaviorCloning =>", m_bc)





