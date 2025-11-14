# ============================================================
# Hardcoded Expert Trajectories + Simple Q-learning Imitation
# No environment, no coordinates, no matrix math.
# ============================================================

import json
import random

# ---------------------------------------
# Actions and indices
# ---------------------------------------
ACTIONS = ["N", "S", "E", "W"]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ---------------------------------------
# 7 hardcoded expert trajectories
# (action-only; no positions)
# ---------------------------------------
traj1 = ["N", "N", "E", "E", "S", "S", "W"]
traj2 = ["E", "E", "E", "W", "W", "E"]
traj3 = ["N", "N", "N", "E", "E", "E"]
traj4 = ["E", "E", "S", "S", "E", "N", "N"]
traj5 = ["W", "W", "W", "E", "E", "S"]
traj6 = ["N", "E", "N", "E", "S", "W"]
traj7 = ["S", "S", "E", "E", "N", "N"]

expert_trajs = [traj1, traj2, traj3, traj4, traj5, traj6, traj7]

# ---------------------------------------
# One-hot features for actions
# ---------------------------------------
def action_to_feature(a):
    idx = ACTION_IDX[a]
    feat = [0] * len(ACTIONS)
    feat[idx] = 1
    return feat

phi_trajs = [[action_to_feature(a) for a in traj] for traj in expert_trajs]

# ---------------------------------------
# Simple tabular Q-learning agent
# State = time step index (0..max_len-1)
# ---------------------------------------
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.5, gamma=0.9, epsilon=0.2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table: list of lists, Q[s][a]
        self.Q = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        q_vals = self.Q[state]
        return max(range(self.n_actions), key=lambda a: q_vals[a])

    def update(self, s, a, r, s_next, done):
        old = self.Q[s][a]
        if done:
            target = r
        else:
            best_next = max(self.Q[s_next])
            target = r + self.gamma * best_next
        self.Q[s][a] = (1 - self.alpha) * old + self.alpha * target

# ---------------------------------------
# Train Q-learning to imitate trajectories
# ---------------------------------------
def train_q_from_trajectories(trajs, episodes=500):
    max_len = max(len(t) for t in trajs)
    agent = QLearningAgent(n_states=max_len, n_actions=len(ACTIONS))
    for _ in range(episodes):
        for traj in trajs:
            for t in range(len(traj)):
                s = t
                expert_action = traj[t]
                expert_idx = ACTION_IDX[expert_action]

                a = agent.choose_action(s)
                # reward +1 if matches expert, else -1
                r = 1 if a == expert_idx else -1
                done = (t == len(traj) - 1)
                s_next = s if done else s + 1

                agent.update(s, a, r, s_next, done)
    return agent.Q

# ---------------------------------------
# Save everything
# ---------------------------------------
def save_all():
    # raw expert trajectories (actions)
    with open("expert_trajectories.json", "w") as f:
        json.dump(expert_trajs, f, indent=2)

    # one-hot features for IRL-style use
    with open("phi_trajectories.json", "w") as f:
        json.dump(phi_trajs, f, indent=2)

    # Q-table learned from imitation
    qtable = train_q_from_trajectories(expert_trajs)
    with open("qtable.json", "w") as f:
        json.dump(qtable, f, indent=2)

    print("Saved expert_trajectories.json, phi_trajectories.json, qtable.json")

# ---------------------------------------
# Main
# ---------------------------------------
if __name__ == "__main__":
    save_all()