import numpy as np
import matplotlib.pyplot as plt

# Basic version of max entropy IRL learning on a very simple problem that hopefully we can extend to the shopping problem

def softmax(vect):
    e_x = np.exp(vect - np.max(vect))
    return e_x / np.sum(e_x)

def feature_count(trajectory):
    return np.sum(np.array(trajectory), axis=0) / len(trajectory)

def average_feature_counts(trajectories):
    total = sum(feature_count(traj) for traj in trajectories)
    return total / len(trajectories)

class MaxEntropyIRL:
    def __init__(self, theta, actions, getNextState):
        self.theta = theta
        self.actions = actions
        self.getNextState = getNextState
        self.maxTrajectoryLength = 25

    def reward(self, state):
        return np.dot(self.theta, state)

    def stochastic_trajectory(self):
        currentState = (0,0)
        trajectory = [currentState]
        while currentState != (4,4) and len(trajectory) < self.maxTrajectoryLength:
            q_values = []
            next_states = []
            for a in self.actions:
                ns = self.getNextState(currentState, a)
                if ns == currentState:
                    continue
                next_states.append(ns)
                q_values.append(self.reward(ns))
            prob_values = softmax(np.array(q_values))

            randomIdx = np.random.choice(len(next_states), p=prob_values)
            currentState = next_states[randomIdx]
            trajectory.append(currentState)

        return trajectory


    def learn(self, expert_trajectories, num_iterations=200, num_samples=50, alpha = 0.01):
        muhat_expert = average_feature_counts(expert_trajectories)
        grad_norms = []
        for iteration in range(num_iterations):
            # First sample some trajectories according to our current reward estimate (theta)
            sample_trajectories = [self.stochastic_trajectory() for m in range(num_samples)]
            # Get the average feature counts according to our model
            muhat = average_feature_counts(sample_trajectories)
            # Update step: expert minus model
            grad = muhat_expert - muhat
            self.theta += alpha * grad
            grad_norms.append(np.linalg.norm(grad))

        return self.theta, grad_norms



# Sample problem: imagine a simple 5x5 grid where you start at 0,0 and get reward for arriving at 4,4
# Define our feature representation as (x,y)
ACTIONS = ['N', 'S', 'E', 'W']
ACTIONS_DELTA = {
    'N': (0,1),
    'S': (0,-1),
    'E': (1,0),
    'W': (-1,0)
}
def getNextState(state, action):
    delta = ACTIONS_DELTA[action]
    newX = max(0, min(4, state[0] + delta[0]))
    newY = max(0, min(4, state[1] + delta[1]))
    return (newX, newY)

# this should decrease if our agent is learning
def plot_grad_norms(grad_norms):
    plt.figure(figsize=(7,5))
    plt.plot(range(len(grad_norms)), grad_norms)
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    expertTrajectories = [
        [(0,0), (1,0), (2,0), (3,0), (4,0), (4,1), (4,2), (4,3), (4,4)],
        [(0,0), (0,1), (0,2), (0,3), (0,4), (1,4), (2,4), (3,4), (4,4)],
        [(0,0), (1,0), (1,1), (2,1), (2,2), (3,2), (3,3), (4,3), (4,4)]
    ]

    np.random.seed(42)
    theta_random = np.random.uniform(low=0.0, high=0.1, size=2)

    learner = MaxEntropyIRL(theta=theta_random, actions=ACTIONS, getNextState=getNextState)
    theta_hat, grad_norms = learner.learn(expertTrajectories)
    print(theta_hat)
    sampleTrajectory = learner.stochastic_trajectory()
    print(sampleTrajectory)
    plot_grad_norms(grad_norms)
