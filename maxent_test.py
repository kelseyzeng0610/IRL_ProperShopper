import numpy as np
import matplotlib.pyplot as plt
from max_entropy_irl import MaxEntropyIRL

# Basic version of max entropy IRL learning on a very simple problem that hopefully we can extend to the shopping problem
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
