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


    # BASIC TEST 1:
    # starting from 0,0, navigate to 4,4
    def basicTest1():
        expertTrajectories = [
            [(0,0), (1,0), (2,0), (3,0), (4,0), (4,1), (4,2), (4,3), (4,4)],
            [(0,0), (0,1), (0,2), (0,3), (0,4), (1,4), (2,4), (3,4), (4,4)],
            [(0,0), (1,0), (1,1), (2,1), (2,2), (3,2), (3,3), (4,3), (4,4)]
        ]
        def phi(state):
            return np.asarray(state) # for now phi is just x,y
        def gameOver(state):
            return np.allclose(state, np.asarray([4,4]))

        np.random.seed(42)
        theta_random = np.random.uniform(low=0.0, high=0.1, size=2)

        learner = MaxEntropyIRL(
            theta=theta_random, 
            actions=ACTIONS, 
            probeNextState=getNextState,
            initialState=(0,0),
            gameOver=gameOver,
            phi=phi,
        )
        theta_hat, _ = learner.learn(expertTrajectories)
        assert theta_hat[0] > 0
        assert theta_hat[1] > 0

        print("Pass basic test 1: both theta values are positive")
        sampleTrajectory = learner.stochastic_trajectory()
        x, y = [step[0] for step in sampleTrajectory], [step[1] for step in sampleTrajectory]
        plt.plot(x, y, 'o-', color='blue', label='Sampled trajectory')
        plt.grid(True)
        plt.legend()
        plt.show()

    # BASIC TEST 2:
    # Navigate from 4,0 to 2,4 - this requires us to change phi
    def basicTest2():
        expertTrajectories = [
            [(4,0), (3,0), (2,0), (2,1), (2,2), (2,3), (2,4)],
            [(4,0), (4,1), (4,2), (4,3), (4,4), (3,4), (2,4)],
            [(4,0), (3,0), (3,1), (3,2), (3,3), (2,3), (2,4)],
        ]
       
        def getphi(goal):
            def phi(state):
                delta = [goal[0] - state[0], goal[1] - state[1]]
                return np.asarray(delta) # now phi is a vector pointing to the goal
            return phi
        def gameOver(state):
            return np.allclose(state, np.asarray([2,4]))

        np.random.seed(42)
        theta_random = np.random.uniform(low=0.0, high=0.1, size=2)

        learner = MaxEntropyIRL(
            theta=theta_random, 
            actions=ACTIONS, 
            probeNextState=getNextState,
            initialState=(4,0),
            gameOver=gameOver,
            phi=getphi((2,4)),
        )
        theta_hat, _ = learner.learn(expertTrajectories)

        # Now that phi is a vector, if theta is negative then that means we are learning to make the vector 0 -> that means we are approaching the goal location
        assert theta_hat[0] < 0
        assert theta_hat[1] < 0

        print("Pass basic test 2: both theta values are negative")
        sampleTrajectory = learner.stochastic_trajectory(maxLength=100)
        x, y = [step[0] for step in sampleTrajectory], [step[1] for step in sampleTrajectory]
        plt.plot(x, y, 'o-', color='blue', label='Sampled trajectory')
        plt.grid(True)
        plt.legend()
        plt.show() 

    basicTest1()
    basicTest2()

