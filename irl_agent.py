import numpy as np
import matplotlib.pyplot as plt
from max_entropy_irl import MaxEntropyIRL
import socket
from utils import recv_socket_data
import json
import pickle

INT_TO_DIRECTION_STR = {
    0: "NORTH",
    1: "SOUTH", 
    2: "EAST",
    3: "WEST"
}

def roundToPointTwoFive(number):
    return np.round(number * 4) / 4

def inTargetArea(playerX, playerY):
    inX = playerX >= 5.5 and playerX <= 6
    inY = playerY >= 11 and playerY <= 11.5
    in_area_flag = int(inX and inY)
    return in_area_flag

def phi(gameState):
    # this should return our internal state representation from a game state object
    [playerX, playerY] = gameState["observation"]["players"][0]["position"]
    direction = gameState['observation']['players'][0]['direction']
    direction_one_hot = [0] * 4
    direction_one_hot[direction] = 1
    in_area_flag = inTargetArea(playerX, playerY)

    return tuple([roundToPointTwoFive(playerX), roundToPointTwoFive(playerY)] + direction_one_hot + [in_area_flag])

def gameOver(phi_state):
    x= phi_state[-1]
    return x

def plot_grad_norms(grad_norms):
    plt.figure(figsize=(7,5))
    plt.plot(range(len(grad_norms)), grad_norms)
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.grid(True)
    plt.show()

# in the store, positive y direction is down and positive x direction is to the right
def getDelta(action):
    delta = 0.25
    if action == 0:
        return 0, -1 * delta
    elif action == 1:
        return 0, delta
    elif action == 2:
        return delta, 0
    elif action == 3:
        return -1 * delta, 0
    else:
        raise Exception("Can't compute delta for undefined direction")

# This is a very simple model of how the environment changes with actions, that doesn't take into account 
# the obstacles or layout of the store
def updateGameState(phi_state, action):
    direction_one_hot = [0] * 4
    direction_one_hot[action] = 1
    in_area_flag = phi_state[-1]
    direction_phi = phi_state[2:6]
    if direction_phi != direction_one_hot:
        # if the action is not in the same direction as the current state, rotate.
        return [phi_state[0], phi_state[1]] + direction_one_hot + [in_area_flag] # same position, new direction
    else:
        # else move by step size in the direction of action
        deltaX, deltaY = getDelta(action)
        newX, newY = phi_state[0] + deltaX, phi_state[1] + deltaY
        new_in_area = inTargetArea(newX, newY)
        new_phi = [newX, newY] + direction_one_hot + [new_in_area]
        return new_phi



def loadTrajectories(fileName):
    data = []
    with open(fileName, 'rb') as f:
        data = pickle.load(f)

    for trajectory in data:
        for step in trajectory:
            s, a, sprime = step[0], step[1], step[2]
    return data

def jsonSerializeTrajectoryStep(trajectory):
    # Trajectory is (s, a, s')
    s, a, sprime = trajectory[0], trajectory[1], trajectory[2]

    jsonS = [float(s[0]), float(s[1]), int(s[2]), int(s[3]), int(s[4]), int(s[5]), int(s[6])]
    jsonSPrime = [float(sprime[0]), float(sprime[1]), int(sprime[2]), int(sprime[3]), int(sprime[4]), int(sprime[5]), int(sprime[6])]
    jsonAction = int(a)
    return [
        jsonS,
        jsonAction,
        jsonSPrime,
    ]

def writeSampleTrajectory(trajectory, fileName="generated-actions.json"):
    with open(fileName, "w") as f:
        json.dump([jsonSerializeTrajectoryStep(step) for step in trajectory], f, indent=2)

def writeLearnedTheta(theta, fileName="theta-out.pkl"):
    with open(fileName, 'wb') as f:
        pickle.dump(theta, f)

def loadLearnedTheta(fileName="theta-out.pkl"):
    with open(fileName, 'rb') as f:
        theta = pickle.load(f)
        return theta
    
def plotGeneratedTrajectory(sampleTrajectory, expertTrajectories):
    sampleX, sampleY = [], []
    for step in sampleTrajectory:
        initialState = step[0]
        x, y, = initialState[0], initialState[1]
        sampleX.append(x)
        sampleY.append(y)
    plt.plot(sampleX, sampleY, "o-", color='blue', label='Learned')
    
    for expertTrajectory in expertTrajectories:
        expertX, expertY = [], []
        for step in expertTrajectory:
            state = step[0]
            x, y, = state[0], state[1]
            expertX.append(x)
            expertY.append(y)
        plt.plot(expertX, expertY, 's-', color='red', label='Expert')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()


ACTIONS = ['NORTH', 'SOUTH', 'EAST', 'WEST']

learningMode = False

if __name__ == "__main__":
    action_space = len(ACTIONS)
    initial_phi_state = [1.25, 15.5, 0, 0, 1, 0, 0]
    expertTrajectories = loadTrajectories("trajectories.pkl") # load the pkl file

    if learningMode:
        np.random.seed(42)
        theta_random = np.random.uniform(low=0.0, high=0.1, size=len(initial_phi_state))

        learner = MaxEntropyIRL(
            theta=theta_random,
            actions=np.arange(len(ACTIONS)),
            probeNextState=updateGameState,
            initialState=initial_phi_state,
            gameOver=gameOver,
            phi=phi,
        )
        theta_hat, grad_norms, theta_iterations = learner.learn(expertTrajectories, num_iterations=500, alpha=0.025, num_samples=50)
        print("\n\n Predicted Theta:\n\n", theta_hat)
        writeLearnedTheta(theta_hat)
        plot_grad_norms(grad_norms)
    else:
        learnedTheta = loadLearnedTheta()
        learner = MaxEntropyIRL(
            theta=learnedTheta,
            actions=np.arange(len(ACTIONS)),
            probeNextState=updateGameState,
            initialState=initial_phi_state,
            gameOver=gameOver,
            phi=phi,
        )
        sampleTrajectory = learner.stochastic_trajectory(maxLength=100)
        writeSampleTrajectory(sampleTrajectory)
        plotGeneratedTrajectory(sampleTrajectory, expertTrajectories[:1])

