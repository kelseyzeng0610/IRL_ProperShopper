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

# Instead of using the game socket to find out what happens when executing action a in state s, 
# we can try to guess it ourselves - when we load our trajectories we have a bunch of state-action-newstate 
# pairs. so we can build a cache of state-action : newstate. So when we later want to know what happens when
# we execute action a in state s, we can check if we've actually done that already in the expert trajectory.
# if so, great we can just use the same output state that the expert saw. 
# If we haven't seen that, then we need to figure something else out. 
def updateGameState(phi_state, action):
    # phi_tuple = tuple(phi_state)
    # if (phi_tuple, action) in cache:
    #     # randomly sample an option
    #     randomIdx = np.random.choice(len(cache[(phi_tuple, action)]))
    #     choice = cache[(phi_tuple, action)][randomIdx]
    #     return choice
    # # We have not observed action a in state s, so we can't just lookup the next state.
    # elif phi_state[2] != action:
    direction_one_hot = [0] * 4
    direction_one_hot[action] = 1
    in_area_flag = phi_state[-1]
    direction_phi = phi_state[2:6]
    if direction_phi != direction_one_hot:
        # We do know that if the action is not in the same direction as the current state, the agent will rotate.
        return [phi_state[0], phi_state[1]] + direction_one_hot + [in_area_flag] # same position, new direction
    else:
        # The thing we don't know is what happens if the action is in the same direction as we're facing - we will move in that direction but by how much?
        deltaX, deltaY = getDelta(action)
        newX, newY = phi_state[0] + deltaX, phi_state[1] + deltaY
        new_in_area = inTargetArea(newX, newY)
        new_phi = [newX, newY] + direction_one_hot + [new_in_area]
        return new_phi

# def updateGameState(sock_game, currentState, action):
#     actionStr = INT_TO_DIRECTION_STR[action]
#     sock_game.send(str.encode("0 " + actionStr))
#     next_state = recv_socket_data(sock_game)
#     next_state = json.loads(next_state)

#     sock_game.send(str.encode("0 REVERT"))
#     reverted_state = recv_socket_data(sock_game)
#     reverted_state = json.loads(reverted_state)

#     return phi(next_state)



def loadTrajectories(fileName):
    data = []
    with open(fileName, 'rb') as f:
        data = pickle.load(f)

    # build a cache of observed state-action pairs to lookup new states later
    cache = {}
    for trajectory in data:
        for step in trajectory:
            s, a, sprime = step[0], step[1], step[2]
            # if (s, a) not in cache:
            #     cache[(s, a)] = []
            # if sprime not in cache[(s, a)]:
            #     cache[(s, a)].append(sprime)
            cache[(s,a)] = sprime
    return data, cache


def jsonSerializeTrajectory(trajectory):
    # Trajectory is (s, a, s')
    s, a, sprime = trajectory[0], trajectory[1], trajectory[2]

    # s and s' are (x, y, direction) ie (float, float, int) [float(t[0]), float(t[1]), int(t[2])]
    jsonS = [float(s[0]), float(s[1]), int(s[2])]
    jsonSPrime = [float(sprime[0]), float(sprime[1]), int(sprime[2])]
    jsonAction = int(a)
    return [
        jsonS,
        jsonAction,
        jsonSPrime,
    ]
def writeSampleTrajectory(trajectory, fileName="generated-actions.json"):
    # data = []
    # for t in trajectory:
    #     action = t[1]
    #     actions.append(action)
    # write the actions taken to a file
    # with open(fileName, 'w') as f:
    #     for action in actions:
    #         actionName = INT_TO_DIRECTION_STR[action]
    #         f.write(actionName + "\n")
    trajectories = [trajectory]
    with open(fileName, "w") as f:
        json.dump([[jsonSerializeTrajectory(t) for t in traj] for traj in trajectories], f, indent=2)


ACTIONS = ['NORTH', 'SOUTH', 'EAST', 'WEST']

if __name__ == "__main__":
    action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST']
    action_space = len(action_commands)

    initial_phi_state = [1.25, 15.5, 0, 0, 1, 0, 0]
    expertTrajectories, _ = loadTrajectories("trajectories.pkl") # load the pkl file
    np.random.seed(42)
    theta_random = np.random.uniform(low=0.0, high=0.1, size=len(initial_phi_state))

    def probeNextState(phi_state, action):
        # return updateGameState(sock_game, phi_state, action)
        return updateGameState(phi_state, action)

    learner = MaxEntropyIRL(
        theta=theta_random,
        actions=np.arange(len(ACTIONS)),
        probeNextState=probeNextState,
        initialState=initial_phi_state,
        gameOver=gameOver,
        phi=phi,
    )
    theta_hat, grad_norms, theta_iterations = learner.learn(expertTrajectories, num_iterations=500, alpha=0.025, num_samples=50)
    print("\n\n Predicted Theta:\n\n", theta_hat)
    sampleTrajectory = learner.stochastic_trajectory(maxLength=200)
    writeSampleTrajectory(sampleTrajectory)
    plot_grad_norms(grad_norms)

