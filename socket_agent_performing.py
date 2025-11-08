#Author: Hang Yu

import json
import socket

from env import SupermarketEnv
from utils import recv_socket_data

from Q_Learning_agent import QLAgent  # Make sure to import your QLAgent class
from socket_agent_training import preloadQTable, buildGoals, achievedGoal, getAction

if __name__ == "__main__":
    action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'INTERACT', 'TOGGLE_CART']
    # Initialize Q-learning agent
    action_space = len(action_commands)   # Assuming your action space size is equal to the number of action commands

    qTableFilePath = "training-output.json"

    # For performing, we set epsilon to 0 because we don't want to randomly try actions that aren't optimal
    agent = QLAgent(action_space, epsilon=0.0)
    preloadedQTables = preloadQTable(qTableFilePath)
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    training_time = 100
    episode_length = 4000 # Episodes can be long now, so extend the default max - if it gets stuck we would notice a lack of progress messages anyway
    for i in range(training_time):
        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        cnt = 0

        # Keep track of our goal so we know when to end the simulation
        goals, basketMode = buildGoals(state)
        agent.setGoals(goals)
        agent.withQTables(preloadedQTables)
        current_goal, current_goal_idx = goals[0], 0
        while not state['gameOver']:
            cnt += 1
            # Choose an action based on the current state
            action, action_index, learningMode, forbidden = getAction(agent, state, current_goal, basketMode, action_commands)

            sock_game.send(str.encode(action))  # send action to env

            next_state = recv_socket_data(sock_game)  # get observation from env
            next_state = json.loads(next_state)

            # Update state
            state = next_state

            if achievedGoal(state, current_goal):
                print("** Achieved", current_goal['name'])
                current_goal_idx += 1
                if current_goal_idx == len(goals):
                    print("******", i+1, "COMPLETED ALL GOALS *******")
                    break

                current_goal = goals[current_goal_idx]
                agent.achievedGoal() # tell the agent we achieved the goal

            if cnt > episode_length:
                break
        # Additional code for end of episode if needed

    # Close socket connection
    sock_game.close()

