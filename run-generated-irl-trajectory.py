import json
import socket
import time
from utils import recv_socket_data
import numpy as np
from socket_agent_expert import hasBasket

action_commands = {
    0: 'NORTH',
    1: 'SOUTH',
    2: 'EAST',
    3: 'WEST',
    4: "INTERACT",
}
    
def load_generated_actions(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return data
    
def get_required_direction(player_pos, object_pos):
    dx = object_pos[0] - player_pos[0]
    dy = object_pos[1] - player_pos[1]
    
    if np.abs(dx) > np.abs(dy):
        return 2 if dx > 0 else 3
    else:
        return 1 if dy > 0 else 0
    
def turn(action, sock_game):
    turn_command = f"0 {action_commands[action]}"
    sock_game.send(str.encode(turn_command))
    actual_state = recv_socket_data(sock_game)
    actual_state = json.loads(actual_state)
    return actual_state

if __name__ == "__main__":
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    filename = 'generated_actions_per_subgoal.json'
    generated_actions = load_generated_actions(filename)

    print(f"Loaded generated actions from {filename}")

    BASKET_LOCATION = [3.5, 18.5]
    
    # wrapper around the actual action execution to get rid of some of the issues.
    # removes the turning aspect and always executes the movement actions directly.
    # for interact actions, turns to face the object first.
    def execute_action_with_turning(action, current_state):
        current_direction = current_state['observation']['players'][0]['direction']
        current_position = current_state['observation']['players'][0]['position']
        has_basket = hasBasket(current_state)
        
        # if movement is different than current direction, turn first
        if action in [0, 1, 2, 3] and current_direction != action:
            actual_state = turn(action, sock_game)
            current_direction = actual_state['observation']['players'][0]['direction']
        
        # for interact actions if we aren't already facing the basket, turn to face it
        if action == 4 and not has_basket:
            required_direction = get_required_direction(current_position, BASKET_LOCATION)
            if current_direction != required_direction:
                actual_state = turn(required_direction, sock_game)
                current_direction = actual_state['observation']['players'][0]['direction']
        elif action == 4 and has_basket:
            # do the interact twice - simulation env is finnicky
            sock_game.send(str.encode("0 INTERACT"))
            _ = recv_socket_data(sock_game)
            sock_game.send(str.encode("0 INTERACT"))
            actual_state = recv_socket_data(sock_game)
            actual_state = json.loads(actual_state)
            return actual_state
        
        # the real action
        command = f"0 {action_commands[action]}"
        sock_game.send(str.encode(command))
        actual_state = recv_socket_data(sock_game)
        actual_state = json.loads(actual_state)
        
        return actual_state
    
    sock_game.send(str.encode("0 RESET"))
    start_state = recv_socket_data(sock_game)
    current_state = json.loads(start_state)

    for action in generated_actions:
        current_state = execute_action_with_turning(action, current_state)
