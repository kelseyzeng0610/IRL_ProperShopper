import json
import socket
import time
from utils import recv_socket_data

def load_generated_actions(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return data

if __name__ == "__main__":
    action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST']
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    filename = 'generated_actions_per_subgoal.json'
    generated_actions = load_generated_actions(filename)

    print(f"Loaded generated actions from {filename}")

    action_commands = {
        0: 'NORTH',
        1: 'SOUTH',
        2: 'EAST',
        3: 'WEST',
        4: "INTERACT",
    }
    
    # wrapper around the actual action execution to get rid of some of the issues.
    # removes the turning aspect and always executes the movement actions directly.
    # for interact actions, executes it twice to ensure stability.
    def execute_action_with_turning(action, current_state):
        current_direction = current_state['observation']['players'][0]['direction']
        
        # if movement is different than current direction, turn first
        if action in [0, 1, 2, 3] and current_direction != action:
            turn_command = f"0 {action_commands[action]}"
            sock_game.send(str.encode(turn_command))
            actual_state = recv_socket_data(sock_game)
            actual_state = json.loads(actual_state)
        
        # TODO: there have been issues in the past with interact actions sometimes failing, so at some point
        # we may want to replace every INTERACT with two INTERACTs here for stability.
        # for now this seems to be ok.
        
        # the real action
        command = f"0 {action_commands[action]}"
        sock_game.send(str.encode(command))
        actual_state = recv_socket_data(sock_game)
        actual_state = json.loads(actual_state)
        time.sleep(0.1)
        
        return actual_state
    
    sock_game.send(str.encode("0 RESET"))
    start_state = recv_socket_data(sock_game)
    current_state = json.loads(start_state)

    for action in generated_actions:
        current_state = execute_action_with_turning(action, current_state)
