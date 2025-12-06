import json
import socket
import time
from utils import recv_socket_data
import numpy as np
from socket_agent_expert import hasBasket
import argparse

action_commands = {
    0: 'NORTH',
    1: 'SOUTH',
    2: 'EAST',
    3: 'WEST',
    4: "INTERACT",
}
BASKET_LOCATION = [3.5, 18.5]

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

def turn(action, sock_game, metrics):
    newMetrics = metrics.copy()
    turn_command = f"0 {action_commands[action]}"
    sock_game.send(str.encode(turn_command))
    actual_state = recv_socket_data(sock_game)
    actual_state = json.loads(actual_state)
    newMetrics['num_steps'] += 1
    newMetrics['num_violations'] += len(actual_state['violations'])
    return actual_state, newMetrics

# wrapper around the actual action execution to get rid of some of the issues.
# removes the turning aspect and always executes the movement actions directly.
# for interact actions, turns to face the object first.
def execute_action_with_turning(action, current_state, metrics):
    current_direction = current_state['observation']['players'][0]['direction']
    current_position = current_state['observation']['players'][0]['position']
    has_basket = hasBasket(current_state)

    newMetrics = metrics.copy()

    # if movement is different than current direction, turn first
    if action in [0, 1, 2, 3] and current_direction != action:
        actual_state, newMetrics = turn(action, sock_game, newMetrics)
        current_direction = actual_state['observation']['players'][0]['direction']
    
    # for interact actions if we aren't already facing the basket, turn to face it
    if action == 4 and not has_basket:
        required_direction = get_required_direction(current_position, BASKET_LOCATION)
        if current_direction != required_direction:
            actual_state, newMetrics = turn(required_direction, sock_game, newMetrics)
            current_direction = actual_state['observation']['players'][0]['direction']
    elif action == 4 and has_basket:
        # do the interact twice - simulation env is finnicky
        sock_game.send(str.encode("0 INTERACT"))
        actual_state = recv_socket_data(sock_game)
        actual_state = json.loads(actual_state)
        newMetrics['num_violations'] += len(actual_state['violations'])
        newMetrics['num_steps'] += 1

        sock_game.send(str.encode("0 INTERACT"))
        actual_state = recv_socket_data(sock_game)
        actual_state = json.loads(actual_state)
        newMetrics['num_violations'] += len(actual_state['violations'])
        newMetrics['num_steps'] += 1

        gameOver = 'Player 0 exited through an entrance' in actual_state['violations']

        return actual_state, newMetrics, gameOver
    
    # the real action
    command = f"0 {action_commands[action]}"
    sock_game.send(str.encode(command))
    actual_state = recv_socket_data(sock_game)
    actual_state = json.loads(actual_state)
    newMetrics['num_steps'] += 1
    newMetrics['num_violations'] += len(actual_state['violations'])
    gameOver = 'Player 0 exited through an entrance' in actual_state['violations']

    
    return actual_state, newMetrics, gameOver

if __name__ == "__main__":
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        type=str,
        help="file containing the actions to execute",
        default="generated_actions_per_subgoal.json"
    )
    parser.add_argument(
        '--output',
        type=str,
        help="file to write the final state to",
        default="final_state.json"
    )
    parser.add_argument(
        "--run_id",
        type=int,
        help="ID of the current run for metrics",
        default=0
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        help="file to write metrics to",
        default="generated_action_metrics.json"
    )
    parser.add_argument(
        "--success",
        action='store_true',
        help="whether the generated trajectory was successful",
    )
    args = parser.parse_args()
    filename = args.file
    output_file = args.output
    run_id = args.run_id
    metrics_file = args.metrics_file
    success = args.success
    
    generated_actions = load_generated_actions(filename)

    print(f"Loaded generated actions from {filename}")

    sock_game.send(str.encode("0 RESET"))
    start_state = recv_socket_data(sock_game)
    current_state = json.loads(start_state)
    metrics = {
        'run_id': run_id,
        'num_steps': 0,
        'num_violations': 0,
        'success': False,
    }

    for action in generated_actions:
        current_state, metrics, gameOver = execute_action_with_turning(action, current_state, metrics)
        if gameOver:
            print("Game over detected, stopping execution of further actions.")
            break


    with open(output_file, "w") as f:
        json.dump(current_state, f, indent=2)
    print(f"Finished executing actions, wrote final state to {output_file}")

    metrics['success'] = success and not gameOver
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {metrics_file}")