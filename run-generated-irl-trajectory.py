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
        3: 'WEST'
    }
    sock_game.send(str.encode("0 RESET"))
    recv_socket_data(sock_game)

    for action in generated_actions:
        command = f"0 {action_commands[action]}"
        sock_game.send(str.encode(command))
        recv_socket_data(sock_game)
        time.sleep(0.1)
