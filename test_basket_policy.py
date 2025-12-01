import numpy as np
import pickle
import socket
import json
import time
import matplotlib.pyplot as plt
from feature_extraction import StateFeatureExtractor

# --- CONSTANTS (match training) ---
MAP_WIDTH = 20.0
MAP_HEIGHT = 25.0
STEP_SIZE = 0.15
NORM_STEP_X = STEP_SIZE / MAP_WIDTH
NORM_STEP_Y = STEP_SIZE / MAP_HEIGHT
BASKET_X = 3.45 / MAP_WIDTH
BASKET_Y = 17.7 / MAP_HEIGHT
INTERACT_RADIUS = 0.005

def get_features_basket(state):
    
    dist = np.sqrt((state[0] - BASKET_X)**2 + (state[1] - BASKET_Y)**2)
    
    # 2. Success Feature (index 7 is has_basket)
    has_basket = state[7]
    
    # 3. Step Cost
    step_cost = 1.0
    
    return np.array([dist, has_basket, step_cost])

def get_next_state_basket(state, action):
    next_state = state.copy()
    
    # Movement Logic (0:N, 1:S, 2:E, 3:W, 4:Interact)
    if action == 0:   # NORTH
        next_state[1] = max(0.0, state[1] - NORM_STEP_Y)
    elif action == 1: # SOUTH
        next_state[1] = min(1.0, state[1] + NORM_STEP_Y)
    elif action == 2: # EAST
        next_state[0] = min(1.0, state[0] + NORM_STEP_X)
    elif action == 3: # WEST
        next_state[0] = max(0.0, state[0] - NORM_STEP_X)
    elif action == 4: # INTERACT
        dist = np.sqrt((state[0] - BASKET_X)**2 + (state[1] - BASKET_Y)**2)
        if dist < INTERACT_RADIUS:
            next_state[7] = 1.0 
            
    next_state[2] = np.sqrt((next_state[0] - BASKET_X)**2 + (next_state[1] - BASKET_Y)**2)
    
    return next_state

def recv_socket_data(sock):
    BUFF_SIZE = 4096
    data = b''
    while True:
        try:
            part = sock.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                break
        except BlockingIOError:
            break
    return data

def main():
    # 1. Load Policy
    print("Loading policy...")
    with open('basket_policy_params.pkl', 'rb') as f:
        params = pickle.load(f)
        theta = params['theta']
    print(f"Theta: {theta}")

    HOST = '127.0.0.1'
    PORT = 9000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    
    print("Resetting environment...")
    sock.send(b"0 RESET")
    resp = recv_socket_data(sock)
    obs = json.loads(resp)
    
    extractor = StateFeatureExtractor()
    
    steps = 0
    max_steps = 50
    
    action_map = {0: 'NORTH', 1: 'SOUTH', 2: 'EAST', 3: 'WEST', 4: 'INTERACT'}
    path = []
    
    while steps < max_steps:
        # Extract current state features (size 11)
        current_features = extractor.extract(obs)
        
        # Check if done
        if current_features[7] > 0.5: # has_basket
            print("Success! Basket acquired.")
            break
            
        # Select best action
        best_action = -1
        best_val = -float('inf')
        
        # Lookahead 1 step
        for a in range(5):
            # Simulate next state
            next_features = get_next_state_basket(current_features, a)
            
            # phi converts size 11 -> size 3
            phi_next = get_features_basket(next_features)
            reward = np.dot(theta, phi_next)
            
            if reward > best_val:
                best_val = reward
                best_action = a
        
        # Execute action
        cmd = action_map[best_action]
        
        # Debug info
        pos_x = current_features[0] * MAP_WIDTH
        pos_y = current_features[1] * MAP_HEIGHT
        dist = np.sqrt((current_features[0] - BASKET_X)**2 + (current_features[1] - BASKET_Y)**2)
        print(f"Step {steps}: Pos({pos_x:.2f}, {pos_y:.2f}) Dist({dist:.4f}) -> Action {cmd} (Reward: {best_val:.3f})")
        
        # Record path
        path.append((pos_x, pos_y))
        
        sock.send(f"0 {cmd}".encode())
        resp = recv_socket_data(sock)
        if not resp:
            break
        obs = json.loads(resp)
        steps += 1
        time.sleep(0.1)

    sock.close()
    
    # Plot the real-world trajectory
    if path:
        path_x, path_y = zip(*path)
        plt.figure(figsize=(8, 10))
        plt.plot(path_x, path_y, 'b-o', label='Real Execution Path', linewidth=2)
        plt.scatter([path_x[0]], [path_y[0]], c='g', s=100, label='Start')
        plt.scatter([path_x[-1]], [path_y[-1]], c='purple', s=100, label='End')
        plt.scatter([BASKET_X*MAP_WIDTH], [BASKET_Y*MAP_HEIGHT], c='r', marker='x', s=150, label='Target (Basket)')
        
        
        plt.title("Real-World Closed-Loop Trajectory")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.gca().invert_yaxis() # Match game coords
        plt.savefig('real_world_basket_trajectory.png')
        print("Saved execution plot to real_world_basket_trajectory.png")

if __name__ == "__main__":
    main()
