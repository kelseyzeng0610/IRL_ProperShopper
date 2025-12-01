import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from max_entropy_irl import MaxEntropyIRL 
from feature_extraction import StateFeatureExtractor
from hirl import HIRLSegmenter

MAP_WIDTH = 20.0
MAP_HEIGHT = 25.0
STEP_SIZE = 0.15 # need to make sure 
NORM_STEP_X = STEP_SIZE / MAP_WIDTH
NORM_STEP_Y = STEP_SIZE / MAP_HEIGHT

BASKET_X = 3.45 / MAP_WIDTH
BASKET_Y = 17.7 / MAP_HEIGHT
INTERACT_RADIUS = 0.005 #tight radius to force exact positioning

def get_features_basket(state):
    """
    0: x
    1: y
    2: dist_to_basket
    3: dist_to_cart
    7: has_basket
    8: has_cart
    """
    # 1. Distance Feature (minimize this)
    dist = np.sqrt((state[0] - BASKET_X)**2 + (state[1] - BASKET_Y)**2)
    
    # 2. Success Feature (maximize this)
    has_basket = state[7]
    
    # 3. Step Cost (minimize this)
    step_cost = 1.0
    
    return np.array([dist, has_basket, step_cost])

def get_next_state_basket(state, action):
    """
    Simulates the environment dynamics (Transition Function).
    """
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
            # Set the 'has_basket' flag
            next_state[7] = 1.0
            
    # Update the distance feature in the state vector (index 2)
    new_dist = np.sqrt((next_state[0] - BASKET_X)**2 + (next_state[1] - BASKET_Y)**2)
    next_state[2] = new_dist
    
    return next_state

def is_done(state):
    """Episode ends when basket is acquired."""
    return state[7] > 0.5


def segment_data_with_hirl(trajectories):
    """
    Uses HIRL Segmenter to automatically find the basket subgoal and slice data.
    """
    print("Extracting features for HIRL...")
    extractor = StateFeatureExtractor()
    feature_trajs = []
    for traj in trajectories:
        feats = [extractor.extract(obs) for obs in traj['observations']]
        feature_trajs.append(np.array(feats))
        
    print("Running HIRL Segmenter...")
    segmenter = HIRLSegmenter()
    # Use window_size=2 as per visualization
    subgoals = segmenter.subgoals(feature_trajs, windowSize=2)
    
    if len(subgoals) == 0:
        print("HIRL found no subgoals!")
        return []
        
    # The first subgoal should be the basket
    basket_subgoal = subgoals[0]
    print(f"HIRL identified first subgoal at features: {basket_subgoal}")
    print(f"  (Approx Pos: {basket_subgoal[0]*MAP_WIDTH:.2f}, {basket_subgoal[1]*MAP_HEIGHT:.2f})")
    
    # Slice trajectories up to this subgoal
    basket_segments = []
    for f_traj in feature_trajs:
        segment = []
        for feat in f_traj:
            segment.append(feat)
            # Check if we are close to the subgoal (in feature space)
            # Using a simple distance threshold
            if np.linalg.norm(feat - basket_subgoal) < 0.1: 
                # We reached the subgoal
                break
        
        if len(segment) > 0:
            basket_segments.append(np.array(segment))
            
    print(f"Segmented {len(basket_segments)} trajectories using HIRL.")
    return basket_segments

def main():
    # 1. Load Data
    print("Loading demonstrations...")
    try:
        with open('demonstrations/trajectories.pkl', 'rb') as f:
            full_trajs = pickle.load(f)
    except FileNotFoundError:
        print("No demonstrations found. Please run collect_expert_data.py first.")
        return
        
    expert_segments = segment_data_with_hirl(full_trajs) # New HIRL way
    
    if not expert_segments:
        print("No valid basket segments found!")
        return

    # Debug: Check segment endpoints
    print(f"Segment 0 Start: {expert_segments[0][0][:2]}")
    print(f"Segment 0 End:   {expert_segments[0][-1][:2]}")
    print(f"Target Basket:   ({BASKET_X:.3f}, {BASKET_Y:.3f})")

    # 3. Initialize Learner
    # Heuristic Initialization: 
    # [Dist (Negative), Success (Positive), Step (Negative)]
    init_theta = np.array([-0.5, 1.0, -0.1]) # Stronger priors
    
    learner = MaxEntropyIRL(
        theta=init_theta,
        actions=np.arange(5),
        probeNextState=get_next_state_basket,
        initialState=expert_segments[0][0], 
        gameOver=is_done,
        phi=get_features_basket
    )
    
    # 4. Train
    print("Training MaxEnt IRL...")
    theta, grad_norms = learner.learn(
        expert_segments,
        num_iterations=500, # Increased for better convergence
        alpha=0.05,        # Increased
        maxTrajectoryLength=50
    )
    
    print(f"Final Weights: {theta}")
    # Expect: [Neg, Pos, Neg]
    
    # 5. Save
    output = {'theta': theta, 'norm_step_x': NORM_STEP_X, 'basket_pos': (BASKET_X, BASKET_Y)}
    with open('basket_policy_params.pkl', 'wb') as f:
        pickle.dump(output, f)
    print("Saved to basket_policy_params.pkl")

    # 6. Quick Visualization
    print("Generating Validation Path...")
    # Use epsilon=0.0 to get the pure greedy policy (no noise)
    path, actions = learner.greedy_trajectory(maxLength=50, epsilon=0.0, recordActions=True)
    
    actions_list = [int(a) for a in actions]
    with open('generated_actions_per_subgoal.json', 'w') as f:
        json.dump(actions_list, f)
    print("Saved generated actions to generated_actions_per_subgoal.json")
    
    # Convert back to Map units for readability
    path_x = [p[0]*MAP_WIDTH for p in path]
    path_y = [p[1]*MAP_HEIGHT for p in path]
    
    plt.figure()
    plt.plot(path_x, path_y, marker='o', label='Learned Path')
    plt.scatter([BASKET_X*MAP_WIDTH], [BASKET_Y*MAP_HEIGHT], c='r', marker='x', s=100, label='Target (Basket)')
    plt.legend()
    plt.title("Learned Basket Pickup Trajectory")
    plt.gca().invert_yaxis() 
    plt.savefig('basket_validation.png')
    print("Saved validation plot to basket_validation.png")

if __name__ == "__main__":
    main()