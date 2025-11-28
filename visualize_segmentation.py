import numpy as np
import matplotlib.pyplot as plt
import pickle
from hirl import HIRLSegmenter, MAP_WIDTH, MAP_HEIGHT

def visualize_segmentation(window_size=2):
    print(f"\n--- Running with Window Size: {window_size} ---")
    print("Loading features...")
    try:
        features = np.load("hirl_features/feature_trajectories.npy", allow_pickle=True)
        expertTrajectories = [f for f in features]
    except Exception as e:
        print(f"Failed to load features: {e}")
        return

    print(f"Loaded {len(expertTrajectories)} trajectories.")
    
    print("Running segmenter...")
    segmenter = HIRLSegmenter()
    subgoals = segmenter.subgoals(expertTrajectories, windowSize=window_size)
    
    # Get raw transitions for debugging
    labels = segmenter.labels(expertTrajectories, windowSize=window_size)
    transitions, _ = segmenter.transitions(expertTrajectories, labels, windowSize=window_size)
    
    print(f"\nFound {len(transitions)} raw transitions (before clustering):")
    for i, t in enumerate(transitions):
        # Use correct indices 1 and 2 for X and Y
        tx = t[1] * MAP_WIDTH
        ty = t[2] * MAP_HEIGHT
        print(f"  Transition {i}: Map({tx:.2f}, {ty:.2f})")
    
    print(f"Found {len(subgoals)} subgoals.")
    
    # --- Plotting ---
    plt.figure(figsize=(10, 12))
    
    # 1. Plot Map Boundaries
    plt.xlim(0, MAP_WIDTH)
    plt.ylim(MAP_HEIGHT, 0) # Invert Y axis to match game coordinates (0,0 is top-left)
    
    # 2. Plot Object Locations (Hardcoded from print_locations.py)
    # Cart Returns
    plt.scatter([1, 2], [18.5, 18.5], c='orange', marker='s', s=100, label='Cart Return')
    # Milk Shelves
    plt.scatter([5.5, 7.5], [1.5, 1.5], c='cyan', marker='s', s=100, label='Milk Shelf')
    # Registers
    plt.scatter([1, 1], [4.5, 9.5], c='purple', marker='s', s=100, label='Register')
    # Exits (Left side, roughly)
    plt.plot([0, 0], [0, MAP_HEIGHT], 'k--', linewidth=2, label='Exit Wall')
    
    # 3. Plot Trajectories
    for i, traj in enumerate(expertTrajectories):
        traj = np.array(traj)
        # Convert normalized coordinates back to map coordinates
        # Feature indices: [0]=x_norm, [1]=y_norm
        xs = traj[:, 0] * MAP_WIDTH
        ys = traj[:, 1] * MAP_HEIGHT
        plt.plot(xs, ys, 'b.-', alpha=0.1, label='Expert Trajectory' if i == 0 else "")
        
    # Plot subgoals
    if len(subgoals) > 0:
        print("\nDetected Subgoals:")
        # Convert subgoal features to map coordinates
        # Feature indices: [0]=x_norm, [1]=y_norm
        sx = subgoals[:, 0] * MAP_WIDTH
        sy = subgoals[:, 1] * MAP_HEIGHT
    
        for i, (x, y) in enumerate(zip(sx, sy)):
            print(f"  Subgoal {i+1}: Map({x:.2f}, {y:.2f}) | Feature(x_norm={subgoals[i][0]:.2f}, y_norm={subgoals[i][1]:.2f})")
        
        plt.scatter(sx, sy, c='red', s=200, marker='*', label='Detected Subgoals', zorder=10)
        
        for i, (x, y) in enumerate(zip(sx, sy)):
            plt.annotate(f"Subgoal {i+1}", (x, y), xytext=(10, 10), textcoords='offset points', fontsize=12, color='red', fontweight='bold')

    plt.title(f"HIRL Segmentation (Window Size {window_size}): {len(subgoals)} Subgoals")
    plt.xlabel("X Position (Map)")
    plt.ylabel("Y Position (Map)")
    plt.xlim(0, MAP_WIDTH)
    plt.ylim(0, MAP_HEIGHT)
    plt.gca().invert_yaxis() # Pygame coordinates are inverted y
    plt.grid(True)
    plt.legend()

    output_file = f"hirl_visualization_w{window_size}.png"
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    
    visualize_segmentation(2)
