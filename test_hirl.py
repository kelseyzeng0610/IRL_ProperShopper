import numpy as np
import matplotlib.pyplot as plt
from hirl import HIRLSegmenter

def visualize():
    print("Generating complex trajectories...")
    trajectories = []
    # Create 5 noisy C-shaped trajectories
    for _ in range(5):
        traj = []
        # Segment 1: (0,0) to (5,0)
        for x in range(6): 
            traj.append((x + np.random.normal(0, 0.1), 0 + np.random.normal(0, 0.1)))
        
        # Segment 2: (5,1) to (5,5)
        for y in range(1, 6): 
            traj.append((5 + np.random.normal(0, 0.1), y + np.random.normal(0, 0.1)))
            
        # Segment 3: (4,5) to (0,5)
        for x in range(4, -1, -1): 
            traj.append((x + np.random.normal(0, 0.1), 5 + np.random.normal(0, 0.1)))
            
        trajectories.append(traj)

    expertTrajectories = np.asarray(trajectories)
    
    print("Running HIRL Segmenter...")
    segmenter = HIRLSegmenter()
    subgoals = segmenter.subgoals(expertTrajectories)
    print("Subgoals found:", subgoals)

    plt.figure(figsize=(8, 8))
    
    for i, traj in enumerate(expertTrajectories):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], 'b.-', alpha=0.3, label='Expert Trajectory' if i == 0 else "")
        
    if len(subgoals) > 0:
        plt.scatter(subgoals[:, 0], subgoals[:, 1], c='red', s=200, marker='*', label='Detected Subgoals', zorder=10)
        
        for i, subgoal in enumerate(subgoals):
            plt.annotate(f"Subgoal {i+1}", (subgoal[0], subgoal[1]), xytext=(10, 10), textcoords='offset points', fontsize=12, color='red')

    plt.title("HIRL: Trajectories and Spatial Clusters")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    
    output_file = "hirl_visualization.png"
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    visualize()
