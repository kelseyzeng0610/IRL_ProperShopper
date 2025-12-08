import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
from irl_agents_separate import plotSampledTrajectory, START_STATE, THETA_SIZE, getExpertTrajectoriesWithNoise

if __name__ == "__main__":
    # load trajectories
    expertTrajectories = getExpertTrajectoriesWithNoise("trajectories.pkl")
    startState = START_STATE

    # plot the expert demos
    plt.figure(figsize=(10, 8))
    for i, traj in enumerate(expertTrajectories):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], 'b.-', alpha=0.2, linewidth=0.5,
                label='Expert Trajectory' if i == 0 else "")
    
    # x, y = [step[0] for step in sampleTrajectory], [step[1] for step in sampleTrajectory]
    # plt.plot(x, y, 'o-', color='purple', linewidth=2, markersize=4, label='Per-Subgoal Agent Trajectory')
    
    # if len(subgoals) > 0:
    #     plt.scatter(subgoals[:, 0], subgoals[:, 1], c='red', s=250, marker='*',
    #             label='Subgoals', zorder=10)
    
    # Add numbers to subgoals
    # for i, subgoal in enumerate(subgoals):
    #     plt.text(subgoal[0] + 0.2, subgoal[1] + 0.2, str(i+1), fontsize=12, fontweight='bold',
    #             ha='center', va='center', color='black', zorder=11,
    #             bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5))
    
    plt.scatter(startState[0], startState[1], c='green', s=200, marker='o',
                label='Start', zorder=10)
    
    plt.title("Expert Trajectories Visualization")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("presentation-plots/expert_trajectories.png")
    plt.show()
    