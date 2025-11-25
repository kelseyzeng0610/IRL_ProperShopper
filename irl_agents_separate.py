from hirl import HIRLSegmenter
import numpy as np
from max_entropy_irl import MaxEntropyIRL
import json
import pickle
import matplotlib.pyplot as plt
from irl_agent import shoppingActionMap, load_expert_trajectories


def segmentTrajectoriesBySubgoal(expertTrajectories, subgoals, tol=0.3):
    segments_by_subgoal = [[] for _ in range(len(subgoals))]
    
    for trajectory in expertTrajectories:
        current_segment = []
        current_subgoal_idx = 0
        
        for step in trajectory:
            current_segment.append(step)
            
            # Check if we reached the current subgoal
            if current_subgoal_idx < len(subgoals):
                if np.allclose(step, subgoals[current_subgoal_idx], atol=tol):
                    # Save this segment
                    segments_by_subgoal[current_subgoal_idx].append(current_segment)
                    current_segment = [step]  # Start next segment from this subgoal
                    current_subgoal_idx += 1
        
    return segments_by_subgoal


def trainPerSubgoalMaxEnt(expertTrajectories, subgoals, initialXY, tol=0.3, num_iterations=200):
    segments_by_subgoal = segmentTrajectoriesBySubgoal(expertTrajectories, subgoals, tol)
    
    learned_agents = []
    
    for i, (subgoal, segments) in enumerate(zip(subgoals, segments_by_subgoal)):
        print(f"\nTraining agent for subgoal {i+1}/{len(subgoals)}: {subgoal}")
        print(f"  Number of expert segments: {len(segments)}")
        
        if len(segments) == 0:
            print(f"  Warning: No segments found for subgoal {i}, skipping")
            learned_agents.append(None)
            continue
        
        # Initial state is the start position for first segment, previous subgoal for others
        initial_state = initialXY if i == 0 else subgoals[i-1]
        
        def make_phi(goal):
            def phi(state):
                return np.abs(goal - state)
            return phi
        
        def make_game_over(goal, tolerance):
            def game_over(state):
                return np.allclose(state, goal, atol=tolerance)
            return game_over
        
        theta_random = np.random.uniform(low=0.0, high=0.1, size=2)
        learner = MaxEntropyIRL(
            theta=theta_random,
            actions=np.arange(4),
            probeNextState=lambda state, action: shoppingActionMap[action](state),
            initialState=initial_state,
            gameOver=make_game_over(subgoal, tol),
            phi=make_phi(subgoal)
        )
        
        theta_hat, _ = learner.learn(segments, num_iterations=num_iterations, alpha=0.05, num_samples=50)
        print(f"  Learned theta: {theta_hat}")
        
        learned_agents.append((theta_hat, learner, subgoal, initial_state))
    
    return learned_agents


def generatePerSubgoalTrajectory(learned_agents, maxLength=200, epsilon=0.05):
    full_trajectory = []
    
    for i, agent_data in enumerate(learned_agents):
        if agent_data is None:
            print(f"Skipping subgoal {i} (no trained agent)")
            continue
        
        theta, learner, subgoal, initial_state = agent_data
        
        # Update learner's initial state to current position
        if len(full_trajectory) > 0:
            learner.initialState = full_trajectory[-1]
        
        # Generate segment trajectory
        segment = learner.greedy_trajectory(maxLength=maxLength, epsilon=epsilon)
        
        # Add to full trajectory (skip first point if not the first segment to avoid duplication)
        if len(full_trajectory) > 0:
            full_trajectory.extend(segment[1:])
        else:
            full_trajectory.extend(segment)
        
        print(f"Subgoal {i+1}: Generated segment with {len(segment)} steps, reached {segment[-1]}")
    
    return full_trajectory

if __name__ == "__main__":
    noise = 0.05
    # Mask the shape so only x,y get noise
    # TODO: to train on the cart, switch the file path below
    # will also need to adjust start state and final state, and set the third value of maskShape to True so we don't add noise to the cart flag
    expertTrajectories = load_expert_trajectories("trajectories.pkl", noise=noise, maskShape=[False, False])

    segmenter = HIRLSegmenter()
    subgoals = np.unique(segmenter.subgoals(expertTrajectories), axis=0)

    startState = np.asarray([1.25, 15.5])

    finalGoalLocation = np.asarray([5.75, 11])
    subgoals = np.vstack([np.round(subgoals * 4) / 4, finalGoalLocation])
    print("Subgoals:", subgoals)

    print("\n" + "="*60)
    print("TRAINING PER-SUBGOAL AGENTS")
    print("="*60)
    
    tol = 0.2
    learned_agents = trainPerSubgoalMaxEnt(
        expertTrajectories,
        subgoals,
        initialXY=startState,
        tol=tol,
        num_iterations=200
    )
    
    print("\n" + "="*60)
    print("GENERATING TRAJECTORY WITH PER-SUBGOAL AGENTS")
    print("="*60)
    
    per_subgoal_trajectory = generatePerSubgoalTrajectory(learned_agents, maxLength=100, epsilon=0.05)
    
    # Save trajectory
    trajectory_to_save = [step.tolist() for step in per_subgoal_trajectory]
    with open("generated_trajectory_per_subgoal.json", "w") as f:
        json.dump(trajectory_to_save, f, indent=2)
    print(f"\nSaved per-subgoal trajectory to generated_trajectory_per_subgoal.json")
    print(f"Total trajectory length: {len(per_subgoal_trajectory)}")
    
    # Visualize
    plt.figure(figsize=(10, 8))
    for i, traj in enumerate(expertTrajectories):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], 'b.-', alpha=0.2, linewidth=0.5,
                label='Expert Trajectory' if i == 0 else "")
    
    x, y = [step[0] for step in per_subgoal_trajectory], [step[1] for step in per_subgoal_trajectory]
    plt.plot(x, y, 'o-', color='purple', linewidth=2, markersize=4, label='Per-Subgoal Agent Trajectory')
    
    plt.scatter(subgoals[:, 0], subgoals[:, 1], c='red', s=250, marker='*',
                label='Subgoals', zorder=10)
    plt.scatter(startState[0], startState[1], c='green', s=200, marker='o',
                label='Start', zorder=10)
    
    plt.title("Per-Subgoal MaxEnt Agents: Generated Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("per_subgoal_agents_trajectory.png")
    print("Visualization saved to per_subgoal_agents_trajectory.png")
    plt.show()
    