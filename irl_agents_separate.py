from hirl import HIRLSegmenter
import numpy as np
from max_entropy_irl import MaxEntropyIRL
import json
import pickle
import matplotlib.pyplot as plt
from irl_agent import shoppingActionMap, load_expert_trajectories
from dp_trajectory_segmentation import segmentTrajectoryBySubgoals

# TODO: when adding a new item, add its location to the target list, then update the start/final state, and the theta size.
BASKET_LOCATION = np.array([3.5, 18.5])
SAUSAGE_LOCATION = np.array([6.5, 10.5])
MILK_LOCATION = np.array([6.5, 2.5])
BANANA_LOCATION = np.array([10.5, 6.5])
REGISTER_LOCATION = np.array([2.75, 3.75])

TARGET_LOCATIONS = [BASKET_LOCATION, SAUSAGE_LOCATION, MILK_LOCATION, BANANA_LOCATION, REGISTER_LOCATION]

START_STATE = np.asarray([1.25, 15.5, 0.0, 0.0, 0.0, 0.0, 0.0])
FINAL_GOAL_LOCATION = np.asarray([2.75, 3.75, 1.0, 1.0, 1.0, 1.0, 1.0])

THETA_SIZE = 7

def makeGetNextState(targets):
    def getNextState(state, action):
        currentPosition, hasItems = state[:2], state[2:]
        if action == 4:
            # Now, we need to know if we are trying to pick up the basket or interact with a shelf
            # interaction - if we are within a threshold of the basket location, we pick it up. else nothing
            distances = np.linalg.norm(targets - currentPosition, axis=1)
            closest_target_idx = np.argmin(distances)
            closest_target = targets[closest_target_idx]
            # assumes whatever item is closest is the intended target
            if np.allclose(currentPosition, closest_target, atol=0.75):
                newState = state.copy()
                newState[closest_target_idx + 2] = 1.0 # pick up something
                return newState
            else:
                return state
        else:
            # normal movement, flags doesn't change
            newPosition = shoppingActionMap[action](currentPosition)
            return np.concatenate([newPosition, hasItems])


    return getNextState

def makeLearner(theta, initialState, subgoal, tol):
    def make_phi(goal):
        def phi(state):
            # phi is xdist, ydist, current flags (not distances)
            return np.concatenate([np.abs(goal[:2] - state[:2]), state[2:]])
        return phi
    
    def make_game_over(goal, tolerance):
        def game_over(state):
            return np.allclose(state, goal, atol=tolerance)
        return game_over
    
    getNextState = makeGetNextState(np.array(TARGET_LOCATIONS))

    learner = MaxEntropyIRL(
        theta=theta,
        actions=np.arange(5),
        probeNextState=getNextState,
        initialState=initialState,
        gameOver=make_game_over(subgoal, tol),
        phi=make_phi(subgoal)
    )

    return learner

def segmentTrajectoriesBySubgoal(expertTrajectories, subgoals):
    segments_by_subgoal = [[] for _ in range(len(subgoals))]
    for trajectory in expertTrajectories:
        assignment = segmentTrajectoryBySubgoals(np.array(trajectory), subgoals)
        startIdx = 0
        for j, trajectoryIdx in enumerate(assignment):
            segment = trajectory[startIdx:trajectoryIdx + 1]
            segments_by_subgoal[j].append(segment)
            startIdx = trajectoryIdx + 1

    return segments_by_subgoal


def trainPerSubgoalMaxEnt(segments_by_subgoal, subgoals, initialXY, tol=0.2, num_iterations=200):    
    learned_agents = []
    
    for i, (subgoal, segments) in enumerate(zip(subgoals, segments_by_subgoal)):
        print(f"\nTraining agent for subgoal {i+1}/{len(subgoals)}: {subgoal}")
        print(f"  Number of expert segments: {len(segments)}")
        
        if len(segments) == 0:
            print(f"  Warning: No segments found for subgoal {i}, skipping")
            learned_agents.append(None)
            continue

        theta_random = np.random.uniform(low=0.0, high=0.1, size=THETA_SIZE)

        # Initial state is the start position for first segment, previous subgoal for others
        initial_state = initialXY if i == 0 else subgoals[i-1]

        learner = makeLearner(theta_random, initial_state, subgoal, tol)
        
        theta_hat, _ = learner.learn(segments, num_iterations=num_iterations, alpha=0.05, num_samples=50)
        print(f"  Learned theta: {theta_hat}")
        
        learned_agents.append((theta_hat, learner, subgoal, initial_state))
    
    return learned_agents


def generatePerSubgoalTrajectory(learned_agents, maxLength=200, epsilon=0.05):
    full_trajectory = []
    full_actions = []
    
    for i, agent_data in enumerate(learned_agents):
        if agent_data is None:
            print(f"Skipping subgoal {i} (no trained agent)")
            continue
        
        theta, learner, subgoal, initial_state = agent_data
        
        # Update learner's initial state to current position
        if len(full_trajectory) > 0:
            learner.initialState = full_trajectory[-1]
        
        # Generate segment trajectory
        segment, actions = learner.greedy_trajectory(maxLength=maxLength, epsilon=epsilon, recordActions=True)
        
        # Add to full trajectory (skip first point if not the first segment to avoid duplication)
        if len(full_trajectory) > 0:
            full_trajectory.extend(segment[1:])
        else:
            full_trajectory.extend(segment)
        
        full_actions.extend(actions)
        
        print(f"Subgoal {i+1}: Generated segment with {len(segment)} steps, reached {segment[-1]}")
    
    return full_trajectory, full_actions

def getExpertTrajectoriesWithNoise(filePath, noise=0.05):
    # Mask the shape so only x,y get noise
    mask = np.full((THETA_SIZE), True)
    mask[[0,1]] = False

    expertTrajectories = load_expert_trajectories(filePath, noise=noise, maskShape=mask)
    return expertTrajectories

def getSubgoals(expertTrajectories):
    segmenter = HIRLSegmenter()
    subgoals = segmenter.subgoals(expertTrajectories)
    
    # Round subgoals and remove duplicates while preserving order
    subgoals = np.round(subgoals * 4) / 4
    _, unique_indices = np.unique(subgoals, axis=0, return_index=True)
    subgoals = subgoals[np.sort(unique_indices)]
    
    subgoals = np.vstack([subgoals, FINAL_GOAL_LOCATION])
    
    # Filter out subgoals that don't have expert data
    segments_by_subgoal = segmentTrajectoriesBySubgoal(expertTrajectories, subgoals)
    valid_subgoals = []
    for i, segments in enumerate(segments_by_subgoal):
        if len(segments) > 0:
            valid_subgoals.append(subgoals[i])
        else:
            print(f"Removing subgoal {i+1} (no expert segments): {subgoals[i]}")
    
    subgoals = np.array(valid_subgoals)

    return subgoals, segments_by_subgoal

def learnSegments(subgoals, segments_by_subgoal, startState, tol=0.2):
    print("\n" + "="*60)
    print("TRAINING PER-SUBGOAL AGENTS")
    print("="*60)

    learned_agents = trainPerSubgoalMaxEnt(
        segments_by_subgoal,
        subgoals,
        initialXY=startState,
        tol=tol,
        num_iterations=200
    )

    return learned_agents

def saveLearnedAgents(learnedAgents):
    # Save the learned agents (theta, subgoal, initial_state - enough to reconstruct)
    with open("learned_per_subgoal_agents.pkl", "wb") as f:
        pickle.dump([(theta, subgoal, initial_state) for theta, learner, subgoal, initial_state in learnedAgents if learner is not None], f)
    print("\nSaved learned per-subgoal agents to learned_per_subgoal_agents.pkl")

def loadLearnedAgents(tol):
    print("\n" + "="*60)
    print("LOADING PRE-LEARNED PER-SUBGOAL AGENTS FROM FILE, SKIPPING TRAINING")
    print("="*60)

    with open("learned_per_subgoal_agents.pkl", "rb") as f:
        agent_data = pickle.load(f)
        learned_agents = []
        for theta, subgoal, initial_state in agent_data:
            learner = makeLearner(theta, initial_state, subgoal, tol)
            learned_agents.append((theta, learner, subgoal, initial_state))
    return learned_agents

def generateLearnedTrajectory(learned_agents):
    print("\n" + "="*60)
    print("GENERATING TRAJECTORY WITH PER-SUBGOAL AGENTS")
    print("="*60)
    
    per_subgoal_trajectory, per_subgoal_actions = generatePerSubgoalTrajectory(learned_agents, maxLength=100, epsilon=0.05)

    # Save trajectory
    trajectory_to_save = [step.tolist() for step in per_subgoal_trajectory]
    with open("generated_trajectory_per_subgoal.json", "w") as f:
        json.dump(trajectory_to_save, f, indent=2)
    print(f"\nSaved per-subgoal trajectory to generated_trajectory_per_subgoal.json")
    print(f"Total trajectory length: {len(per_subgoal_trajectory)}")

    # Save actions
    with open("generated_actions_per_subgoal.json", "w") as f:
        json.dump(np.asarray(per_subgoal_actions).tolist(), f, indent=2)
    print(f"Saved per-subgoal actions to generated_actions_per_subgoal.json")

    return per_subgoal_trajectory

def plotSampledTrajectory(sampleTrajectory, expertTrajectories, subgoals, startState):
    plt.figure(figsize=(10, 8))
    for i, traj in enumerate(expertTrajectories):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], 'b.-', alpha=0.2, linewidth=0.5,
                label='Expert Trajectory' if i == 0 else "")
    
    x, y = [step[0] for step in sampleTrajectory], [step[1] for step in sampleTrajectory]
    plt.plot(x, y, 'o-', color='purple', linewidth=2, markersize=4, label='Per-Subgoal Agent Trajectory')
    
    plt.scatter(subgoals[:, 0], subgoals[:, 1], c='red', s=250, marker='*',
                label='Subgoals', zorder=10)
    
    # Add numbers to subgoals
    for i, subgoal in enumerate(subgoals):
        plt.text(subgoal[0] + 0.2, subgoal[1] + 0.2, str(i+1), fontsize=12, fontweight='bold',
                ha='center', va='center', color='black', zorder=11,
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5))
    
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


# If you want to train agent from scratch, set to True
learnMode = True

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(142)

    expertTrajectories = getExpertTrajectoriesWithNoise("trajectories.pkl")
    subgoals, segments_by_subgoal = getSubgoals(expertTrajectories)
    print("Subgoals:\n", subgoals)

    tol = 0.2
    
    if learnMode:
        learned_agents = learnSegments(subgoals, segments_by_subgoal, startState, tol)
        saveLearnedAgents(learned_agents)
    else:
        learned_agents = loadLearnedAgents(tol)
    
    sampleTrajectory = generateLearnedTrajectory(learned_agents)
    
    plotSampledTrajectory(sampleTrajectory, expertTrajectories, subgoals, startState)
