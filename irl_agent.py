from hirl import HIRLSegmenter
import numpy as np
from max_entropy_irl import MaxEntropyIRL
import json
import pickle
import matplotlib.pyplot as plt

def get_xy(gameState):
    playerPos = gameState["observation"]["players"][0]["position"]
    return (playerPos[0], playerPos[1])

def addNoiseToTrajectory(trajectory, noiseLevel=0.05, maskShape=None):
    noisy = []
    for step in trajectory:
        if maskShape is not None:
            assert len(maskShape) == len(step), "Mask shape must match step dimension"
            noise = np.array([0.0 if maskShape[i] else np.random.uniform(-noiseLevel, noiseLevel) for i in range(len(step))])
        else:
            noise = np.random.uniform(-noiseLevel, noiseLevel, size=step.shape)
        noisyStep = step + noise
        noisy.append(noisyStep)
    
    return np.asarray(noisy)

def load_expert_trajectories(fileName, noise=0.05, maskShape=None):
    with open(fileName, "rb") as f:
        data = pickle.load(f)
    
    # Problem: because of the rounding in the shopping environment and because our state doesn't
    # take orientation into account, our trajectories are often exactly the same
    # To resolve this, add random noise to each point in the trajectory
    return [addNoiseToTrajectory(traj, noiseLevel=noise, maskShape=maskShape) for traj in data]


def plotSubgoals(expertTrajectories, subgoals):
    plt.figure(figsize=(8, 8))
    for i, traj in enumerate(expertTrajectories):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], 'b.-', alpha=0.3, label='Expert Trajectory' if i == 0 else "")
        
    if len(subgoals) > 0:
        plt.scatter(subgoals[:, 0], subgoals[:, 1], c='red', s=200, marker='*', label='Detected Subgoals', zorder=10)
        startPoint = expertTrajectories[0][0]
        plt.scatter(startPoint[0], startPoint[1], c='green', s=200, marker='o', label='Start Point', zorder=10)

        plt.annotate(f"Start", (expertTrajectories[0][0][0], expertTrajectories[0][0][1]), xytext=(10, 10), textcoords='offset points', fontsize=12, color='green', rotation=15)
        for i, subgoal in enumerate(subgoals):
            if i < len(subgoals) - 1:
                plt.annotate(f"Subgoal {i+1}", (subgoal[0], subgoal[1]), xytext=(10, 10), textcoords='offset points', fontsize=12, color='red', rotation=15)
            else:
                plt.annotate(f"Final Goal", (subgoal[0], subgoal[1]), xytext=(10, 10), textcoords='offset points', fontsize=12, color='red', rotation=15)

    plt.title("Actual Shopping Trajectories with HIRL-Detected Subgoals")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    output_file = "hirl_shopping_trajectories.png"
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")


shoppingActionMap = {
    0: lambda x: np.asarray([x[0], x[1] - 0.25]),
    1: lambda x: np.asarray([x[0], x[1] + 0.25]),
    2: lambda x: np.asarray([x[0] - 0.25, x[1]]),
    3: lambda x: np.asarray([x[0] + 0.25, x[1]]),
}


def augmentTrajectoriesShopping(trajectories, subgoals, tol=0.2):
    numSubgoals = len(subgoals)
    augmented = []
    for trajectory in trajectories:
        seenSubgoals = np.zeros(numSubgoals, dtype=int)
        stepsWithProgress = []
        for step in trajectory:
            position = np.asarray(step)
            for subgoalIdx, subgoal in enumerate(subgoals):
                if np.allclose(position, subgoal, atol=tol):
                    seenSubgoals[subgoalIdx] = 1
            stateWithProgress = np.concatenate([position, seenSubgoals])
            stepsWithProgress.append(stateWithProgress)
        augmented.append(stepsWithProgress)
    return augmented


def makeLearnerShopping(theta, subgoals, initialXY, tol=0.2):
    def getNextState(state, action):
        currentPosition, progress = np.asarray(state[:2]), np.asarray(state[2:]).copy()
        nextPosition = shoppingActionMap[action](currentPosition)
        
        # Check if we reached any subgoal, but only mark it complete if it's the next one in sequence
        for subgoalIdx, subgoal in enumerate(subgoals):
            if np.allclose(subgoal, nextPosition, atol=tol):
                # Only mark this subgoal complete if all previous subgoals are already complete
                if subgoalIdx == 0 or np.all(progress[:subgoalIdx] == 1):
                    progress[subgoalIdx] = 1
        
        return np.concatenate([nextPosition, progress])
    
    def getphi(subgoals):
        def phi(state):
            pos = state[:2]
            progress = state[2:]
            
            # Find the next subgoal in sequence (first unvisited one)
            nextSubgoalIdx = None
            for i, visited in enumerate(progress):
                if visited == 0:
                    nextSubgoalIdx = i
                    break
            
            if nextSubgoalIdx is None:
                # All subgoals visited - return zeros
                return np.zeros(2 + len(subgoals))
            
            # Distance to the next subgoal only (x, y)
            nextSubgoal = subgoals[nextSubgoalIdx]
            distToNext = np.abs(nextSubgoal - pos)
            
            # One-hot encoding of which subgoal is currently being targeted
            oneHot = np.zeros(len(subgoals))
            oneHot[nextSubgoalIdx] = 1
            
            # Features: [dist_x, dist_y, one_hot_0, one_hot_1, ..., one_hot_N]
            feat = np.concatenate([distToNext, oneHot])
            return feat
        return phi
    
    def gameOver(state):
        targetProgress = state[2:]
        return np.all(targetProgress == 1)
    
    initialProgress = np.zeros(len(subgoals), dtype=int)
    initialState = np.concatenate([initialXY, initialProgress])
    phi = getphi(subgoals)
    
    learner = MaxEntropyIRL(
        theta=theta,
        actions=np.arange(4),
        probeNextState=getNextState,
        initialState=initialState,
        gameOver=gameOver,
        phi=phi,
    )
    return learner


def trainShoppingHIRL(expertTrajectories, subgoals, initialXY, tol=0.2, num_iterations=200):
    augmentedTrajectories = augmentTrajectoriesShopping(expertTrajectories, subgoals, tol=tol)
    theta_random = np.random.uniform(low=0.0, high=0.1, size=2 + len(subgoals)) # 2 coordinates for distance to next subgoal + one-hot for progress
    
    learner = makeLearnerShopping(theta_random, subgoals, initialXY, tol=tol)
    theta_hat, _ = learner.learn(augmentedTrajectories, num_iterations=num_iterations, alpha=0.05, num_samples=50)
    
    return theta_hat


def plotShoppingTrajectory(learner, subgoals, expertTrajectories):
    plt.figure(figsize=(10, 8))
    
    for i, traj in enumerate(expertTrajectories):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], 'b.-', alpha=0.2, linewidth=0.5, 
                label='Expert Trajectory' if i == 0 else "")
    
    # Generate and plot learned trajectory
    sampleTrajectory = learner.greedy_trajectory(maxLength=200, epsilon=0.05)
    
    # Save trajectory (convert numpy arrays to lists for JSON serialization)
    trajectory_to_save = [step.tolist() for step in sampleTrajectory]
    with open("generated_trajectory.json", "w") as f:
        json.dump(trajectory_to_save, f, indent=2)
    print(f"Saved generated trajectory to generated_trajectory.json")
    
    print(f"Generated a trajectory of length {len(sampleTrajectory)}:")
    
    print(f"\nFinal state: {sampleTrajectory[-1]}")
    x, y = [step[0] for step in sampleTrajectory], [step[1] for step in sampleTrajectory]
    plt.plot(x, y, 'o-', color='green', linewidth=2, markersize=4, label='Generated Trajectory')
    
    # Plot subgoals
    if len(subgoals) > 0:
        plt.scatter(subgoals[:, 0], subgoals[:, 1], c='red', s=250, marker='*', 
                   label='Subgoals', zorder=10)
        
        # Annotate subgoals
        for i, subgoal in enumerate(subgoals):
            if i < len(subgoals) - 1:
                plt.annotate(f"Subgoal {i+1}", (subgoal[0], subgoal[1]), 
                           xytext=(8, 8), textcoords='offset points', 
                           fontsize=10, color='red')
            else:
                plt.annotate("Final Goal", (subgoal[0], subgoal[1]), 
                           xytext=(8, 8), textcoords='offset points', 
                           fontsize=10, color='red', fontweight='bold')
    
    # Mark start
    if len(expertTrajectories) > 0:
        start = expertTrajectories[0][0]
        plt.scatter(start[0], start[1], c='green', s=200, marker='o', 
                   label='Start', zorder=10)
    
    plt.title("Shopping HIRL: Generated Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    output_file = "hirl_shopping_generated.png"
    plt.savefig(output_file)
    print(f"Generated trajectory visualization saved to {output_file}")
    plt.show()


learnMode = False

if __name__ == "__main__":
    noise = 0.05
    expertTrajectories = load_expert_trajectories("trajectories.pkl", noise=noise)

    segmenter = HIRLSegmenter()
    subgoals = segmenter.subgoals(expertTrajectories)

    startState = np.asarray([1.25, 15.5])

    finalGoalLocation = np.asarray([5.75, 11])
    subgoals = np.vstack([np.round(subgoals * 4) / 4, finalGoalLocation])
    # print("Subgoals:", subgoals)
    # plotSubgoals(expertTrajectories, subgoals)

    # Tolerance: 0.3 puts us within one step of the subgoal (0.25 step size)
    tol = 0.3

    if learnMode:
        theta_hat = trainShoppingHIRL(
            expertTrajectories, 
            subgoals, 
            initialXY=startState, 
            tol=tol,
            num_iterations=200
        )
        print("Learned theta:", theta_hat)
        with open("hirl-full-params-shopping.pkl", "wb") as f:
            pickle.dump((subgoals, theta_hat), f)
        print("Saved learned parameters to hirl-full-params-shopping.pkl")
        learner = makeLearnerShopping(theta_hat, subgoals, initialXY=startState, tol=tol)
        plotShoppingTrajectory(learner, subgoals, expertTrajectories)
    else:
        with open("hirl-full-params-shopping.pkl", "rb") as f:
            subgoals, theta_hat = pickle.load(f)
        print("Loaded subgoals:", subgoals)
        print("Loaded theta:", theta_hat)
        learner = makeLearnerShopping(theta_hat, subgoals, initialXY=startState, tol=tol)
        plotShoppingTrajectory(learner, subgoals, expertTrajectories)

    



