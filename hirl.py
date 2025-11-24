import numpy as np
from max_entropy_irl import MaxEntropyIRL
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt
import pickle

class HIRLSegmenter:
    def __init__(self):
        return
    
    def labels(self, expertTrajectories, windowSize):
        windows = []
        for trajIdx in range(len(expertTrajectories)):
            trajectory = expertTrajectories[trajIdx]
            for i in range(len(trajectory) - windowSize + 1):
                window = np.asarray(trajectory[i : i+windowSize])
                
                # TODO: switching to position-invariant version breaks the toy example
                # Use deltas (velocities) instead of absolute positions
                # allows finding changes in direction
                # window = np.diff(window, axis=0).flatten()
                window = window.flatten()
                windows.append(window)
        windows = np.asarray(windows)

        bgm = BayesianGaussianMixture(
            n_components=5,
            random_state=42,
        )
        bgm.fit(windows)
        labels = bgm.predict(windows)
        return labels
    
    def transitions(self, expertTrajectories, labels, windowSize):
        transitions = []
        windowIdx = 0
        transitionMetadata = [] # this is to store trajectoryIndex and timeIndex for each transition
        for trajIdx in range(len(expertTrajectories)):
            trajectory = expertTrajectories[trajIdx]
            startIdx = windowIdx
            endIdx = startIdx + len(trajectory) - windowSize + 1
            trajectoryLabels = labels[startIdx:endIdx]
            for i in range(len(trajectoryLabels) - 1):
                if trajectoryLabels[i] != trajectoryLabels[i+1]:
                    transitions.append(trajectory[i+1])
                    transitionMetadata.append((trajIdx, i+1))
            windowIdx = endIdx
        transitions = np.asarray(transitions)
        return transitions, transitionMetadata
    
    def subgoals(self, expertTrajectories, windowSize=2):
        labels = self.labels(expertTrajectories, windowSize)
        transitions, transitionMetadata = self.transitions(expertTrajectories, labels, windowSize)
        # transitions = np.unique(transitions, axis=0)

        if len(transitions) == 1:
            return transitions

        # otherwise we have multiple subgoals
        nComponents = min(5, len(transitions))
        subgoalsBGM = BayesianGaussianMixture(
            n_components=nComponents,
            random_state=42
        )
        subgoalsBGM.fit(transitions)
        subgoalLabels = subgoalsBGM.predict(transitions)
        subgoals = []
        for cluster in np.unique(subgoalLabels):
            points = transitions[subgoalLabels == cluster]
            subgoals.append(points.mean(axis=0))
        subgoals = np.asarray(subgoals)
        
        # Sort subgoals by when they appear in the trajectory on average
        # can do this by finding the average index of the transition points in each cluster
        # The transitions array is ordered by trajectory then by time.
        # so the order in transitions is roughly in order
        cluster_avg = []
        unique_labels = np.unique(subgoalLabels)
        for cluster in unique_labels:
            indices = np.where(subgoalLabels == cluster)[0]
            trajToIndex = {}
            for idx in indices:
                trajIdx, timeIdx = transitionMetadata[idx]
                if trajIdx not in trajToIndex:
                    trajToIndex[trajIdx] = []
                trajToIndex[trajIdx].append(timeIdx)
            avgIndexPerTraj = [np.mean(trajToIndex[traj]) for traj in trajToIndex]
            avgAcrossTraj = np.mean(avgIndexPerTraj)
            cluster_avg.append(avgAcrossTraj)
            
        # sort subgoals based on these average indices
        sorted_indices = np.argsort(cluster_avg)
        subgoals = subgoals[sorted_indices]
        
        return subgoals

actionMap = {
    0: lambda x: np.asarray([x[0], x[1]-1]),
    1: lambda x: np.asarray([x[0], x[1]+1]),
    2: lambda x: np.asarray([x[0]-1, x[1]]),
    3: lambda x: np.asarray([x[0]+1, x[1]])
}

def trainMaxEntBySubtask(expertTrajectories, subgoals):
    # split the trajectories by segment so each IRL agent can learn a single segment
    segmentedTrajectories = []
    for trajectory in expertTrajectories:
        segments = []
        subgoalIdx = 0
        subgoal = subgoals[subgoalIdx]
        segment = []
        for step in trajectory:
            if np.allclose(step, subgoal):
                segment.append(step)
                segments.append(segment)
                segment = []
                subgoalIdx += 1
                subgoal = subgoals[subgoalIdx] if subgoalIdx < len(subgoals) else None
            else:
                segment.append(step)
        segmentedTrajectories.append(segments)

    numSegments = len(segmentedTrajectories[0])
    for traj in segmentedTrajectories:
        assert len(traj) == numSegments

    trajectoriesBySegment = [[segmentedTrajectories[j][i] for j in range(len(segmentedTrajectories))] for i in range(numSegments)]

    def getNextState(state, action):
        return actionMap[action](state)

    trajectoriesToPlot = []
    # For each segment, train a MaxEntropyIRL to learn that segment
    for i in range(numSegments):
        expertSegments = trajectoriesBySegment[i]
        subgoalLocation = subgoals[i]

        initialLocation = (0, 0) if i == 0 else subgoals[i-1]

        def getphi(goal):
            def phi(state):
                return np.asarray([np.abs(goal[0] - state[0]), np.abs(goal[1] - state[1])])
            return phi
        
        def getReachedSubgoal(subgoal):
            def gameOver(state):
                return np.allclose(state, subgoal)
            return gameOver

        phi = getphi(subgoalLocation)

        theta_random = np.random.uniform(low=0.0, high=0.1, size=len(phi(initialLocation)))
        learner = MaxEntropyIRL(
            theta=theta_random,
            actions=np.arange(4),
            probeNextState=getNextState,
            initialState=initialLocation,
            gameOver=getReachedSubgoal(subgoalLocation),
            phi=phi,
        )
        theta_hat, _ = learner.learn(expertSegments, num_iterations=200, alpha=0.05, num_samples=50)
        print("For segment", i, "predicted theta:", theta_hat)
        sampleTrajectory = learner.stochastic_trajectory(maxLength=40)
        trajectoriesToPlot.append(sampleTrajectory)
    
    assert len(trajectoriesToPlot) == 2

    x1, y1 = [step[0] for step in trajectoriesToPlot[0]], [step[1] for step in trajectoriesToPlot[0]]
    x2, y2 = [step[0] for step in trajectoriesToPlot[1]], [step[1] for step in trajectoriesToPlot[1]]
    plt.plot(x1, y1, 'o-', color='blue', label='Subtask 1')
    plt.plot(x2, y2, 'o-', color='red', label='Subtask 2')
    plt.grid(True)
    plt.legend()
    plt.show()

def augmentTrajectories(trajectories, subgoals, noise):
    # add a progress vector to the trajectories
    numSubgoals = len(subgoals)
    augmented = []
    for trajectory in trajectories:
        seenSubgoals = np.zeros(numSubgoals, dtype=int)
        stepsWithProgress = []
        for step in trajectory:
            position = np.asarray(step)
            for subgoalIdx, subgoal in enumerate(subgoals):
                if np.allclose(position, subgoal, atol=noise):
                    # we reached one of the subgoals
                    seenSubgoals[subgoalIdx] = 1
            stateWithProgress = np.concatenate([position, seenSubgoals])
            stepsWithProgress.append(stateWithProgress)
        augmented.append(stepsWithProgress)
    return augmented

def makeLearner(theta, subgoals, initialXY, noise):
    def getNextState(state, action):
        # state here has the progress vector
        currentPosition, progress = np.asarray(state[:2]), np.asarray(state[2:]).copy()
        nextPosition = actionMap[action](currentPosition)
        for subgoalIdx, subgoal in enumerate(subgoals):
            if np.allclose(subgoal, nextPosition, atol=noise):
                progress[subgoalIdx] = 1
        
        return np.concatenate([nextPosition, progress])
    
    def getphi(subgoals):
        def phi(state):
            pos = state[:2]
            progress = state[2:]
            distances = np.abs(subgoals - pos).flatten()
            mask = np.repeat(1-progress, 2)
            feat = np.concatenate([distances*mask, progress])
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


def trainSingleMaxEntForFullTask(expertTrajectories, subgoals, initialXY, noise=0.0):
    augmentedTrajectories = augmentTrajectories(expertTrajectories, subgoals, noise=noise)    
    theta_random = np.random.uniform(low=0.0, high=0.1, size=3 * len(subgoals)) # 2 coordinates + 1 progress per subgoal
    learner = makeLearner(theta_random, subgoals, initialXY, noise=noise)
    theta_hat, _ = learner.learn(augmentedTrajectories, num_iterations=200, alpha=0.05, num_samples=50)

    return theta_hat

def sampleAndPlotFullHIRL(learner):
    sampleTrajectory = learner.stochastic_trajectory(maxLength=50)
    x, y = [step[0] for step in sampleTrajectory], [step[1] for step in sampleTrajectory]
    plt.plot(x, y, 'o-', color='green', label='Full Task')
    plt.grid(True)
    plt.legend()
    plt.title("Sample Trajectory with single MaxEnt IRL Agent for Full Task")
    plt.show()
    

if __name__ == "__main__":
    # Example - a 5x5 grid where you start at 0,0 and the goal is to navigate to 4,0, then 2,4
    # So it's a compound goal that cannot be learned with just a standard IRL max entropy model like we did before

    expertTrajectories = np.asarray([
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (3, 0), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (3, 4), (2, 4)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (3, 1), (3, 2), (2, 2), (2, 3), (2, 4)],
    ])

    segmenter = HIRLSegmenter()
    subgoals = segmenter.subgoals(expertTrajectories)
    subgoals = np.vstack([subgoals, np.asarray([2,4])]) # add the final goal
    print(subgoals)

    fullHIRL = True
    preLearned = False

    if fullHIRL:
        if preLearned: 
            with open("hirl-full-params.pkl", "rb") as f:
                theta_hat = pickle.load(f)
            print("Loaded pre-learned parameters for full HIRL:", theta_hat)
        else:
            theta_hat = trainSingleMaxEntForFullTask(expertTrajectories, subgoals, initialXY=np.asarray([0,0]), noise=0.0)
            print("Learned parameters for full HIRL:", theta_hat)
        learner = makeLearner(theta_hat, subgoals, initialXY=np.asarray([0,0]), noise=0.0)
        sampleAndPlotFullHIRL(learner)
    else:
        trainMaxEntBySubtask(expertTrajectories, subgoals)

    



