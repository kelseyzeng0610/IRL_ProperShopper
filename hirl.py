import numpy as np
from max_entropy_irl import MaxEntropyIRL
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt

class HIRLSegmenter:
    def __init__(self):
        return
    
    def labels(self, expertTrajectories, windowSize):
        windows = []
        for trajIdx in range(len(expertTrajectories)):
            trajectory = expertTrajectories[trajIdx]
            for i in range(len(trajectory) - windowSize + 1):
                window = np.asarray(trajectory[i : i+windowSize])
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
        for trajIdx in range(len(expertTrajectories)):
            trajectory = expertTrajectories[trajIdx]
            startIdx = windowIdx
            endIdx = startIdx + len(trajectory) - windowSize + 1
            trajectoryLabels = labels[startIdx:endIdx]
            for i in range(len(trajectoryLabels) - 1):
                if trajectoryLabels[i] != trajectoryLabels[i+1]:
                    transitions.append(trajectory[i+1])
            windowIdx = endIdx
        transitions = np.asarray(transitions)
        return transitions
    
    def subgoals(self, expertTrajectories, windowSize=2):
        labels = self.labels(expertTrajectories, windowSize)
        transitions = self.transitions(expertTrajectories, labels, windowSize)
        transitions = np.unique(transitions, axis=0)

        if len(transitions) == 1:
            return transitions
        
        raise Exception("not implemented for more than one transition point")
        # otherwise we have multiple subgoals
        # nComponents = min(5, len(transitions))
        # subgoalsBGM = BayesianGaussianMixture(
        #     n_components=nComponents,
        #     random_state=42
        # )
        # subgoalsBGM.fit(transitions)
        # subgoalLabels = subgoalsBGM.predict(transitions)
        # subgoals = []
        # for cluster in np.unique(subgoalLabels):
        #     points = transitions[subgoalLabels == cluster]
        #     subgoals.append(points.mean(axis=0))
        # subgoals = np.asarray(subgoals)
        # return subgoals

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
    subgoals = np.vstack([subgoals, np.asarray([2,4])])

    # subgoals = np.array([[4,0], [2,4]])
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
        
    actionMap = {
        0: lambda x: np.asarray([x[0], x[1]-1]),
        1: lambda x: np.asarray([x[0], x[1]+1]),
        2: lambda x: np.asarray([x[0]-1, x[1]]),
        3: lambda x: np.asarray([x[0]+1, x[1]])
    }
    def getNextState(state, action):
        return actionMap[action](state)

    trajectoriesToPlot = []
    # For each segment, train a MaxEntropyIRL to learn that segment
    for i in range(numSegments):
        expertSegments = trajectoriesBySegment[i]
        # expertSegments = [[(s, None, None) for s in expertseg] for expertseg in expertSegments]
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
        theta_hat, grad_norms = learner.learn(expertSegments, num_iterations=200, alpha=0.05, num_samples=50)
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



