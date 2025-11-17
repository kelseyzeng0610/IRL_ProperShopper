import numpy as np
from max_entropy_irl import MaxEntropyIRL
from sklearn.mixture import BayesianGaussianMixture

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
        return subgoals

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

    # TODO: now need to segment trajectories by subgoals returned, and call MaxEntropyIRL on each of the segments

    print(subgoals)


