import numpy as np

def softmax(vect):
    e_x = np.exp(vect - np.max(vect))
    return e_x / np.sum(e_x)

def feature_count(trajectory):
    f = [s for (s, a, sprime) in trajectory]
    return np.sum(f, axis=0) / len(trajectory)

def average_feature_counts(trajectories):
    total = sum(feature_count(traj) for traj in trajectories)
    return total / len(trajectories)

def writeProgressBar(i, n):
    progress = i / n
    barLength = 40
    bar = "#" * int(np.round(progress * barLength)) + "-" * int(np.round((1-progress) * barLength))
    print(f"\r[{bar}] {progress*100:5.1f}% ", end="", flush=True)

class MaxEntropyIRL:
    def __init__(self, theta, actions, probeNextState, initialState, gameOver, phi, maxTrajectoryLength=50):
        self.theta = theta
        self.actions = actions
        self.probeNextState = probeNextState
        self.initialState = initialState
        self.maxTrajectoryLength = maxTrajectoryLength
        self.gameOver = gameOver
        self.phi = phi

    def reward(self, state):
        return np.dot(self.theta, state)

    def stochastic_trajectory(self, maxLength=None):
        currentState = self.initialState
        trajectory = []
        while not self.gameOver(currentState) and (maxLength is None or len(trajectory) < maxLength):
            q_values = []
            next_states = []
            for a in self.actions:
                ns = self.probeNextState(currentState, a)
                next_states.append(ns)
                q_values.append(self.reward(ns))
            prob_values = softmax(np.array(q_values))

            randomIdx = np.random.choice(len(next_states), p=prob_values)
            nextState = next_states[randomIdx]
            trajectory.append([currentState, randomIdx, nextState])
            currentState = nextState

        return trajectory
    

    def learn(self, expert_trajectories, num_iterations=200, num_samples=50, alpha = 0.01):
        muhat_expert = average_feature_counts(expert_trajectories)
        grad_norms, thetas = [], []
        for iteration in range(num_iterations):
            thetas.append(self.theta)
            # First sample some trajectories according to our current reward estimate (theta)
            sample_trajectories = [self.stochastic_trajectory(maxLength=self.maxTrajectoryLength) for m in range(num_samples)]
            # Get the average feature counts according to our model
            muhat = average_feature_counts(sample_trajectories)
            # Update step: expert minus model
            grad = muhat_expert - muhat
            self.theta += alpha * grad
            grad_norms.append(np.linalg.norm(grad))
            writeProgressBar(iteration+1, num_iterations)
            

        return self.theta, grad_norms, thetas
