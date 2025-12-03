import numpy as np

def softmax(vect):
    e_x = np.exp(vect - np.max(vect))
    return e_x / np.sum(e_x)

def writeProgressBar(i, n):
    progress = i / n
    barLength = 40
    bar = "#" * int(np.round(progress * barLength)) + "-" * int(np.round((1-progress) * barLength))
    print(f"\r[{bar}] {progress*100:5.1f}% ", end="", flush=True)

class MaxEntropyIRL:
    def __init__(self, theta, actions, probeNextState, initialState, gameOver, phi):
        self.theta = theta
        self.actions = actions
        self.probeNextState = probeNextState
        self.initialState = initialState
        self.gameOver = gameOver
        self.phi = phi

    def reward(self, state):
        # ps = self.phi(state)
        # d = np.dot(self.theta, ps)
        # return d
        return np.dot(self.theta, self.phi(state))

    def stochastic_trajectory(self, maxLength=None):
        currentState = self.initialState
        trajectory = [currentState]
        while not self.gameOver(currentState) and (maxLength is None or len(trajectory) < maxLength):
            q_values = []
            next_states = []
            for a in self.actions:
                ns = np.asarray(self.probeNextState(currentState, a))
                next_states.append(ns)
                q_values.append(self.reward(ns))

            prob_values = softmax(np.array(q_values))
            randomIdx = np.random.choice(len(next_states), p=prob_values)
            nextState = next_states[randomIdx]

            trajectory.append(nextState)
            currentState = nextState

        return trajectory

    def greedy_trajectory(self, maxLength=None, epsilon=0.1, recordActions=False):
        currentState = self.initialState
        trajectory = [currentState]
        visited_positions = {}  # Track visit counts per position
        actions = []
        
        while not self.gameOver(currentState) and (maxLength is None or len(trajectory) < maxLength):
            q_values = []
            next_states = []
            for a in self.actions:
                ns = np.asarray(self.probeNextState(currentState, a))
                next_states.append(ns)
                q_values.append(self.reward(ns))

            q_values = np.array(q_values)
            
            # Penalize revisiting positions too many times
            for i, ns in enumerate(next_states):
                pos_key = tuple(ns[:2])  # Only use x,y position
                visit_count = visited_positions.get(pos_key, 0)
                if visit_count > 0:
                    # Add penalty that grows with visit count
                    q_values[i] -= 0.5 * visit_count
            
            # Epsilon-greedy: sometimes take random action to escape local optima
            if np.random.random() < epsilon:
                bestIdx = np.random.choice(len(next_states))
            else:
                # Add small noise to break ties consistently  
                q_values = q_values + np.random.normal(0, 1e-5, len(q_values))
                bestIdx = np.argmax(q_values)
            
            nextState = next_states[bestIdx]
            if recordActions:
                actions.append(self.actions[bestIdx])
            
            # Track visit count
            pos_key = tuple(nextState[:2])
            visited_positions[pos_key] = visited_positions.get(pos_key, 0) + 1
            
            trajectory.append(nextState)
            currentState = nextState

        return trajectory, actions
    
    def feature_count(self, trajectory):
        features = np.array([self.phi(step) for step in trajectory])

        # only sum the x,y features
        avg_xy = np.mean(features[:, :2], axis=0)
        final_flags = features[-1, 2:]
        return np.concatenate([avg_xy, final_flags])
        # return np.sum(features, axis=0) / len(trajectory)

    def average_feature_counts(self, trajectories):
        total = sum(self.feature_count(traj) for traj in trajectories)
        return total / len(trajectories)
    

    def learn(self, expert_trajectories, num_iterations=200, num_samples=50, alpha = 0.01, maxTrajectoryLength=50):
        muhat_expert = self.average_feature_counts(expert_trajectories)
        grad_norms = []
        for iteration in range(num_iterations):
            # First sample some trajectories according to our current reward estimate (theta)
            sample_trajectories = [self.stochastic_trajectory(maxLength=maxTrajectoryLength) for _ in range(num_samples)]
            # Get the average feature counts according to our model
            muhat = self.average_feature_counts(sample_trajectories)
            # Update step: expert minus model
            grad = muhat_expert - muhat
            self.theta += alpha * grad
            grad_norms.append(np.linalg.norm(grad))
            writeProgressBar(iteration+1, num_iterations)
            

        return self.theta, grad_norms

