import numpy as np

def roundToPointTwoFive(number):
    return np.round(number * 4) / 4

class ExpertAgent:
    # here Are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.8, epsilon=0.1, mini_epsilon=0.005, decay=0.999, goals=[]):
        self.action_space = action_space 
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        if self.epsilon == 0:
            self.mini_epsilon = 0
        self.decay = decay               # value to decay the epsilon over time
        
        # We store the qtable as a 2D dictionary. 
        # First key will be a goal location, which will index to another dictionary, where the key is a state representation
        # and values will be numpy arrays of q-values (with dimension equal to the number of actions)
        self.goals = goals
        self.currentGoalIdx = 0

        qtables = {}
        for goal in goals:
            qtables[goal['type']] = {}
        self.qtables = qtables
        self.interactionSequence = []

    def setGoals(self, goals):
        qtables = {}
        for goal in goals:
            qtables[goal['type']] = {}
        self.qtables = qtables
        self.goals = goals
        self.currentGoalIdx = 0

    # This is a helper method for setting up an agent with a custom q-table (helpful for using a pretrained q-table)
    def withQTables(self, qtables={}):
        self.qtables = qtables
 
    # Wrapper method for getting a q-value. Defaults to 0 for state values not yet seen.
    def getQTableValue(self, state, action):
        qt = self.qtables[self.goals[self.currentGoalIdx]['type']]
        if state not in qt:
            return 0
        
        return qt[state][action]

    # Wrapper method for setting a q-value for a (state, action) input. 
    # For an unseen state value, will also initialize q-values for all remaining actions to 0.
    def setQTableValue(self, state, action, value):
        qt = self.qtables[self.goals[self.currentGoalIdx]['type']]
        if state not in qt:
            qt[state] = np.zeros(self.action_space)

        qt[state][action] = value
        return
    

    # randomly choose an action from the action space according to uniform distribution
    def randomAction(self):
        return np.random.randint(0, self.action_space)
    
    # Wrapper method for finding the maximum q-value for a state
    # Returns (action, value) where the first parameter is the action corresponding to the maximum q-value for the input state
    # For unseen states, randomly chooses an action
    # Handles ties in maximum q-values by letting numpy.argmax decide - this seemingly picks the first occurrence of the max q-value.
    def maxQTableValue(self, state, forbiddenActions=[]):        
        qt = self.qtables[self.goals[self.currentGoalIdx]['type']]
        if state not in qt:
            # we haven't seen this state before, so randomly pick an action
            randomAction = self.randomAction()
            while randomAction in forbiddenActions:
                randomAction = self.randomAction()
            self.setQTableValue(state, randomAction, 0)

        qValues = qt[state] # this is a np array of values for each action
        qValuesAllowed = np.copy(qValues)
        qValuesAllowed[forbiddenActions] = -np.inf # pretend the "forbidden" actions have very bad q values
        maxAction = np.argmax(qValuesAllowed)
        maxValue = qValues[maxAction]

        return maxAction, maxValue, qValues 


    def trans(self, state):
        [playerX, playerY] = state["observation"]["players"][0]["position"]        
        goalX, goalY = self.goals[self.currentGoalIdx]['position'][0], self.goals[self.currentGoalIdx]['position'][1]
        vector = (goalX - playerX, goalY - playerY)
        # hasItem = 'sausage' in state['observation']['baskets'][0]['contents']
        hasItem = False

        return (roundToPointTwoFive(vector[0]), roundToPointTwoFive(vector[1]), hasItem)
        

    # The main q-table update step. 
    def learning(self, action, rwd, game_state, next_game_state, forbiddenActions=[]):
        # implement the Q-learning function
        # Q-table gets updated at (state, action) to: (1-alpha)*Q(s,a) + alpha * [reward + gamma * next_max]
        stateRepr, nextStateRepr = self.trans(game_state), self.trans(next_game_state)
        oldValue = self.getQTableValue(stateRepr, action)

        _, nextMax, _ = self.maxQTableValue(nextStateRepr, forbiddenActions)
        newValue = (1 - self.alpha) * oldValue + self.alpha * (rwd + self.gamma * nextMax)
        self.setQTableValue(stateRepr, action, newValue)

    # Randomly with probability of `self.epsilon`, ignore the best action and choose a random one.
    # Else, look up the action with the maximum q-value for the current state and choose that action.
    def choose_action(self, gameState, forbiddenActions=[]):
        stateRepr = self.trans(gameState)
        if np.random.uniform() < self.epsilon:
            randomAction = self.randomAction()
            while randomAction in forbiddenActions:
                randomAction = self.randomAction()
            return randomAction
            
        nextAction, _, qtablevalues = self.maxQTableValue(stateRepr, forbiddenActions)

        # decay our epsilon
        decayed = self.epsilon * self.decay
        self.epsilon = max(decayed, self.mini_epsilon)
        return nextAction
    
    def achievedGoal(self):
        self.currentGoalIdx += 1

    def resetGoals(self, goals):
        self.goals = goals
        self.currentGoalIdx = 0
