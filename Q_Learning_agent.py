import numpy as np

def roundToPointTwoFive(number):
    return np.round(number * 4) / 4

class QLAgent:
    # here are some default parameters, you can use different ones
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


    def trans(self, state, granularity=0.25):
        [playerX, playerY] = state["observation"]["players"][0]["position"]
        currentGoalType = self.goals[self.currentGoalIdx]['type']

        # for some goals we represent the state as just the direction, because here the goal is simply to perform an action regardless of our position
        if currentGoalType in ["INTERACT", "TOGGLE_CART", "PICKUP_ITEM"]:
            return state['observation']['players'][0]['direction']
        
        goalX, goalY = self.goals[self.currentGoalIdx]['position'][0], self.goals[self.currentGoalIdx]['position'][1]
        vector = (goalX - playerX, goalY - playerY)

        # For some goals we only care about moving in one direction, so to speed up learning / generalize between situations, just pick the x value or y value of the relative vector
        # Example: in an aisle, moving to a shelf in only the x direction. regardless of our y position we can treat these states the same
        if currentGoalType in ["WALKWAY", "SHELF_NAV", "EAST_WALKWAY"]:
            return roundToPointTwoFive(vector[0])
        elif currentGoalType in ["AISLE", "LEAVE_COUNTERS", "COUNTER_NAV"]:
            return roundToPointTwoFive(vector[1])
        
        return (roundToPointTwoFive(vector[0]), roundToPointTwoFive(vector[1]))
        

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

    # Manually handle the interactions of taking items from the shelf or the counter
    def setInteractionQueue(self, basketMode, counterMode=False):
        if not counterMode:
            # taking an item off the shelf
            if len(self.interactionSequence) == 0:
                seq = []
                if basketMode:
                    seq = ['APPROACH_SHELF', 'INTERACT', 'INTERACT']
                else:
                    seq = ['TOGGLE_CART', 'APPROACH_SHELF', 'INTERACT', 'INTERACT', 'APPROACH_CART', 'INTERACT', 'INTERACT', 'TOGGLE_CART']
                seq.reverse()
                self.interactionSequence = seq
        else:
            # interacting with one of the counters
            if len(self.interactionSequence) == 0:
                seq = []
                if basketMode:
                    seq = ['APPROACH_COUNTER', 'INTERACT', 'INTERACT']
                else:
                    seq = ['FACE_UP', 'TOGGLE_CART', 'APPROACH_COUNTER', 'INTERACT', 'INTERACT', 'APPROACH_CART', 'INTERACT', 'INTERACT', 'TOGGLE_CART']
                seq.reverse()
                self.interactionSequence = seq

    def getInteraction(self, state, shelfGoal):
        if self.interactionSequence[-1] == "APPROACH_SHELF":
            # check if the state is more than some threshold away from the shelf - if so, get closer
            shelfPosition, playerPosition = shelfGoal['position'], state['observation']['players'][0]['position']
            yDist = np.abs(shelfPosition[1] - playerPosition[1])
            if yDist > 0.25:
                return "NORTH"
            else:
                # we are close enough now
                self.interactionSequence.pop()
                # if we aren't facing north, we still need to, since we are below the shelf
                if state['observation']['players'][0]['direction'] != 0:
                    return "NORTH"
        elif self.interactionSequence[-1] == "APPROACH_CART":
            cartPosition, playerPosition = state['observation']['carts'][0]['position'], state['observation']['players'][0]['position']
            yDist = np.abs(cartPosition[1] - playerPosition[1])
            if yDist > 0.25:
                return "SOUTH"
            else:
                # we are close enough now, just turn to face the cart 
                self.interactionSequence.pop()
                if state['observation']['carts'][0]['position'][0] > state['observation']['players'][0]['position'][0]:
                    return "EAST"
                else:
                    return "WEST"
            
        action = self.interactionSequence.pop()
        return action
    
    def getInteractionCounter(self, state, counterGoal):
        if self.interactionSequence[-1] == "FACE_UP": # facing up moves the cart out of the way
            self.interactionSequence.pop()
            if state['observation']['players'][0]['direction'] != 0:
                return "NORTH"
            else:
                # continue to the next move
                action = self.interactionSequence.pop()
                return action
        elif self.interactionSequence[-1] == "APPROACH_COUNTER":
            counterPosition, playerPosition = counterGoal['position'], state['observation']['players'][0]['position']
            xDist = np.abs(counterPosition[0] - playerPosition[0])
            if xDist > 0.25:
                return "EAST"
            else:
                # we are close enough now
                self.interactionSequence.pop()
                action = self.interactionSequence.pop()
                return action
        elif self.interactionSequence[-1] == "APPROACH_CART":
            cartPosition, playerPosition = state['observation']['carts'][0]['position'], state['observation']['players'][0]['position']
            xDist = np.abs(cartPosition[0] - playerPosition[0])
            if xDist > 0.25:
                return "WEST"
            else:
                # we are close enough now, just turn to face the cart 
                self.interactionSequence.pop()
                return "NORTH"
        
        action = self.interactionSequence.pop()
        return action

    # Manually handle the cashier interaction
    def setPurchaseQueue(self, basketMode):
        if len(self.interactionSequence) == 0:
            seq = []
            if basketMode:
                seq = ['APPROACH_REGISTER', 'INTERACT', 'INTERACT']
            else:
                seq = ['TOGGLE_CART', 'APPROACH_REGISTER', 'INTERACT', 'INTERACT', 'APPROACH_CART', 'TOGGLE_CART']
            seq.reverse()
            self.interactionSequence = seq

    def getInteractionPurchase(self, state, payGoal):
        if self.interactionSequence[-1] == "APPROACH_REGISTER":
            payPosition, playerPosition = payGoal['position'], state['observation']['players'][0]['position']
            yDist = np.abs(payPosition[1] - playerPosition[1])
            if yDist > 0.25:
                return "SOUTH"
            else:
                # we are close enough
                self.interactionSequence.pop()
                # if we aren't facing south we still need to
                if state['observation']['players'][0]['direction'] != 1:
                    return "SOUTH"
        elif self.interactionSequence[-1] == "APPROACH_CART":
            cartPosition, playerPosition = state['observation']['carts'][0]['position'], state['observation']['players'][0]['position']
            yDist = np.abs(cartPosition[1] - playerPosition[1])
            if yDist > 0.25:
                return "NORTH"
            else:
                # we are close enough now, just turn to face the cart
                self.interactionSequence.pop()
                if state['observation']['carts'][0]['position'][0] > state['observation']['players'][0]['position'][0]:
                    return "EAST"
                else:
                    return "WEST"
        
        action = self.interactionSequence.pop()
        return action
