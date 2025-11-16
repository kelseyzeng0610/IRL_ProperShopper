#Author Hang Yu

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from IRLExpertAgent import ExpertAgent
import pickle
import numpy as np
import time
from enum import Enum


# Random functional utils
def some(l, f):
    return len([x for x in l if f(x)]) > 0
def find(l, f):
    r = [x for x in l if f(x)]
    return None if len(r) == 0 else r[0]

cart = False
exit_pos = [-0.8, 15.6] # The position of the exit in the environment from [-0.8, 15.6] in x, and y = 15.6
cart_pos_left = [1, 18.5] # The position of the cart in the environment from [1, 2] in x, and y = 18.5
cart_pos_right = [2, 18.5] 
# can penalize for running into something

class Direction(Enum):
    NONE = 0,
    NORTH = 1,
    SOUTH = 2,
    EAST = 3,
    WEST = 4

INT_TO_DIRECTION = {
    0: Direction.NORTH, 
    1: Direction.SOUTH, 
    2: Direction.EAST, 
    3: Direction.WEST
}
DIRECTION_TO_INT = {
    Direction.NORTH: 0,
    Direction.SOUTH: 1,
    Direction.EAST: 2,
    Direction.WEST: 3,
}

REWARD_DIST_THRESHOLD = 0.5

def can_interact_default(obj:dict, player:dict, range=0.5):
    if INT_TO_DIRECTION[player['direction']] == Direction.NORTH:
        return collision(obj, player, player['position'][0], player['position'][1] - range)
    elif INT_TO_DIRECTION[player['direction']] == Direction.SOUTH:
        return collision(obj, player, player['position'][0], player['position'][1] + range)
    elif INT_TO_DIRECTION[player['direction']] == Direction.WEST:
        return collision(obj, player, player['position'][0] - range, player['position'][1])
    elif INT_TO_DIRECTION[player['direction']] == Direction.EAST:
        return collision(obj, player, player['position'][0] + range, player['position'][1])
    
    return False

def collision(obj1:dict, obj2:dict, x_position, y_position):
    return overlap(obj1['position'][0], obj1['position'][1], obj1['width'], obj1['height'],x_position, y_position, obj2['width'], obj2['height'])

def overlap(x1, y1, width_1, height_1, x2, y2, width_2, height_2):
    return not (x1 > x2 + width_2 or x2 > x1 + width_1 or y1 > y2 + height_2 or y2 > y1 + height_1)

def get_position(state):
    return state['observation']['players'][0]['position']

def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def hasCart(state):
    return state['observation']['players'][0]['curr_cart'] >= 0

def hasBasket(state):
    return some(state['observation']['baskets'], lambda basket: basket['owner'] == 0)

def basketHasItem(state, item):
    return some(state['observation']['baskets'], lambda basket: basket['owner'] == 0 and item in basket['contents'])

def getPlayerDirection(state):
    return INT_TO_DIRECTION[state['observation']['players'][0]['direction']]


# This function gets the ideal location to be facing from agent_position with a target of target_position
# This is an "x-first" goal, meaning until we are close in the x-coordinate, we prioritize moving in the x-direction.
# This also accepts an xOnly parameter, which means we will never recommend moving in the north/south directions.
def get_direction_to_target_x_first(agent_position, target_position, xOnly=False, threshold=REWARD_DIST_THRESHOLD):
    deltaX, deltaY = np.round(target_position[0] - agent_position[0], 2), np.round(target_position[1] - agent_position[1], 2)

    if np.abs(deltaX) >= threshold or np.abs(deltaY) <= 0.1 or xOnly:
        if deltaX > 0:
            return 2 # EAST
        else: 
            return 3 # WEST
    else:
        if deltaY > 0:
            return 1 # SOUTH
        else:
            return 0 # NORTH
        
# Analogous to x-first direction, but emphasizing moves in the y-direction first.
def get_direction_to_target_y_first(agent_position, target_position, threshold=REWARD_DIST_THRESHOLD):
    deltaX, deltaY = target_position[0] - agent_position[0], target_position[1] - agent_position[1]
    if np.abs(deltaY) >= threshold or np.abs(deltaX) <= 0.1:
        if deltaY > 0:
            return 1 # SOUTH
        else: 
            return 0 # NORTH
    else:
        if deltaX > 0:
            return 2 # EAST
        else:
            return 3 # WEST
        

# Preloads a qtable saved at filePath.
def preloadQTable(filePath):
    with open(filePath, 'r') as file:
        jsonifiedQTable = json.load(file)
        qtables = {}
        for goalName in jsonifiedQTable:
            qtable = {}
            for k, v in jsonifiedQTable[goalName].items(): 
                klist = json.loads(k)
                x = np.float64(klist[0])
                y = np.float64(klist[1])
                direction = [np.int64(klist[i]) for i in range(2, len(klist))]
                keyTuple = tuple([x, y] + direction)
                qtable[keyTuple] = np.asarray(v)
            qtables[goalName] = qtable

        return qtables

# dumps a provided q table to a file. we have to serialize both the key tuples and the value ndarrays before we can write to json.
def writeQTableToFile(savedQTable, filePath):
    jsonifiedTable = {goal_location: {json.dumps((k[0], k[1], int(k[2]), int(k[3]), int(k[4]), int(k[5]), int(k[6]))): savedQTable[goal_location][k].tolist() for k in savedQTable[goal_location]} for goal_location in savedQTable}
    with open(filePath, 'w') as file:
        json.dump(jsonifiedTable, file)


def jsonSerializeTrajectoryStep(step):
    # Trajectory is (s, a, s')
    s, a, sprime = step[0], step[1], step[2]

    jsonS = [float(s[0]), float(s[1]), int(s[2]), int(s[3]), int(s[4]), int(s[5]), int(s[6])]
    jsonSPrime = [float(sprime[0]), float(sprime[1]), int(sprime[2]), int(sprime[3]), int(sprime[4]), int(sprime[5]), int(sprime[6])]
    jsonAction = int(a)
    return [
        jsonS,
        jsonAction,
        jsonSPrime,
    ]


def writeTrajectoriesToFile(trajectories, filePath):
    with open(filePath, 'wb') as f:
        pickle.dump(trajectories, f)
    with open("trajectories.json", "w") as f:
        json.dump([[jsonSerializeTrajectoryStep(step) for step in traj] for traj in trajectories], f, indent=2)

# this is a function that returns True/False of whether we are in the "walkway"
# I have defined this as being between 4 and 4.7 in the agent's x-position
def walkwayAchieved(state):
    position = get_position(state)
    return np.round(position[0], decimals=2) >= 4 and np.round(position[0], decimals=2) <= 4.7

def eastWalkwayAchieved(state):
    position = get_position(state)
    return np.round(position[0], 2) >= 16.5 and np.round(position[0], 2) <= 17.5

def pickupCartAchieved(state):
    return hasCart(state)

def pickupBasketAchieved(state):
    return hasBasket(state)

def dropCartAchieved(state):
    return not hasCart(state)

def registerAchieved(state):
    position = get_position(state)
    return np.round(position[0], decimals=2) >= 2 and np.round(position[0], decimals=2) <= 2.9 and np.round(position[1], 2) >= 3 and np.round(position[1], 2) <= 4

def exitAchieved(state):
    position = get_position(state)
    return position[0] <= 0

def getFoodInCartAchieved(shelfName, basketMode, counterMode=False):
    if basketMode:
        return lambda state: hasBasket(state) and basketHasItem(state, shelfName)
    return lambda state: hasCart(state) and shelfName in state['observation']['carts'][0]['contents']

def leaveCounterAchieved(state):
    position = get_position(state)
    return np.round(position[0], 2) <= 17.5 and np.round(position[0], 2) >= 16.5 and np.round(position[1], 2) >= 3 and np.round(position[1], 2) <= 4

def getCounterAchieved(yTarget):
    def inRange(state):
        position = get_position(state)
        return position[0] >= 16.5 and position[1] >= yTarget[0] and position[1] <= yTarget[1]
    return inRange

def getPaid(basketMode):
    if basketMode:
        return lambda state: hasBasket(state) and len(state['observation']['baskets'][0]['contents']) == 0 and len(state['observation']['baskets'][0]['purchased_contents']) > 0
    return lambda state: hasCart(state) and len(state['observation']['carts'][0]['contents']) == 0 and len(state['observation']['carts'][0]['purchased_contents']) > 0

# This constructs an aisle goal object and a shelf goal object for the agent to navigate to.
# We look up the shelf name in a json file which contains a sample state object, and once we find the shelf, we construct the aisle and shelf goal locations based on the shelf position.
def lookupAisleShelfGoal(shelfName, basketMode):
    startState = {}
    with open('sample-start-state.json', 'r') as f:
        startState = json.load(f)
    
    shelfEntranceBuffer = 0.5
    for possibleShelf in startState['shelves']:
        if possibleShelf['food'] == shelfName:
            shelfPosition, shelfHeight, shelfWidth = possibleShelf['position'], possibleShelf['height'], possibleShelf['width']
            # We first construct an "aisle entrance" goal position. This helps the agent navigate through the walkway up/down until we reach the correct aisle for the shelf.
            # We define the aisle entrance's position as always having an x-coordinate of 5.5, and having a y-coordinate equal to the shelf position plus its height (since the shelf position is the top left of the shelf) plus a buffer
            aislePosition = (5.5, shelfPosition[1] + shelfHeight + shelfEntranceBuffer)
            aisleEntranceGoal = {
                "name": "aisle for " + shelfName,
                "type": "AISLE", 
                "original_position": aislePosition,
                "position": aislePosition,
                "height": shelfHeight,
                "width": shelfWidth,
                "threshold": (REWARD_DIST_THRESHOLD, 0.25)
            }

            # We then build a goal position for the actual shelf. This is defined as the position plus half the shelf width (this puts us in the center of the shelf width-wise since we start from the top left corner)
            # and a y-coordinate of the y position plus the height. So essentially our target location is the bottom edge of the shelf, halfway between the two sides.
            # From that position we will later calculate the agent's distance to determine our reward function
            shelfTargetPosition = (shelfPosition[0] + 0.5 * shelfWidth, shelfPosition[1] + shelfHeight)
            shelfGoal = {
                "name": "shelf for " + shelfName,
                "type": "SHELF_NAV",
                "original_position": (shelfPosition[0], shelfPosition[1]),
                "position": shelfTargetPosition,
                "height": shelfHeight,
                "width": shelfWidth,
            }

            shelfInteractGoal = {
                "name": "take item from " + shelfName,
                "type": "PICKUP_ITEM",
                "position": shelfTargetPosition,
                "height": 0.5, 
                "width": 0.5,
                "achieved": getFoodInCartAchieved(shelfName, basketMode)
            }

            return [aisleEntranceGoal, shelfGoal, shelfInteractGoal]
        
    raise Exception("Unknown shelf name chosen")

def getShelfLocation(state, shelfName):
    shelf = find(state['observation']['shelves'], lambda shelf: shelf['food'] == shelfName)
    return shelf['position'] if shelf is not None else None

def buildShoppingList(state):
    shoppingList = []
    items, quantities = state['observation']['players'][0]['shopping_list'], state['observation']['players'][0]['list_quant']
    if len(items) != len(quantities):
        raise Exception("Invalid state provided - mismatching shopping list & quantities")
    
    counters = []
    for i in range(len(items)):
        item, quantity = items[i], quantities[i]
        if item == 'fresh fish' or item == 'prepared foods':
            counters.append({'item': item, 'quantity': quantity})
        else:
            shoppingList.append({"item": item, "quantity": quantity})

    def getkey(entry):
        shelfLocation = getShelfLocation(state, entry['item'])
        return (shelfLocation[1], -1 * shelfLocation[0])
    
    # Visit the aisles from bottom to top, and the shelves left to right
    sortedShelfItems = sorted(shoppingList, key=getkey, reverse=True)

    # now add the fresh fish and prepared foods last, if applicable
    sortedCounters = sorted(counters, key=lambda x: x['item'], reverse=True)
    return sortedShelfItems + sortedCounters

# Much like regular shelves, to visit a counter we first navigate to the east walkway, then up/down to the counter, and finally handle the interaction
def buildCounterGoals(state, item, basketMode):
    counter = find(state['observation']['counters'], lambda counter: counter['food'] == item)
    counterPosition = counter['position']
    counterTargetPosition = (counterPosition[0] - 0.5, np.round(counterPosition[1] + counter['height'] * 0.5, 2))
    yRange = (4.75, 6.5) if item == 'prepared foods' else (10.75, 12.5)
    counterNavGoal = {
        "name": "counter for " + item,
        "type": "COUNTER_NAV",
        "original_position": (counterPosition[0], counterPosition[1]),
        "position": counterTargetPosition,
        "height": 0.5,
        "width": 0.5,
        "achieved": getCounterAchieved(yRange)
    }
    counterPickupItemGoal = {
        "name": "pickup item from counter " + item,
        "type": "PICKUP_COUNTER",
        "position": counterTargetPosition,
        "height": 0.5,
        "width": 0.5,
        "achieved": getFoodInCartAchieved(item, basketMode, counterMode=True)
    }

    return [counterNavGoal, counterPickupItemGoal]


# This function builds a list of goal objects, that represents the sequence of navigations the agent needs to take.
# We accept a list of shelf names, and automatically assume we must first navigate to the cart return, then after all the shelves, to the register.
def buildGoals(state):
    # These are from the standard json observation output
    cart_return = {"name": "CART_RETURN", "type": "CART_RETURN", "original_position": (2, 18.5), "position": (2, 18.5), "height": 6, "width": 0.7}
    basket_return = {"name": "BASKET_RETURN", "type": "BASKET_RETURN", "original_position": (3.5, 18.5), "position": (3.5, 18.5), "height": 0.2, "width": 0.3}

    register = {"name": "REGISTER", "type": "REGISTER", "original_position": (2.5, 3.5), "position": (2.5, 3.5), "height": 1, "width": 1, 'achieved': registerAchieved}
    exit = {"name": "EXIT", "type": "EXIT", "original_position": (-0.5, 3), "position": (-0.5, 3), "height": 0.5, "width": 0.5, 'achieved': exitAchieved}

    # This is a custom defined, fake "location" that represents being in the main aisle/walkway
    # We will inject this between aisles so that the agent learns to leave the aisle before attempting to reach another shelf
    walkway = {"name": "WALKWAY", "type": "WALKWAY", "original_position": (4.5, 14), "position": (4.5, 14), "height": 0.5, "width": 0.5, "achieved": walkwayAchieved}
    eastWalkway = {"name": "EAST WALKWAY", "type": "EAST_WALKWAY", "original_position": (17, 9), "position": (17, 9), "height": 0.5, "width": 0.5, "achieved": eastWalkwayAchieved}

    # always start the sequence with cart return, and picking up the cart
    pickupCart = {"name": "pickup cart", "type": "INTERACT", "achieved": pickupCartAchieved}
    pickupBasket = {"name": "pickup basket", "type": "INTERACT", "achieved": pickupBasketAchieved}
    
    # shoppingList = buildShoppingList(state)
    # TODO: at first, just use the same item every time
    shoppingList = [{'item': 'sausage', 'quantity': 1}]
    print("--- Shopping List ---\n", [item['item'] for item in shoppingList])
    
    goals = []
    totalQuant = np.sum([item['quantity'] for item in shoppingList])

    # basketMode = totalQuant < 6
    basketMode = True # TODO: for initial work, just use the basket

    # revert this
    # if basketMode:
    #     goals = [basket_return, pickupBasket]
    # else:
    #     goals = [cart_return, pickupCart]

    # At each shelf name we process, check if the previous shelf is in the same aisle
    prevAisle = None
    for i in range(len(shoppingList)):
        item, quantity = shoppingList[i]['item'], shoppingList[i]['quantity']

        if item == "prepared foods" or item == "fresh fish":
            [counterNav, pickupCounterItem] = buildCounterGoals(state, item, basketMode)
            goals.append(eastWalkway)
            goals.append(counterNav)
            for n in range(quantity):
                goals.append(pickupCounterItem)
            continue
            

        [aisle, shelfNav, pickupItem] = lookupAisleShelfGoal(item, basketMode)
        # so we have a goal for navigating to the item and picking it off the shelf
        # we can repeat this n times

        if prevAisle is not None and prevAisle == aisle['original_position'][1]:
            # if this is not the first shelf, and the last shelf we requested has the same y-coordinate,
            # we assume they are the same aisle. Therefore, from the previous shelf, we can directly navigate to the next shelf. 
            goals.append(shelfNav)
        else:
            # Otherwise, we are either visiting the first shelf or a shelf that is in a different aisle.
            # In that case, we must first navigate to the walkway. Then we navigate to the aisle containing our shelf, and finally to the shelf itself.
            goals.append(walkway)
            goals.append(aisle)
            goals.append(shelfNav)

        # TODO: don't pick up items yet
        # for n in range(quantity):
        #     goals.append(pickupItem)

        prevAisle = aisle['original_position'][1]

    # TODO: skip everything else for now
    return goals, basketMode
    
    # At the end of all the shelves, we navigate back to the walkway, and then to the register.
    # check the last goal before this, and if it's a counter, then add another checkpoint which gets us out of the way of the shelves so we can easily approach the register and exit
    if goals[-1]['type'] == "PICKUP_COUNTER":
        goals.append({
            "name": "leave counters",
            "type": "LEAVE_COUNTERS",
            "position": (17, 3.5),
            "height": 0.5,
            "width": 0.5,
            "achieved": leaveCounterAchieved,
        })
    goals.append(walkway)
    goals.append(register)
    
    pay = {"name": "PAY", "type": "PAY", "position": (2.5, 4), "height": 0.5, "width": 0.5, "achieved": getPaid(basketMode)}
    goals.append(pay)
    goals.append(exit)

    return goals, basketMode

# for certain goals we always want to punish certain actions
def wrongAction(current_goal, action):
    match current_goal['type']:
        case "INTERACT":
            return action == "TOGGLE_CART"
        case "TOGGLE_CART":
            return action == "INTERACT"
        case _:
            return action == "INTERACT" or action == "TOGGLE_CART"

# This is a function that determines the reward for a state change, such that to navigate to our current goal,
# we want to primarily move in the x-direction first. This can be for a few reasons, such as we are in the middle of 
# an aisle, and therefore we want to get out of the aisle before moving north/south.
# This also accepts a 'xOnly' param, which means we only consider the x-direction.
def x_first_reward(previous_state, current_state, current_goal, action, xOnly=False, threshold=REWARD_DIST_THRESHOLD):
    # if we interacted or toggled the cart, then penalize - we only want to move
    if wrongAction(current_goal, action):
        return -1

    # If x is more than `threshold` away from the goal's x position, then reward only for moving in the right x direction or for facing it
    previousPosition, currentPosition = get_position(previous_state), get_position(current_state)
    goal_position = current_goal["position"]
    currentDirection = current_state['observation']['players'][0]['direction']

    currentXDist, prevXDist = np.abs(goal_position[0] - currentPosition[0]), np.abs(goal_position[0] - previousPosition[0])
    currentYDist, prevYDist = np.abs(goal_position[1] - currentPosition[1]), np.abs(goal_position[1] - previousPosition[1])
    if currentXDist >= threshold or xOnly:
        idealDirection = get_direction_to_target_x_first(currentPosition, goal_position, xOnly)
        # reward only for facing / moving towards the target position in the x direction
        if currentXDist < prevXDist or currentDirection == idealDirection:
            return 1
        else:
            return -1
    else:
        # reward for facing / moving towards the target in the y direction
        idealDirection = get_direction_to_target_y_first(currentPosition, goal_position)
        if currentYDist < prevYDist or currentDirection == idealDirection:
            return 1
        else:
            return -1

# This is analogous to the x-direction version of this, but for y first.
def y_first_reward(previous_state, current_state, current_goal, action, threshold=REWARD_DIST_THRESHOLD):
    # if we interacted or toggled the cart, then penalize - we only want to move
    if wrongAction(current_goal, action): 
        return -1
    
    # If y is more than `threshold` away from the goal's y position, then reward only for moving in the right y direction or facing it
    previousPosition, currentPosition = get_position(previous_state), get_position(current_state)
    goal_position = current_goal["position"]
    currentDirection = current_state['observation']['players'][0]['direction']

    currentXDist, prevXDist = np.abs(goal_position[0] - currentPosition[0]), np.abs(goal_position[0] - previousPosition[0])
    currentYDist, prevYDist = np.abs(goal_position[1] - currentPosition[1]), np.abs(goal_position[1] - previousPosition[1])
    if currentYDist >= threshold:
        idealDirection = get_direction_to_target_y_first(currentPosition, goal_position, threshold=threshold)
        # reward only for facing / moving towards the target position in the y direction
        if currentYDist < prevYDist or currentDirection == idealDirection:
            return 1
        else:
            return -1
    else:
        # reward for facing / moving towards the target in the x direction
        idealDirection = get_direction_to_target_x_first(currentPosition, goal_position)
        if currentXDist < prevXDist or currentDirection == idealDirection:
            return 1
        else:
            return -1        

# Our reward function for moving to the cart emphasizes moving in the x direction first
def reward_cart(previous_state, current_state, current_goal, action):
    return x_first_reward(previous_state, current_state, current_goal, action)

# Our reward function for moving to an aisle (from the walkway) emphasizes moving in the y direction first
def reward_aisle(previous_state, current_state, current_goal, action):
    return y_first_reward(previous_state, current_state, current_goal, action, threshold=0.25)

# Our reward function for moving to a shelf (from the aisle entrance) emphasizes moving in the x direction first
def reward_shelf(previous_state, current_state, current_goal, action):
    return y_first_reward(previous_state, current_state, current_goal, action, threshold=1.25)

# Our reward function for moving to the register (from the walkway) emphasizes moving in the y direction first
def reward_register(previous_state, current_state, current_goal, action):
    return y_first_reward(previous_state, current_state, current_goal, action)

# Our reward function for moving to the walkway (from the cart, or from an aisle) only rewards for moving in the x-direction
def reward_walkway(previous_state, current_state, current_goal, action):
    return x_first_reward(previous_state, current_state, current_goal, action, xOnly=True)

def reward_east_walkway(previous_state, current_state, current_goal, action):
    return x_first_reward(previous_state, current_state, current_goal, action, xOnly=True)

def reward_pickup_cart_from_rack(previous_state, current_state, current_goal, action):
    # we already know we didn't pick up the item
    if action == "INTERACT":
        if getPlayerDirection(current_state) == Direction.SOUTH:
            return 1
        else: 
            return -1
    else:
        return -1
    

# This is a custom (stricter) version of can_interact_default, which requires us to be in a smaller bounded region
# for interacting with the shelf. The can_interact_default returns True in some positions that are quite far away from 
# the shelf, and I am not convinced we can actually interact with the shelf from there.
def can_interact_shelf(goal, player, requireFacing):
    # the center of our interaction zone for a shelf is the x-center (x position + half width, since we start from top left)
    # and the bottom edge (y position + height)
    obj_center_interaction = (goal['position'][0] + 0.5 * goal['width'], goal['position'][1] + goal['height'])
    # the acceptable x range is the center above +/- half the width
    xRange = (obj_center_interaction[0] - 0.25 * goal['width'], obj_center_interaction[0] + 0.25 * goal['width'])
    # the acceptable y range is from the bottom/center edge up to 0.5 away.
    yRange = (obj_center_interaction[1], obj_center_interaction[1] + 1.25)

    # we can interact if the player's center position (disregarding their width/height) are within the x and y bounds we defined above
    inXRange = player['position'][0] >= xRange[0] and player['position'][0] <= xRange[1]
    inYRange = player['position'][1] >= yRange[0] and player['position'][1] <= yRange[1]

    # and require that the agent is facing up
    if requireFacing:
        return inXRange and inYRange and player['direction'] == 0

    return inXRange and inYRange

def can_interact_modified(goal, player):
    match goal['type']:
        case "CART_RETURN" | "BASKET_RETURN":
            return can_interact_default(goal, player)
        case "AISLE" | "COUNTER_NAV":
            return can_interact_default(goal, player)
        case "SHELF_NAV":
            return can_interact_shelf(goal, player, False)
        case "REGISTER":
            return can_interact_default(goal, player)
        case "WALKWAY" | "EAST_WALKWAY":
            return False

    raise Exception("Should not get here")

# Since we massaged object positions for navigation, we keep track of the object's original position from the state json.
# this is so we pass in the true position to can_interact_default instead of our created position for navigation.
def can_interact(current_goal, player):
    # We need the original object location for the can_interact_default to work properly
    goal_copy = {k: current_goal[k] for k in current_goal}
    goal_copy['position'] = current_goal['original_position']
    return can_interact_modified(goal_copy, player)

# We achieved our goal if we can interact with the object, or if the goal we are trying to satisfy
# has a custom condition for when we've achieved it (this is only for the walkway goal as of now) and we meet that condition
def achievedGoal(state, current_goal):
    if current_goal['type'] == "INTERACT":
        return current_goal["achieved"](state)

    achievedFuncResult = False
    if "achieved" in current_goal:
        achievedFuncResult = current_goal["achieved"](state)
        return achievedFuncResult
    return can_interact(current_goal, state['observation']['players'][0]) or achievedFuncResult

# Reward function which changes based on the goal
def calculate_reward(previous_state, current_state, current_goal, action, basketMode):
    if achievedGoal(current_state, current_goal):
        return 10
    
    # big penalty for a violation
    current_violations = current_state['violations']
    if len(current_violations) > 0:
        return -10
    
    droppedCart = hasCart(previous_state) and not hasCart(current_state)
    droppedBasket = hasBasket(previous_state) and action == "TOGGLE_CART"
    if basketMode and droppedBasket:
        current_state['gameOver'] = True
        return -10
    
    # if we dropped the cart and we weren't supposed to (we would have hit achieved already), big penalty
    if not basketMode and droppedCart and current_goal['type'] not in ["TOGGLE_CART", "PICKUP_ITEM"]:
        current_state['gameOver'] = True
        return -10

    if droppedCart and current_state['observation']['players'][0]['direction'] == 0 and current_goal['type'] == "PICKUP_ITEM":
        current_state['gameOver'] = True
        return -10 
    
    # depending on our goal, different reward logic
    match current_goal['type']:
        case "CART_RETURN" | "BASKET_RETURN":
            return reward_cart(previous_state, current_state, current_goal, action)
        case "AISLE":
            return reward_aisle(previous_state, current_state, current_goal, action)
        case "SHELF_NAV":
            return reward_shelf(previous_state, current_state, current_goal, action)
        case "REGISTER":
            return reward_register(previous_state, current_state, current_goal, action)
        case "WALKWAY":
            return reward_walkway(previous_state, current_state, current_goal, action)
        case "EAST_WALKWAY":
            return reward_east_walkway(previous_state, current_state, current_goal, action)
        case "INTERACT":
            return -1 # when we want to interact we should hit the 'achieved' lambda function, otherwise we didn't achieve it and we don't care about what we actually did, just penalize
        case "EXIT":
            return x_first_reward(previous_state, current_state, current_goal, action, True)
        case "COUNTER_NAV":
            return y_first_reward(previous_state, current_state, current_goal, action)
        case "LEAVE_COUNTERS":
            return y_first_reward(previous_state, current_state, current_goal, action)
    

    raise Exception("shouldnt get here")


def getAction(agent, state, action_commands):
    action_index = agent.choose_action(state)
    action = "0 " + action_commands[action_index]

    return action, action_index

if __name__ == "__main__":
    action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST']
    # Initialize Q-learning agent
    action_space = len(action_commands)

    # To train an agent on new shelves, simply replace the shelf names in the array below. Note that they must appear in this array
    # exactly as they appear in the state object of the game. Case matters.

    agent = ExpertAgent(action_space, epsilon=0.1)

####################
    #Once you have your agent trained, or you want to continue training from a previous training session, you can load the qtable from a json file
    #agent.qtable = pd.read_json('qtable.json')
####################
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    allGoalTypes = ['CART_RETURN', 'BASKET_RETURN', 'INTERACT', 'WALKWAY', 'REGISTER', 'EXIT', 'AISLE', 'SHELF_NAV', 'TOGGLE_CART', 'PICKUP_ITEM', 'COUNTER_NAV', 'PICKUP_COUNTER', 'EAST_WALKWAY', 'LEAVE_COUNTERS', 'PAY']

    numTrajectories, numTrainingEpisodes = 5, 40
    episode_length = 200 # Increase max episode length since sequences can be long
    success = 0

    trajectories = []
    generateTrajectories = True

    savedQTables = preloadQTable("training-output.json") if generateTrajectories else {t: {} for t in allGoalTypes}
    n = numTrajectories if generateTrajectories else numTrainingEpisodes
    for i in range(n):
        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        cnt = 0

        # keep track of our current goal so we know when to end the training loop
        goals, basketMode = buildGoals(state)
        agent.setGoals(goals)
        current_goal, current_goal_idx = goals[0], 0

        # For this iteration of training, use the most recent updated Q Table
        # At the first iteration, this will be blank
        # In subsequent iterations, we will save the qtable at the end when we detect the agent has left the store, and start this iteration with that qtable as we left it, so that each subsequent training loop can build on the knowledge of the previous.
        # Also reset the goals at each training loop
        agent.resetGoals(goals)
        agent.withQTables(savedQTables)

        trajectory = []
        while not state['gameOver']:
            cnt += 1
            action, action_index = getAction(agent, state, action_commands)
            phi_state = agent.trans(state)

            sock_game.send(str.encode(action))  # send action to env

            next_state = recv_socket_data(sock_game)  # get observation from env
            next_state = json.loads(next_state)
            next_phi_state = agent.trans(next_state)

            trajectory.append((phi_state, action_index, next_phi_state))

            if not generateTrajectories:
                # Define the reward based on the state and next_state
                reward = calculate_reward(state, next_state, current_goal, action_commands[action_index], basketMode)  # You need to define this function
                # Update Q-table
                agent.learning(action_index, reward, state, next_state)

            # Update state
            state = next_state

            if achievedGoal(state, current_goal):
                # update the current goal
                print("** Achieved", current_goal['name'])
                current_goal_idx += 1
                if current_goal_idx == len(goals):
                    # We finished all goals
                    print("******", i+1, "COMPLETED ALL GOALS ********")
                    trajectories.append(trajectory)
                    savedQTables = agent.qtables
                    break
                
                current_goal = goals[current_goal_idx]
                agent.achievedGoal() # Tell the agent we achieved our goal


            if cnt > episode_length:
                print("too many moves this loop. ending now")
                break

    if generateTrajectories:
        writeTrajectoriesToFile(trajectories, "trajectories.pkl")
        print("\n\n===== Wrote trajectories to trajectories.pkl =====\n\n")
    else:
        # At the end of training, write our saved q table to an output file
        with open("training-output.json", "w") as file:
            writeQTableToFile(savedQTables, "training-output.json")
            print("\n\n===== Wrote final Q Table to 'training-output.json' =====\n\n")

    # Close socket connection
    sock_game.close()

