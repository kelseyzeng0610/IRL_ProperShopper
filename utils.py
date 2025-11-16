"""Utility functions for A3
Author: Helen Lu
"""
import time
from enum import Enum

shelves_x_min = 5.5
shelves_x_max = 15.5
player_width = 0.6
cart_width = 0.6
register_x_max = 1.8
register_y_max = 12
counter_min_x = 18.25
exit = [0, 7.25]
entrance = [0, 15.6]



WEST_WALKWAY = { # any x value between 3.75 and 4.5 is in the east walkway
    'x_min': 4, 
    'x_max': 4.5,
}

EAST_WALKWAY = { # any x value between 16.25 and 17.0 is in the east walkway
    'x_min': 16.7, 
    'x_max': 17.05,
}



# The aisles boundaries from the top aisle to the bottom aisle
# AISLE_1 = {
#     'x_min': 3.75, 
#     'x_max': 15.5,
#     'y_min': 2.5,
#     'y_max': 5.5,
# }
# AISLE_2 = {
#     'x_min': 3.75, 
#     'x_max': 15.5,
#     'y_min': 6.5,
#     'y_max': 9.5,
# }
# AISLE_3 = {
#     'x_min': 3.75, 
#     'x_max': 15.5,
#     'y_min': 11.5,
#     'y_max': 13.5,
# }
# AISLE_4 = {
#     'x_min': 3.75, 
#     'x_max': 15.5,
#     'y_min': 15.5,
#     'y_max': 17.5,
# }
# AISLE_5 = {
#     'x_min': 3.75, 
#     'x_max': 15.5,
#     'y_min': 19.5,
#     'y_max': 21.5,
# }
# AISLE_6 = {
#     'x_min': 3.75, 
#     'x_max': 15.5,
#     'y_min': 23.5,
#     'y_max': 25.5,
# }


class Direction(Enum):
    NONE = 0,
    NORTH = 1,
    SOUTH = 2,
    EAST = 3,
    WEST = 4

INT_TO_DIRECTION = {0: Direction.NORTH, 1: Direction.SOUTH, 2: Direction.EAST, 3: Direction.WEST}

def discretize(pos: tuple[float, float], granularity: float = 0.25) -> tuple[int, int]:
    """Discretize a continuous position into grid with increments of granularity.

    Args:
        pos (tuple[float, float]): The continuous (x, y) position.
        granularity (float): The size of each grid cell.

    Returns:
        tuple[int, int]: The discretized (x, y) position as grid coordinates.
    """
    disc_x = round(pos[0] / granularity) * granularity
    disc_y = round(pos[1] / granularity) * granularity
    return (disc_x, disc_y)

def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def rel_pos_exit(state) -> tuple[float, float]:
    """Calculates the distance to the exit from the agent's current position.

    Args:
        state (dict): The current state of the environment.
    """
    player_pos = state['observation']['players'][0]['position']
    exit_pos = exit

    # Calculate the relative position of the exit
    return exit_pos[0] - player_pos[0], exit_pos[1] - player_pos[1]

def rel_pos_cart_return(state) -> tuple[float, float]:
    """Calculates the distance to the closest cart return from the agent's current position.

    Args:
        state (dict): The current state of the environment.
    """
    # check if player has already reached a cart return
    if can_interact_default(state['observation']['cartReturns'][0], state['observation']['players'][0]) or \
       can_interact_default(state['observation']['cartReturns'][1], state['observation']['players'][0]):
        return 0.0, 0.0
    player_pos = state['observation']['players'][0]['position']
    cart_returnA_pos = state['observation']['cartReturns'][0]['position']
    cart_returnB_pos = state['observation']['cartReturns'][1]['position']

    # Calculate the relative position of the closest cart return
    distanceA = euclidean_distance(player_pos, cart_returnA_pos)
    distanceB = euclidean_distance(player_pos, cart_returnB_pos)

    rel_pos_A = cart_returnA_pos[0] - player_pos[0], cart_returnA_pos[1] - player_pos[1]
    rel_pos_B = cart_returnB_pos[0] - player_pos[0], cart_returnB_pos[1] - player_pos[1]

    if distanceA < distanceB:
        return rel_pos_A
    else:
        return rel_pos_B

def rel_pos_basket_return(state) -> tuple[float, float]:
    """Calculates the distance to the closest basket return from the agent's current position.

    Args:
        state (dict): The current state of the environment.
    """
    # check if player has already reached a basket return
    if can_interact_default(state['observation']['basketReturns'][0], state['observation']['players'][0]):
        return 0.0, 0.0
    player_pos = state['observation']['players'][0]['position']
    basket_return_pos = state['observation']['basketReturns'][0]['position']
    
    # Calculate the relative position of the basket return
    return basket_return_pos[0] - player_pos[0], basket_return_pos[1] - player_pos[1]

def rel_pos_register(state) -> tuple[float, float]:
    """Calculates the distance to the closest register from the agent's current position.

    Args:
        state (dict): The current state of the environment.
    """
    # check if player has already reached a register
    if can_interact_default(state['observation']['registers'][0], state['observation']['players'][0]) or \
       can_interact_default(state['observation']['registers'][1], state['observation']['players'][0]):
        return 0.0, 0.0
    player_pos = state['observation']['players'][0]['position']
    register_width = state['observation']['registers'][0]['width']
    register_height = state['observation']['registers'][0]['height']
    # use the mid point of the interaction box on the east side of the register as the register position
    registerA_pos = state['observation']['registers'][0]['position']
    registerB_pos = state['observation']['registers'][1]['position']
    registerA_interact_pos = (registerA_pos[0] + register_width + 0.3, registerA_pos[1] + register_height / 2)
    registerB_interact_pos = (registerB_pos[0] + register_width + 0.3, registerB_pos[1] + register_height / 2)

    # Calculate the relative position of the closest register
    distanceA = euclidean_distance(player_pos, registerA_pos)
    distanceB = euclidean_distance(player_pos, registerB_pos)

    rel_pos_A = (registerA_interact_pos[0] - player_pos[0], registerA_interact_pos[1] - player_pos[1])
    rel_pos_B = (registerB_interact_pos[0] - player_pos[0], registerB_interact_pos[1] - player_pos[1])

    if distanceA < distanceB:
        return rel_pos_A
    else:
        return rel_pos_B

def rel_pos_walkway(state) -> tuple[float, float]:
    """Calculates the relative position of the closest walkway from the player's current position.
    Args:
        state (dict): The current state of the environment.
    Returns:
        tuple[float, float]: The relative (x, y) distance to the closest walkway
    """
    # determine distance to west walkway
    rel_pos_west = rel_pos_west_walkway(state)
    dist_west = abs(rel_pos_west[0])
    # determine distance to east walkway
    rel_pos_east = rel_pos_east_walkway(state)
    dist_east = abs(rel_pos_east[0])
    # return the relative position of the closest walkway
    if dist_west < dist_east:
        return rel_pos_west
    else:
        return rel_pos_east

def rel_pos_west_walkway(state) -> tuple[float, float]:
    """Calculates the relative position of the west walkway from the player's current position.
    Args:
        state (dict): The current state of the environment.
    Returns:
        tuple[float, float]: The relative (x, y) distance to the west walkway
    """
    player_pos = state['observation']['players'][0]['position']
    # check if player is in west walkway
    if WEST_WALKWAY['x_min'] <= player_pos[0] <= WEST_WALKWAY['x_max']:
        return 0.0, 0.0
    # if not in west walkway, calculate distance to it
    closest_x = max(WEST_WALKWAY['x_min'], min(player_pos[0], WEST_WALKWAY['x_max']))
    return (closest_x - player_pos[0], 0.0)

def rel_pos_east_walkway(state) -> tuple[float, float]:
    """Calculates the relative position of the east walkway from the player's current position.
    Args:
        state (dict): The current state of the environment.
    Returns:
        tuple[float, float]: The relative (x, y) distance to the east walkway
    """
    player_pos = state['observation']['players'][0]['position']
    # check if player is in east walkway
    if EAST_WALKWAY['x_min'] <= player_pos[0] <= EAST_WALKWAY['x_max']:
        return 0.0, 0.0
    # if not in east walkway, calculate distance to it
    closest_x = max(EAST_WALKWAY['x_min'], min(player_pos[0], EAST_WALKWAY['x_max']))
    return (closest_x - player_pos[0], 0.0)

def rel_pos_aisle(state, aisle="milk_aisle") -> tuple[float, float]:
    """Assuming player is in the west walkway, this calculates the distance to the aisle containing the specified shelf.

    Args:
        state (dict): The current state of the environment.
        aisle (str, optional): The name of the aisle to check for. Defaults to "milk_aisle".

    Returns:
        tuple[float, float]: The relative (x, y) distance to the aisle containing the specified shelf
    """

    # get the shelf name without the '_aisle' suffix
    aisle = aisle.replace('_aisle', '')
    player_pos = state['observation']['players'][0]['position']
    # iterate through shelves to find the specified shelf
    shelf_pos = None
    shelf_height = None
    for s in state['observation']['shelves']:
        if s['food'] == aisle:
            shelf_pos = s['position']
            shelf_height = s['height']
            break
    if shelf_pos is None:
        raise ValueError(f"Shelf '{aisle}' not found in the environment.")
    # boundaries of the aisle containing the shelf
    aisle_min_x = 3.75
    aisle_max_x = 18.00
    aisle_min_y = shelf_pos[1] + shelf_height + 0.4
    aisle_max_y = aisle_min_y + 2
    # if the player is already in the aisle, distance is 0
    if aisle_min_x <= player_pos[0] <= aisle_max_x and aisle_min_y <= player_pos[1] <= aisle_max_y:
        return 0.0, 0.0
    # calculate the distance to the closest point in the aisle
    closest_x = max(aisle_min_x, min(player_pos[0], aisle_max_x))
    closest_y = max(aisle_min_y, min(player_pos[1], aisle_max_y))
    return (closest_x - player_pos[0], closest_y - player_pos[1])

def rel_pos_shelf(state, shelf="milk") -> tuple[float, float]:
    """Assuming the player is in the aisle, this function calculates the distance to the specified shelf from the agent's current position.

    Args:
        state (dict): The current state of the environment.
        shelf (str, optional): The name of the shelf to check for. Defaults to "milk".

    Returns:
        tuple[float, float]: The relative (x, y) distance to the specified shelf
    """
    # assert rel_pos_aisle(state, aisle=f"{shelf}_aisle") == (0.0, 0.0), "Player must be in the aisle containing the shelf to calculate distance to shelf."
    player_pos = state['observation']['players'][0]['position']
    shelf_obj = None
    for s in state['observation']['shelves']:
        if s['food'] == shelf:
            shelf_obj = s
            break
    if shelf_obj is None:
        raise ValueError(f"Shelf '{shelf}' not found in the aisle.")
    # make the shelf position the bottom mid point of the shelf's interaction box
    shelf_pos = (shelf_obj['position'][0] + shelf_obj['width'] / 2, shelf_obj['position'][1] + shelf_obj['height'] + 0.4)
    # check if player has already reached the shelf
    player = state['observation']['players'][0]
    if can_interact_shelf(shelf_obj, player):
        return 0.0, 0.0
    return (shelf_pos[0] - player_pos[0], shelf_pos[1] - player_pos[1])

def rel_pos_counter(state, counter='prepared foods') -> tuple[float, float]:
    """Assuming player is in the east walkway, this calculates the distance to the counter from the agent's current position.
    Args:
        state (dict): The current state of the environment.
    Returns:
        tuple[float, float]: The relative (x, y) distance to the counter
    """
    player_pos = state['observation']['players'][0]['position']
    # find the specified counter
    counter_obj = None
    for c in state['observation']['counters']:
        if c['food'] == counter:
            counter_obj = c
            break
    if counter_obj is None:
        raise ValueError(f"Counter '{counter}' not found in the environment.")
    counter_pos = (counter_obj['position'][0] - 0.6, counter_obj['position'][1] + counter_obj['height'] // 2)
    # check if player has already reached the counter
    player = state['observation']['players'][0]
    if can_interact_default(counter_obj, player):
        return 0.0, 0.0
    return counter_pos[0] - player_pos[0], counter_pos[1] - player_pos[1]

def save_state(state, filename):
    """Save state observation in original format (for literal_eval)"""
    with open(filename, "w") as f:
        f.write(str(state['observation']))
    print(f"State saved to {filename} in original format")

def can_interact_shelf(obj:dict, player:dict, range=0.5):

    if INT_TO_DIRECTION[player['direction']] == Direction.NORTH:
        return collision(obj, player, player['position'][0], player['position'][1] - range)
    elif INT_TO_DIRECTION[player['direction']] == Direction.SOUTH:
        return collision(obj, player, player['position'][0], player['position'][1] + range)
    # not interactable from west or east
    return False

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
        return overlap(
            obj1['position'][0], obj1['position'][1], obj1['width'], obj1['height'],
            x_position, y_position, obj2['width'], obj2['height'])

def overlap(x1, y1, width_1, height_1, x2, y2, width_2, height_2):
    return not (x1 > x2 + width_2 or x2 > x1 + width_1 or y1 > y2 + height_2 or y2 > y1 + height_1)



def determine_reached_goal(state, goal_location):
    """Determine if the agent has reached the specified goal location.

    Args:
        state (dict): The current state of the environment.
        goal_location (str): The goal location to check for.
    Returns:
        bool: True if the agent has reached the goal location, False otherwise.
    """
    if goal_location == "cart_return":
        return rel_pos_cart_return(state) == (0.0, 0.0)
    elif goal_location == "west_walkway":
        return rel_pos_west_walkway(state) == (0.0, 0.0)
    elif goal_location == "east_walkway":
        return rel_pos_east_walkway(state) == (0.0, 0.0)
    elif goal_location == "walkway":
        return rel_pos_walkway(state) == (0.0, 0.0)
    elif goal_location == "basket_return":
        return rel_pos_basket_return(state) == (0.0, 0.0)
    elif goal_location.endswith('_aisle'): # goal is to reach an aisle
        return rel_pos_aisle(state, aisle=goal_location) == (0.0, 0.0)
    elif goal_location == "register":
        return rel_pos_register(state) == (0.0, 0.0)
    elif goal_location.endswith('_counter'): # goal is to reach a counter
        return rel_pos_counter(state) == (0.0, 0.0)
    else: # goal is a shelf in an aisle
        return rel_pos_shelf(state, shelf=goal_location) == (0.0, 0.0)

def determine_goal_location(state, goal_locations, index_curr_goal=0):
    """Determine the next goal location based on whether the agent has reached the current goal location.
    The goal locations are:
    1. "cart_return": if the agent is not at the cart return, the goal is to reach a cart return.
    2. "west_walkway": if the agent has reached the cart return, the goal is to reach the east walkway.
    3. "milk_aisle": if the agent has reached the east walkway, the goal is to reach the milk aisle.
    4. "milk": if the agent has reached the milk aisle, the goal is to reach the milk shelf.
    5. "west_walkway": if the agent has reached the milk shelf, the goal is to return to the east walkway.
    6. "sausage_aisle": if the agent has reached the east walkway, the goal is to reach the sausage aisle.
    7. "sausage": if the agent has reached the sausage aisle, the goal is to reach the sausage shelf.
    8. "west_walkway": if the agent has reached the sausage shelf, the goal is to return to the east walkway.
    9. "register": if the agent has reached the east walkway, the goal is to reach a register.

    Args:
        state (dict): The current state of the environment.
        index_curr_goal (int, optional): The index of the current goal in the goal sequence. Defaults to 0.

    Returns:
        str: The current goal location.
    """
    # determine if the agent has reached the current goal location
    current_goal = goal_locations[index_curr_goal]
    if determine_reached_goal(state, current_goal):
        index_curr_goal += 1
    return index_curr_goal
    

def recv_socket_data(sock):
    BUFF_SIZE = 4096  # 4 KiB
    data = b''
    while True:
        time.sleep(0.00001)
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break

    return data