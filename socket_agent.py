
import json
import random
import socket

from env import SupermarketEnv
from utils import *
from utils import can_interact_default

def get_container(output:dict, socket_game) -> dict:
    """Get the correct container based on the shopping list.
    Args:
        output (dict): The current state of the environment.
        socket_game (socket.socket): The socket connection to the game server.
    Returns:
        output (dict): The updated state of the environment after getting the container.
    """
    # assumes that the player is in the west region near the entrance
    # check if player's position is west of the west walkway and below the register
    player = output['observation']['players'][0]
    player_pos = player['position']
    if player_pos[0] > WEST_WALKWAY['x_min'] or player_pos[1] < register_y_max:
        raise ValueError("Player is not in the correct position to get a container.")
    shopping_list = player['shopping_list']
    if len(shopping_list) <= 6: # get a basket
        rel_pos_fn = rel_pos_basket_return
    else: # get a cart
        rel_pos_fn = rel_pos_cart_return

    # navigate to the target container
    rel_pos = discretize(rel_pos_fn(output))
    while rel_pos != (0.0, 0.0):
        # first align x coordinate
        if rel_pos[0] > 0:
            action = "EAST"
        elif rel_pos[0] < 0:
            action = "WEST"
        else: # x coordinate aligned, align y coordinate
            if rel_pos[1] < 0:
                action = "NORTH"
            elif rel_pos[1] > 0:
                action = "SOUTH"
        output = step(socket_game, action)
        rel_pos = discretize(rel_pos_fn(output))
    
    # interact to pick up the container
    output = step(socket_game, "INTERACT")
    output = step(socket_game, "INTERACT")
    return output

def get_to_walkway(output:dict, socket_game) -> dict:
    """Get to the closest walkway

    Args:
        output (dict): The current state of the environment
        socket_game (socket.socket): The socket connection to the game server

    Returns:
        output (dict): The updated state of the environment after reaching the walkway
    """
    rel_pos = discretize(rel_pos_walkway(output))
    step_count = 0
    while rel_pos != (0.0, 0.0):
        # align x coordinate
        if rel_pos[0] > 0:
            action = "EAST"
        else :
            action = "WEST"
       
        output = step(socket_game, action)
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach walkway.")
        rel_pos = discretize(rel_pos_walkway(output))
    return output

def get_to_aisle(output:dict, socket_game, aisle:str) -> dict:
    """Get to an aisle where the food item is located.

    Args:
        output (dict): The current state of the environment
        socket_game (socket.socket): The socket connection to the game server.
        aisle (str): The aisle to navigate to.

    Returns:
        dict: The updated state of the environment after reaching the aisle
    """
    # first get to a walkway
    output = get_to_walkway(output, socket_game)
    # find the relative position of the aisle
    rel_pos = rel_pos_aisle(output, aisle)
    step_count = 0
    while rel_pos !=(0, 0):
        # align y coordinate
        if rel_pos[1] > 0:
            action = "SOUTH"
        else:
            action = "NORTH"
        output = step(socket_game, action)
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach aisle.")
        rel_pos = rel_pos_aisle(output, aisle)
    return output

def get_to_food_shelf(output:dict, socket_game, shelf:str) -> dict:
    """Get to the food shelf where the food item is located.

    Args:
        output (dict): The current state of the environment
        socket_game (socket.socket): The socket connection to the game server.
        shelf (str): The aisle to navigate to.
    Returns:
        dict: The updated state of the environment after reaching the shelf
    """
    # first get to the aisle
    output = get_to_aisle(output, socket_game, shelf)
    # find the relative position of the shelf
    rel_pos = discretize(rel_pos_shelf(output, shelf))
    step_count = 0
    while rel_pos !=(0, 0):
        # first align y coordinate
        if rel_pos[1] > 0:
            action = "SOUTH"
        elif rel_pos[1] < 0:
            action = "NORTH"
        else: # align x coordinate
            if rel_pos[0] > 0:
                action = "EAST"
            else:
                action = "WEST"
        output = step(socket_game, action)
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach shelf.")
        rel_pos = discretize(rel_pos_shelf(output, shelf))
    return output

def get_to_food_counter(output:dict, socket_game, food:str) -> dict:
    """Get to the counter with the food item

    Args:
        output (dict): The current state of the environment
        socket_game (socket.socket): The socket connection to the game server.
        food (str): The food item to get from the counter
    Returns:
        dict: The updated state of the environment after reaching the counter
    """
    # get to the middle aisle
    output = get_to_aisle(output, socket_game, "brie cheese")
    # get to the east walkway
    rel_pos_walkway = rel_pos_east_walkway(output)
    step_count = 0
    while rel_pos_walkway != (0.0, 0.0):
        # align x coordinate
        if rel_pos_walkway[0] > 0:
            action = "EAST"
        else:
            action = "WEST"
        output = step(socket_game, action)
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach east walkway.")
        rel_pos_walkway = rel_pos_east_walkway(output)
    
    # navigate to the food counter
    rel_pos = discretize(rel_pos_counter(output, food))
    step_count = 0
    while rel_pos != (0.0, 0.0):
        # align x coordinate
        if rel_pos[0] > 0:
            action = "EAST"
        elif rel_pos[0] < 0:
            action = "WEST"
        else: # align y coordinate
            if rel_pos[1] > 0:
                action = "SOUTH"
            else:
                action = "NORTH"
        output = step(socket_game, action)
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach counter.")
        rel_pos = discretize(rel_pos_counter(output, food))
    return output


def get_food_from_shelf(output:dict, socket_game, shelf:str) -> dict:
    """Get the food item from the shelf. 

    Args:
        output (dict): The current state of the environment
        socket_game (socket.socket): The socket connection to the game server
        shelf (str): the name of the food item

    Returns:
        dict: The updated state of the environment after getting the food
    """
    # first navigate to the food shelf
    output = get_to_food_shelf(output, socket_game, shelf)
    # check the player's container type
    player = output['observation']['players'][0]
    if player['curr_cart'] != -1: # holding a cart
        # let go of the cart
        output = step(socket_game, "TOGGLE_CART")
    # find the shelf object from the observation

    # assumes that the shelf is on the north side of the player, keep going north until can interact
    rel_pos = rel_pos_shelf(output, shelf)
    step_count = 0
    while rel_pos != (0.0, 0.0):
        output = step(socket_game, "NORTH")
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach shelf for interaction.")
        rel_pos = rel_pos_shelf(output, shelf)
    # interact to get the food item
    output = step(socket_game, "INTERACT")
    return output

def get_food_from_counter(output:dict, socket_game, food:str) -> dict:
    """Get the food item from the counter

    Args:
        output (dict): the current environment
        socket_game (socket.socket): The socket connection to the game server.
        food (str): The food item to get from the counter

    Returns:
        dict: The updated state of the environment after getting the food
    """ 
    # first navigate to the food counter
    output = get_to_food_counter(output, socket_game, food)
    # check the player's container type
    player = output['observation']['players'][0]
    if player['curr_cart'] != -1:  # holding a cart
        # let go of the cart
        output = step(socket_game, "TOGGLE_CART")
    # find the counter object from the observation
    counter_obj = next((obj for obj in output['observation']['counters'] if obj['food'] == food), None)
    if counter_obj is None:
        raise ValueError(f"Counter with food {food} not found in observation")
    # assumes that the counter is on the east side of the player, keep going east until can interact
    rel_pos = rel_pos_counter(output, food)
    step_count = 0
    while rel_pos != (0.0, 0.0):
        output = step(socket_game, "EAST")
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach counter for interaction.")
        rel_pos = rel_pos_counter(output, food)
    # interact to bring up the conversation box
    output = step(socket_game, "INTERACT")
    # interact to get the food item
    output = step(socket_game, "INTERACT")
    return output


def put_food_in_container(output:dict, socket_game, food:str) -> dict:
    """Put the food item in the basket or the cart

    Args:
        output (dict): the current environment
        socket_game (socket.socket)): The socket connection to the game server.
        food (str): The food item to put in the container

    Returns:
        dict: The updated state of the environment after putting food in container
    """
    # get the direction of the container object the player is holding
    cart_index = output['observation']['players'][0]['curr_cart']
    
    # pick up the food item. If it's prepared foods or fish, we need to get it from a counter
    if food in ["prepared foods", "fresh fish"]:
        output = get_food_from_counter(output, socket_game, food)
    else: # pick up the food item from shelf
        output = get_food_from_shelf(output, socket_game, food)
    
    if cart_index != -1: # the player was holding a cart
        cart = output['observation']['carts'][cart_index]
        cart_dir = INT_TO_DIRECTION[cart['direction']].name
    
    # check if the player is holding the food item. If so, put it in the container. 
    player = output['observation']['players'][0]
    if player['holding_food'] == food:
        # interact to close the conversation box
        output = step(socket_game, "INTERACT")
        # assumes player is using a cart, because player does not hold food when using basket
        output = step(socket_game, cart_dir) # turn to face the cart
        # interact to put the food item in the container
        output = step(socket_game, "INTERACT")
        # should no longer be holding the food item
        assert output['observation']['players'][0]['holding_food'] is None
        # check the food item is in the container
        cart = output['observation']['carts'][cart_index]
        assert food in cart['contents']
        # toggle to hold the cart again
        output = step(socket_game, "TOGGLE_CART")
    # interact to close the conversation box
    output = step(socket_game, "INTERACT")
    return output
    
    
def get_to_register(output:dict, socket_game) -> dict:
    """Navigate to one of the registers
    Args:
        output (dict): The current state of the environment
        socket_game (socket.socket): The socket connection to the game server.
    Returns:
        dict: The updated state of the environment after checkout
    """
    cart_index = output['observation']['players'][0]['curr_cart']
    # navigate to the west walkway
    output = get_to_aisle(output, socket_game, "sausage")
    rel_pos_walkway = rel_pos_west_walkway(output)
    step_count = 0
    while rel_pos_walkway != (0.0, 0.0):
        # first align x coordinate
        if rel_pos_walkway[0] > 0:
            action = "EAST"
        else :
            action = "WEST"
        output = step(socket_game, action)
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach walkway for checkout.")
        rel_pos_walkway = rel_pos_west_walkway(output)
    
    if cart_index != -1: # holding a cart
        # navigate to the brie cheese aisle so we can approach the register from the south
        output = get_to_aisle(output, socket_game, "brie cheese")
    else:
        output = get_to_aisle(output, socket_game, "sausage")
    # navigate to the register
    rel_pos_reg = discretize(rel_pos_register(output))
    step_count = 0
    while rel_pos_reg != (0.0, 0.0):
        # align x coordinate first
        if rel_pos_reg[0] > 0:
            action = "EAST"
        elif rel_pos_reg[0] < 0:
            action = "WEST"
        else: # align y coordinate
            if rel_pos_reg[1] > 0:
                action = "SOUTH"
            else:
                action = "NORTH"
    
        output = step(socket_game, action)
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach register for checkout.")
        rel_pos_reg = discretize(rel_pos_register(output)) 
    return output
    # navigate to the brie cheese aisle so we can approach the register from the south
    output = get_to_aisle(output, socket_game, "brie cheese")
    # navigate to the register
    rel_pos_reg = discretize(rel_pos_register(output))
    step_count = 0
    while rel_pos_reg != (0.0, 0.0):
        # align x coordinate first
        if rel_pos_reg[0] > 0:
            action = "EAST"
        elif rel_pos_reg[0] < 0:
            action = "WEST"
        else: # align y coordinate
            if rel_pos_reg[1] > 0:
                action = "SOUTH"
            else:
                action = "NORTH"
       
        output = step(socket_game, action)
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach register for checkout.")
        rel_pos_reg = discretize(rel_pos_register(output)) 
    return output

def checkout(output:dict, socket_game) -> dict:
    """Checkout the items in the container at the register

    Args:
        output (dict): The current state of the environment
        socket_game (socket.socket): The socket connection to the game server.

    Returns:
        dict: The updated state of the environment after checkout
    """
    cart_index = output['observation']['players'][0]['curr_cart']
    # navigate to the register
    output = get_to_register(output, socket_game)
    # check what container the player is holding
    if cart_index != -1: # holding a cart
        cart = output['observation']['carts'][cart_index]
        cart_dir = INT_TO_DIRECTION[cart['direction']].name
        # toggle to let go of the cart
        output = step(socket_game, "TOGGLE_CART")
    # turn to face the register
    rel_pos_reg = rel_pos_register(output)
    step_count = 0
    while rel_pos_reg != (0.0, 0.0):
        # assume the register is on the west side of the player
        action = "WEST"
        output = step(socket_game, action)
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to face register for checkout.")
        rel_pos_reg = rel_pos_register(output)
    # interact to bring up the checkout conversation box
    output = step(socket_game, "INTERACT")
    # interact to checkout
    output = step(socket_game, "INTERACT")

    if cart_index != -1: # was holding a cart
        output = step(socket_game, cart_dir) # turn to face the cart
        # interact to hold the cart again
        output = step(socket_game, "TOGGLE_CART")
    
    return output

def exit_supermarket(output:dict, socket_game) -> dict:
    """Exit the supermarket after checkout

    Args:
        output (dict): The current state of the environment
        socket_game (socket.socket): The socket connection to the game server.

    Returns:
        dict: The updated state of the environment after exiting the supermarket
    """
    # navigate to the west walkway
    rel_pos = discretize(rel_pos_exit(output))
    step_count = 0
    while rel_pos != (0.0, 0.0):
        # align y coordinate
        if rel_pos[1] > 0:
            action = "SOUTH"
        elif rel_pos[1] < 0:
            action = "NORTH"
        else: # align x coordinate
            if rel_pos[0] > 0:
                action = "EAST"
            else:
                action = "WEST"
        output = step(socket_game, action)
        step_count += 1
        if step_count > 300:
            raise RuntimeError("Too many steps taken to reach exit.") 
        rel_pos = discretize(rel_pos_exit(output))
    return output
    

def step(sock_game, action: str):
    """Send an action to the game server and receive the response. Action can be the following:
    - NOP
    - NORTH
    - SOUTH
    - EAST
    - WEST
    - TOGGLE_CART
    - INTERACT

    Args:
        sock_game (socket.socket): The socket connection to the game server.
        action (str): The action to be performed in the game.
    """
    action = "0 " + action 
    sock_game.send(str.encode(action))  # send action to env
    output = recv_socket_data(sock_game)  # get observation from env
    # when the game ends, output will be empty bytes
    if output == b'':
        print("Game has ended.")
        return None
    output = json.loads(output)
    return output


if __name__ == "__main__":

    # Make the env
    # env_id = 'Supermarket-v0'
    # env = gym.make(env_id)

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT']

    print("action_commands: ", action_commands)

    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    successes = 0
    
    for episode in range(10):

        output = step(sock_game, "RESET")
        print("Shopping List: ", output['observation']['players'][0]['shopping_list'])
        
        try:
            output = get_container(output, sock_game)
            shopping_list = output['observation']['players'][0]['shopping_list']
            for food in shopping_list:
                output = put_food_in_container(output, sock_game, food)
            output = checkout(output, sock_game)
            output = exit_supermarket(output, sock_game)
        except RuntimeError as e:
            print(f"Episode {episode} failed: {e}")
            continue
        successes += 1
        print(f"Episode {episode} succeeded.")
    
    print (f"Success rate: {successes}/10")

    

    
