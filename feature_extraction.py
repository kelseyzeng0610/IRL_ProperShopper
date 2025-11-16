"""
State feature extraction

"""

import numpy as np
from collections import defaultdict
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    rel_pos_cart_return,
    rel_pos_basket_return,
    rel_pos_register,
    rel_pos_exit,
    rel_pos_walkway,
    rel_pos_shelf,
    rel_pos_counter
)


class StateFeatureExtractor:
    """Extract compact feature representation from observations"""

    def __init__(self, feature_type='handcrafted'):
        """
        Args:
            feature_type: 'handcrafted' or 'raw'
        """
        self.feature_type = feature_type

    def extract(self, observation):
        """
        Extract features from observation
        """
        if self.feature_type == 'handcrafted':
            return self._extract_handcrafted(observation)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    def _extract_handcrafted(self, obs):
        """
        Extract task-relevant features (RECOMMENDED)

        Features capture:
        - (position)
        - (direction)
        - (holding)
        - (progress)
        - (next_location)
        """
        # Handle both formats: direct obs or wrapped in 'observation' key
        if 'observation' in obs:
            game_obs = obs['observation']
        else:
            game_obs = obs

        player = game_obs['players'][0]

        features = []

        # ===== POSITION FEATURES (2) =====
        features.extend(player['position'])  # x, y position

        # ===== DIRECTION FEATURE  =====
        direction_onehot = [0, 0, 0, 0]
        direction_onehot[player['direction']] = 1
        features.extend(direction_onehot)

        # ===== INVENTORY FEATURES =====
        # Has cart/basket? (2)
        has_cart = 1 if player['curr_cart'] != -1 else 0
        features.append(has_cart)

        has_basket = 1 if self._has_basket(player, game_obs) else 0
        features.append(has_basket)

        # Holding food? (1)
        features.append(1 if player['holding_food'] else 0)

        # Number of items in container (1)
        num_items_in_container = self._count_items_in_container(player, game_obs)
        features.append(num_items_in_container)

        # ===== TASK PROGRESS FEATURES =====
        # Shopping list size (1)
        shopping_list_size = len(player['shopping_list'])
        features.append(shopping_list_size)

        # Items collected so far (1)
        items_collected = num_items_in_container
        features.append(items_collected)

        # Items remaining (1)
        total_items_needed = sum(player['list_quant'])
        items_remaining = total_items_needed - items_collected
        features.append(items_remaining)

        # Completion rate (1)
        completion = items_collected / total_items_needed if total_items_needed > 0 else 1.0
        features.append(completion)

        # ===== SPATIAL FEATURES (distances to key locations) =====
        # Wrap in expected format for utils functions
        state_dict = {'observation': game_obs}

        # Distance to cart/basket return (2)
        dist_to_cart_return = rel_pos_cart_return(state_dict)
        features.append(dist_to_cart_return[0])  # x distance
        features.append(dist_to_cart_return[1])  # y distance

        # Distance to nearest needed shelf (2)
        dist_to_shelf = self._distance_to_needed_shelf(player, game_obs)
        features.append(dist_to_shelf[0])
        features.append(dist_to_shelf[1])

        # Distance to register (2)
        dist_to_register = rel_pos_register(state_dict)
        features.append(dist_to_register[0])
        features.append(dist_to_register[1])

        # Distance to exit (2)
        dist_to_exit = rel_pos_exit(state_dict)
        features.append(dist_to_exit[0])
        features.append(dist_to_exit[1])

        # Distance to walkway (1)
        rel_pos = rel_pos_walkway(state_dict)
        dist_to_walkway = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        features.append(dist_to_walkway)

        # ===== CHECKOUT STATUS =====
        # Items purchased? (1)
        items_purchased = self._count_purchased_items(player, game_obs)
        features.append(items_purchased)

        # All items purchased? (1)
        all_purchased = 1 if items_purchased >= total_items_needed else 0
        features.append(all_purchased)

        # Total: 25 features

        return np.array(features, dtype=np.float32)

    def _has_basket(self, player, obs):
        """Check if player has a basket"""
        for basket in obs['baskets']:
            if basket['owner'] == player['index']:
                return True
        return False

    def _count_items_in_container(self, player, obs):
        """Count total items in player's cart/basket (purchased + unpurchased)"""
        count = 0

        # Check carts
        if player['curr_cart'] != -1:
            cart = obs['carts'][player['curr_cart']]
            count += sum(cart['contents_quant'])
            count += sum(cart['purchased_quant'])

        # Check baskets
        for basket in obs['baskets']:
            if basket['owner'] == player['index']:
                count += sum(basket['contents_quant'])
                count += sum(basket['purchased_quant'])

        return count

    def _count_purchased_items(self, player, obs):
        """Count only purchased items"""
        count = 0

        if player['curr_cart'] != -1:
            cart = obs['carts'][player['curr_cart']]
            count += sum(cart['purchased_quant'])

        for basket in obs['baskets']:
            if basket['owner'] == player['index']:
                count += sum(basket['purchased_quant'])

        return count

    def _distance_to_needed_shelf(self, player, obs):
        """Distance to nearest shelf with item on shopping list"""
        # Find items still needed
        inventory = self._get_inventory_dict(player, obs)

        needed_items = []
        for i, food in enumerate(player['shopping_list']):
            needed_qty = player['list_quant'][i]
            current_qty = inventory.get(food, 0)
            if current_qty < needed_qty:
                needed_items.append(food)

        if not needed_items:
            return (0.0, 0.0)  

        # Find nearest shelf with needed item
        min_dist = float('inf')
        min_rel_pos = (0.0, 0.0)

        state_dict = {'observation': obs}

        for food in needed_items:
            try:
                if food in ['prepared foods', 'fresh fish']:
                    rel_pos = rel_pos_counter(state_dict, food)
                else:
                    rel_pos = rel_pos_shelf(state_dict, food)

                dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
                if dist < min_dist:
                    min_dist = dist
                    min_rel_pos = rel_pos
            except:
                continue

        return min_rel_pos

    def _get_inventory_dict(self, player, obs):
        """Get dictionary of items in inventory"""
        inventory = defaultdict(int)

        if player['holding_food']:
            inventory[player['holding_food']] += 1

        if player['curr_cart'] != -1:
            cart = obs['carts'][player['curr_cart']]
            for i, food in enumerate(cart['contents']):
                inventory[food] += cart['contents_quant'][i]
            for i, food in enumerate(cart['purchased_contents']):
                inventory[food] += cart['purchased_quant'][i]

        for basket in obs['baskets']:
            if basket['owner'] == player['index']:
                for i, food in enumerate(basket['contents']):
                    inventory[food] += basket['contents_quant'][i]
                for i, food in enumerate(basket['purchased_contents']):
                    inventory[food] += basket['purchased_quant'][i]

        return inventory

    def get_feature_dim(self):
        """Return dimensionality of feature vector"""
        if self.feature_type == 'handcrafted':
            return 25
        else:
            raise NotImplementedError()

    def get_feature_names(self):
        """Return names of features for debugging"""
        return [
            'pos_x', 'pos_y',
            'dir_north', 'dir_south', 'dir_east', 'dir_west',
            'has_cart', 'has_basket', 'holding_food',
            'items_in_container', 'shopping_list_size',
            'items_collected', 'items_remaining', 'completion_rate',
            'dist_cart_return_x', 'dist_cart_return_y',
            'dist_shelf_x', 'dist_shelf_y',
            'dist_register_x', 'dist_register_y',
            'dist_exit_x', 'dist_exit_y',
            'dist_walkway',
            'items_purchased', 'all_purchased'
        ]



   
