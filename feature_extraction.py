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
        Extract simplified features for "Cart -> Item -> Checkout" task (8 features)
        1. x coordinate (normalized)
        2. y coordinate (normalized)
        3. Distance to cart return
        4. Distance to target item
        5. Distance to register
        6. Distance to exit
        7. Has cart (binary)
        8. Has item (binary)
        9. Item purchased (binary)
        """
        # Handle both formats: direct obs or wrapped in 'observation' key
        if 'observation' in obs:
            state_dict = obs
            game_obs = obs['observation']
        else:
            state_dict = {'observation': obs}
            game_obs = obs

        player = game_obs['players'][0]
        shopping_list = player['shopping_list']
        position = player['position']

        # Get container contents
        cart_contents, cart_purchased, basket_contents, basket_purchased = self._get_container_contents(player, game_obs)
        all_contents = cart_contents + basket_contents
        all_purchased = cart_purchased + basket_purchased

        total_items = len(shopping_list)

        features = []

        # ========== NAVIGATION FEATURES (5 features) ==========

        # 1. Distance to container (basket or cart based on list size)
        if total_items <= 6:
            # Use basket
            rel_pos = rel_pos_basket_return(state_dict)
        else:
            # Use cart
            rel_pos = rel_pos_cart_return(state_dict)
        distance_to_cart = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        features.append(self._normalize_distance(distance_to_cart))

         # 1. x coordinate (normalized)
        # Map width approx 20
        features.append(position[0] / 20.0)

        # 2. y coordinate (normalized)
        # Map height approx 25
        features.append(position[1] / 25.0)

        # 4. Distance to target item
        distance_to_item = 0.0
        # Find first uncollected item
        target_item = None
        for item in shopping_list:
            if item not in all_contents and item not in all_purchased:
                target_item = item
                break
        
        if target_item:
            try:
                if target_item in ['prepared foods', 'fresh fish']:
                    rel_pos = rel_pos_counter(state_dict, target_item)
                else:
                    rel_pos = rel_pos_shelf(state_dict, target_item)
                distance_to_item = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
            except:
                distance_to_item = 0.0
        features.append(self._normalize_distance(distance_to_item))

        # 5. Distance to register
        rel_pos = rel_pos_register(state_dict)
        distance_to_register = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        features.append(self._normalize_distance(distance_to_register))

        # 6. Distance to exit
        rel_pos = rel_pos_exit(state_dict)
        distance_to_exit = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        features.append(self._normalize_distance(distance_to_exit))

        # 7. Has cart
        has_cart = player['curr_cart'] != -1
        features.append(1.0 if has_cart else 0.0)

        # 8. Has item (collected but not necessarily purchased)
        # Check if we have any item from the list
        has_item = 0.0
        for item in shopping_list:
            if item in all_contents or item in all_purchased:
                has_item = 1.0
                break
        features.append(has_item)

        # 9. Item purchased
        is_purchased = 0.0
        for item in shopping_list:
            if item in all_purchased:
                is_purchased = 1.0
                break
        features.append(is_purchased)

        return np.array(features, dtype=np.float32)

    def _get_container_contents(self, player, obs):
        """Get all container contents"""
        cart_contents = []
        cart_purchased = []
        basket_contents = []
        basket_purchased = []

        # Check cart
        if player['curr_cart'] != -1:
            cart = obs['carts'][player['curr_cart']]
            cart_contents = cart.get('contents', [])
            cart_purchased = cart.get('purchased_contents', [])

        # Check baskets
        for basket in obs['baskets']:
            if basket.get('owner', -1) == player['index']:
                basket_contents.extend(basket.get('contents', []))
                basket_purchased.extend(basket.get('purchased_contents', []))

        return cart_contents, cart_purchased, basket_contents, basket_purchased

    def _normalize_distance(self, dist, max_dist=35.0):
        """Normalize distance to [0, 1] range"""
        return min(dist / max_dist, 1.0)

    def get_feature_dim(self):
        """Return dimensionality of feature vector"""
        if self.feature_type == 'handcrafted':
            return 9
        else:
            raise NotImplementedError()

    def get_feature_names(self):
        """Return names of features for debugging"""
        return [
            'x_norm',
            'y_norm',
            'distance_to_cart',
            'distance_to_item',
            'distance_to_register',
            'distance_to_exit',
            'has_cart',
            'has_item',
            'item_purchased'
        ]