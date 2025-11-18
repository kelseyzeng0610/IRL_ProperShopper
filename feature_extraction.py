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
        Extract task-relevant features (10 features total)

        FINAL DESIGN (10 features):
        - 5 navigation features (Euclidean distances)
        - 3 progress features (task completion + purchase tracking)
        - 2 state features (current conditions)
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

        # Get container contents
        cart_contents, cart_purchased, basket_contents, basket_purchased = self._get_container_contents(player, game_obs)
        all_contents = cart_contents + basket_contents
        all_purchased = cart_purchased + basket_purchased

        # Count progress
        num_collected = len(all_contents)
        num_purchased = len(all_purchased)
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
        distance_to_container = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        features.append(self._normalize_distance(distance_to_container))

        # 2. Distance to next item to collect 
        distance_to_next_item = 0.0
        for item in shopping_list:
            if item not in all_contents:
                try:
                    if item in ['prepared foods', 'fresh fish']:
                        rel_pos = rel_pos_counter(state_dict, item)
                    else:
                        rel_pos = rel_pos_shelf(state_dict, item)
                    distance_to_next_item = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
                except:
                    distance_to_next_item = 0.0
                break  
        features.append(self._normalize_distance(distance_to_next_item))

        # 3. Distance to checkout 
        rel_pos = rel_pos_register(state_dict)
        distance_to_checkout = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        features.append(self._normalize_distance(distance_to_checkout))

        # 4. Distance to exit 
        rel_pos = rel_pos_exit(state_dict)
        distance_to_exit = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        features.append(self._normalize_distance(distance_to_exit))

        # 5. Distance to walkway 
        rel_pos = rel_pos_walkway(state_dict)
        distance_to_walkway = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        features.append(self._normalize_distance(distance_to_walkway))

        # ========== PROGRESS FEATURES (3 features) ==========

        # 6. Collection progress ratio
        features.append(num_collected / total_items if total_items > 0 else 0.0)

        # 7. Has container: 0=none, 0.5=basket, 1.0=cart
        has_basket = len(basket_contents) > 0 or len(basket_purchased) > 0
        has_cart = player['curr_cart'] != -1
        if has_cart:
            features.append(1.0)  # Cart
        elif has_basket:
            features.append(0.5)  # Basket
        else:
            features.append(0.0)  # No container

        # 8. Items purchased (binary - detects checkout transition)
        features.append(1.0 if num_purchased > 0 else 0.0)

        # ========== STATE FEATURES (2 features) ==========

        # 9. Holding item (currently carrying food)
        features.append(1.0 if player.get('holding_food') is not None else 0.0)

        # 10. At interaction location (shelf/counter/register)
        at_interaction = 0.0
        if distance_to_next_item < 0.5:  # Close to item shelf
            at_interaction = 1.0
        elif distance_to_checkout < 0.5:  # Close to register
            at_interaction = 1.0
        elif distance_to_container < 0.5:  # Close to container return
            at_interaction = 1.0
        features.append(at_interaction)

        # Total: 10 features
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
            return 10
        else:
            raise NotImplementedError()

    def get_feature_names(self):
        """Return names of features for debugging"""
        return [
            'distance_to_container',
            'distance_to_next_item',
            'distance_to_checkout',
            'distance_to_exit',
            'distance_to_walkway',
            'collection_progress_ratio',
            'has_container',
            'items_purchased',
            'holding_item',
            'at_interaction_location'
        ]