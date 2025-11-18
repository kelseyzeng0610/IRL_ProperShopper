"""
Trajectory recorder for collecting expert demonstrations
"""

import json
import pickle
import numpy as np
from collections import defaultdict
from datetime import datetime
import os


class TrajectoryRecorder:
    """Records expert demonstrations for imitation learning"""

    def __init__(self, save_dir="./demonstrations"):
        """
        Args:
            save_dir: Directory to save demonstrations
        """
        self.save_dir = save_dir
        self.current_trajectory = None
        self.all_trajectories = []

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def start_episode(self, initial_obs):
        """
        Start recording a new episode

        Args:
            initial_obs: Initial observation from environment reset
        """
        self.current_trajectory = {
            'observations': [initial_obs],
            'actions': [],
            'shopping_list': initial_obs['observation']['players'][0]['shopping_list'].copy(),
            'list_quantities': initial_obs['observation']['players'][0]['list_quant'].copy(),
            'timesteps': [],
            'success': False,
            'episode_length': 0,
            'norm_violations': [],
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'num_items': len(initial_obs['observation']['players'][0]['shopping_list'])
            }
        }

    def record_step(self, action, observation, timestep=None):
        """
        Record a single step

        Args:
            action: Action taken (string format like "NORTH" or "0 NORTH")
            observation: Observation received after action
            timestep: Optional timestep number
        """
        if self.current_trajectory is None:
            raise ValueError("Must call start_episode() first")

        self.current_trajectory['actions'].append(action)
        self.current_trajectory['observations'].append(observation)

        if timestep is not None:
            self.current_trajectory['timesteps'].append(timestep)

        self.current_trajectory['episode_length'] += 1

    def record_norm_violation(self, violation_info):
        """
        Record any norm violations during episode

        Args:
            violation_info: Dict with violation details
        """
        if self.current_trajectory is not None:
            self.current_trajectory['norm_violations'].append(violation_info)

    def end_episode(self, success=False, final_obs=None):
        """
        Finish recording episode

        Args:
            success: Whether episode completed successfully
            final_obs: Final observation (optional)
        """
        if self.current_trajectory is None:
            return

        self.current_trajectory['success'] = success
        self.current_trajectory['metadata']['end_time'] = datetime.now().isoformat()
        if final_obs:
            player = final_obs['observation']['players'][0]
            self.current_trajectory['metadata']['completion_rate'] = self._calculate_completion(
                player, final_obs
            )
        else:
            # Use last observation
            if len(self.current_trajectory['observations']) > 0:
                last_obs = self.current_trajectory['observations'][-1]
                player = last_obs['observation']['players'][0]
                self.current_trajectory['metadata']['completion_rate'] = self._calculate_completion(
                    player, last_obs
                )

        self.all_trajectories.append(self.current_trajectory)
        self.current_trajectory = None

    def _calculate_completion(self, player, obs):
        """
        Calculate what fraction of shopping list was completed

        Args:
            player: Player dict from observation
            obs: Full observation

        Returns:
            Completion rate (0.0 to 1.0)
        """
        # Get player's inventory from all carts/baskets
        inventory = defaultdict(int)

        # Items in hand
        if player['holding_food']:
            inventory[player['holding_food']] += 1

        # Items in carts
        for cart in obs['observation']['carts']:
            if cart['owner'] == player['index']:
                for i, food in enumerate(cart['contents']):
                    inventory[food] += cart['contents_quant'][i]
                for i, food in enumerate(cart['purchased_contents']):
                    inventory[food] += cart['purchased_quant'][i]

        # Items in baskets
        for basket in obs['observation']['baskets']:
            if basket['owner'] == player['index']:
                for i, food in enumerate(basket['contents']):
                    inventory[food] += basket['contents_quant'][i]
                for i, food in enumerate(basket['purchased_contents']):
                    inventory[food] += basket['purchased_quant'][i]

        # Items in bags 
        if 'bagged_items' in player:
            for i, food in enumerate(player['bagged_items']):
                inventory[food] += player['bagged_quant'][i]

        # Calculate completion
        total_needed = sum(player['list_quant'])
        total_collected = 0

        for i, food in enumerate(player['shopping_list']):
            needed = player['list_quant'][i]
            collected = min(inventory.get(food, 0), needed)
            total_collected += collected

        return total_collected / total_needed if total_needed > 0 else 0.0

    def save(self, filename=None):
        """
        Save all trajectories to disk

        Args:
            filename: Optional custom filename. If None, auto-generates with timestamp
        """
        # Only save successful trajectories without violations
        successful_trajectories = [
            t for t in self.all_trajectories 
            if t['success'] and len(t.get('norm_violations', [])) == 0
        ]
        
        if not successful_trajectories:
            print("No successful, violation-free trajectories to save!")
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/trajectories_{timestamp}.pkl"

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save only successful, violation-free trajectories
        with open(filename, 'wb') as f:
            pickle.dump(successful_trajectories, f)

        print(f"Saved {len(successful_trajectories)} violation-free trajectories to {filename}")
        print(f"(Filtered out {len(self.all_trajectories) - len(successful_trajectories)} trajectories with violations or failures)")

        # Also save summary as JSON for easy inspection
        summary = self.get_statistics()
        summary_file = filename.replace('.pkl', '_summary.json')

        with open(summary_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            summary_json = {}
            for key, value in summary.items():
                if isinstance(value, (np.integer, np.floating)):
                    summary_json[key] = float(value)
                elif isinstance(value, dict):
                    summary_json[key] = {k: int(v) if isinstance(v, (np.integer, np.floating)) else v
                                        for k, v in value.items()}
                else:
                    summary_json[key] = value

            json.dump(summary_json, f, indent=2)

        print(f"Saved summary to {summary_file}")

        return filename

    def load(self, filename):
        """
        Load trajectories from disk

        Args:
            filename: Path to saved trajectories file
        """
        with open(filename, 'rb') as f:
            self.all_trajectories = pickle.load(f)
        print(f"Loaded {len(self.all_trajectories)} trajectories from {filename}")

    def get_statistics(self):
        """
        Get statistics about collected trajectories

        Returns:
            Dict with statistics
        """
        if not self.all_trajectories:
            return {"message": "No trajectories recorded yet"}

        # Filter for violation-free trajectories
        violation_free = [t for t in self.all_trajectories if len(t.get('norm_violations', [])) == 0]
        
        stats = {
            'total_trajectories': len(self.all_trajectories),
            'successful_trajectories': sum(1 for t in self.all_trajectories if t['success']),
            'violation_free_trajectories': len(violation_free),
            'trajectories_with_violations': len(self.all_trajectories) - len(violation_free),
            'avg_episode_length': float(np.mean([t['episode_length'] for t in self.all_trajectories])),
            'std_episode_length': float(np.std([t['episode_length'] for t in self.all_trajectories])),
            'min_episode_length': min(t['episode_length'] for t in self.all_trajectories),
            'max_episode_length': max(t['episode_length'] for t in self.all_trajectories),
            'total_transitions': sum(t['episode_length'] for t in self.all_trajectories),
            'shopping_list_distribution': defaultdict(int),
            'avg_completion_rate': 0.0
        }

        # Calculate distributions
        completion_rates = []
        for traj in self.all_trajectories:
            list_size = traj['metadata']['num_items']
            stats['shopping_list_distribution'][list_size] += 1

            if 'completion_rate' in traj['metadata']:
                completion_rates.append(traj['metadata']['completion_rate'])

        if completion_rates:
            stats['avg_completion_rate'] = float(np.mean(completion_rates))
            stats['min_completion_rate'] = float(np.min(completion_rates))
            stats['max_completion_rate'] = float(np.max(completion_rates))

        # Convert defaultdict to regular dict for JSON serialization
        stats['shopping_list_distribution'] = dict(stats['shopping_list_distribution'])

        return stats

    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()

        if 'message' in stats:
            print(stats['message'])
            return

        print("\n" + "="*60)
        print("TRAJECTORY STATISTICS")
        print(f"Total episodes: {stats['total_trajectories']}")
        print(f"Successful: {stats['successful_trajectories']} "
              f"({stats['successful_trajectories']/stats['total_trajectories']*100:.1f}%)")
        print(f"Violation-free: {stats['violation_free_trajectories']} "
              f"({stats['violation_free_trajectories']/stats['total_trajectories']*100:.1f}%)")
        print(f"With violations: {stats['trajectories_with_violations']}")
        print(f"Avg episode length: {stats['avg_episode_length']:.1f} ± {stats['std_episode_length']:.1f}")
        print(f"  Min: {stats['min_episode_length']}, Max: {stats['max_episode_length']}")
        print(f"Total transitions: {stats['total_transitions']}")

        if 'avg_completion_rate' in stats:
            print(f"Avg completion rate: {stats['avg_completion_rate']*100:.1f}%")

        print(f"\nShopping list size distribution:")
        for size, count in sorted(stats['shopping_list_distribution'].items()):
            print(f"  {size} items: {count} episodes")
        print("="*60)

