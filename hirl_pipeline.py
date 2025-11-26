"""
Simplified HIRL Pipeline - Feature Extraction Stage
Loads demonstrations → Extracts features → Later into Segmenter

Usage:
    python hirl_pipeline.py --data_path demonstrations/trajectories_*.pkl
"""
import argparse
import pickle
import numpy as np
from pathlib import Path
import json

from feature_extraction import StateFeatureExtractor


class HIRLPipeline:
    """Simplified pipeline for feature extraction and format verification"""
    
    def __init__(self, feature_dim=9):
        """
        Args:
            feature_dim: Dimensionality of state features (should be 9)
        """
        self.feature_dim = feature_dim
        self.feature_extractor = StateFeatureExtractor()
        
        # Results storage
        self.expert_trajectories_raw = []
        self.expert_trajectories_features = []
        
    def load_demonstrations(self, data_path):
        """
        Load expert demonstrations from pickle file
        
        Args:
            data_path: Path to .pkl file with trajectories
            
        Returns:
            List of trajectory dictionaries
        """
        print(f"\n{'='*60}")
        print("STEP 1: LOADING DEMONSTRATIONS")
        print(f"{'='*60}")
        
        with open(data_path, 'rb') as f:
            trajectories = pickle.load(f)
        
        print(f"✓ Loaded {len(trajectories)} trajectories from {data_path}")
        
        # Store raw trajectories
        self.expert_trajectories_raw = trajectories
        
        # Print statistics
        total_steps = sum(len(t['observations']) for t in trajectories)
        avg_length = total_steps / len(trajectories) if trajectories else 0
        
        print(f"  Total steps: {total_steps}")
        print(f"  Avg trajectory length: {avg_length:.1f}")
        
        # Show shopping list info
        if trajectories and 'shopping_list' in trajectories[0]:
            shopping_lists = [t.get('shopping_list', []) for t in trajectories]
            list_lengths = [len(sl) for sl in shopping_lists]
            print(f"  Shopping list sizes: {min(list_lengths)} - {max(list_lengths)} items")
        
        return trajectories
    
    def extract_features(self):
        """
        Extract features from raw observations
        Converts each observation to a 10D feature vector
        
        Returns:
            List of numpy arrays, each shape (num_steps, 10)
        """
        print(f"\n{'='*60}")
        print("STEP 2: FEATURE EXTRACTION")
        print(f"{'='*60}")
        
        self.expert_trajectories_features = []
        
        for traj_idx, trajectory in enumerate(self.expert_trajectories_raw):
            feature_trajectory = []
            
            for obs in trajectory['observations']:
                # Extract 10D feature vector
                features = self.feature_extractor.extract(obs)
                feature_trajectory.append(features)
            
            # Convert to numpy array
            feature_trajectory = np.array(feature_trajectory)
            self.expert_trajectories_features.append(feature_trajectory)
            
            if traj_idx == 0:
                print(f"\n  Sample trajectory {traj_idx}:")
                print(f"    Raw observations: {len(trajectory['observations'])} steps")
                print(f"    Feature shape: {feature_trajectory.shape}")
                print(f"    Feature dtype: {feature_trajectory.dtype}")
                print(f"\n    Feature sample (first step):")
                feature_names = self.feature_extractor.get_feature_names()
                for i, (name, val) in enumerate(zip(feature_names, feature_trajectory[0])):
                    print(f"      [{i}] {name:30s} = {val:.3f}")
        
        print(f"\n✓ Extracted features from {len(self.expert_trajectories_features)} trajectories")
        print(f"  Feature dimension: {self.feature_dim}D")
        
        return self.expert_trajectories_features
    
    
    def save_features(self, output_dir="./hirl_features"):
        """
        Save extracted features for later use
        
        Args:
            output_dir: Directory to save features
        """
        print("STEP 4: SAVING FEATURES")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save feature trajectories
        features_file = output_path / "feature_trajectories.pkl"
        with open(features_file, 'wb') as f:
            pickle.dump(self.expert_trajectories_features, f)
        print(f" Saved feature trajectories to {features_file}")
        
        # Save as numpy array (for easy inspection)
        features_npy = output_path / "feature_trajectories.npy"
        np.save(features_npy, self.expert_trajectories_features, allow_pickle=True)
        print(f" Saved as numpy array to {features_npy}")
        
        summary = {
            'num_trajectories': len(self.expert_trajectories_features),
            'feature_dim': self.feature_dim,
            'trajectory_shapes': [ft.shape for ft in self.expert_trajectories_features[:5]],
            'feature_names': self.feature_extractor.get_feature_names(),
            'total_steps': sum(len(ft) for ft in self.expert_trajectories_features),
            'avg_trajectory_length': np.mean([len(ft) for ft in self.expert_trajectories_features]),
        }
        
        summary_file = output_path / "features_summary.json"
        with open(summary_file, 'w') as f:
            # Convert numpy types to Python types
            summary_json = {}
            for key, value in summary.items():
                if isinstance(value, (np.integer, np.floating)):
                    summary_json[key] = float(value)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], tuple):
                    summary_json[key] = [list(v) for v in value]
                else:
                    summary_json[key] = value
            json.dump(summary_json, f, indent=2)
        print(f"✓ Saved summary to {summary_file}")
        
        print(f"\n✓ All features saved to {output_dir}")
        
        return output_path
    
    def run_pipeline(self, data_path, output_dir="./hirl_features", window_size=3):
        """
        Run the feature extraction pipeline
        
        Args:
            data_path: Path to expert demonstrations
            output_dir: Where to save features
            window_size: Window size for segmentation verification
        """
        print(f"\n{'#'*60}")
        print("HIRL FEATURE EXTRACTION PIPELINE")
        print(f"{'#'*60}")
        
        self.load_demonstrations(data_path)
        
        # Step 2: Extract features
        self.extract_features()
        
        # Step 3: Save features
        self.save_features(output_dir)
        
        print(f"\n{'#'*60}")
        print("PIPELINE COMPLETE!")
        print(f"{'#'*60}")
        print(f"\nResults saved to: {output_dir}")
        print(f"Extracted features from {len(self.expert_trajectories_features)} trajectories")
        print(f"Total steps: {sum(len(ft) for ft in self.expert_trajectories_features)}")
        
        print(f"\n{'='*60}")
        
        return self


def main():
    parser = argparse.ArgumentParser(description='Extract features for HIRL')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to expert demonstrations (.pkl file)')
    parser.add_argument('--output_dir', type=str, default='./hirl_features',
                       help='Directory to save features')
    parser.add_argument('--window_size', type=int, default=3,
                       help='Window size for segmentation (for verification)')
    
    args = parser.parse_args()
    
    # Verify data file exists
    if not Path(args.data_path).exists():
        print(f"Error: Data file not found: {args.data_path}")
        print("\nPlease collect demonstrations first:")
        print("  python collect_expert_data.py --num_episodes 100")
        return
    
    # Run pipeline
    pipeline = HIRLPipeline(feature_dim=9)
    
    pipeline.run_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        window_size=args.window_size
    )
    


if __name__ == "__main__":
    main()