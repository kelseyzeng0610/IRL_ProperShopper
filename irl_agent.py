from hirl import HIRLSegmenter
import numpy as np
from max_entropy_irl import MaxEntropyIRL
import json
import pickle

# Generate x,y for the shopping task
# Load the x,y as trajectories
# Segment the trajectories using HIRL segmenter
# learn a MaxEnt IRL for the full thing

def get_xy(gameState):
    playerPos = gameState["observation"]["players"][0]["position"]
    return (playerPos[0], playerPos[1])

def load_expert_trajectories():
    with open("trajectories.pkl", "rb") as f:
        data = pickle.load(f)
    
    return [np.asarray(traj) for traj in data]

if __name__ == "__main__":
    expertTrajectories = load_expert_trajectories()
    segmenter = HIRLSegmenter()
    subgoals = segmenter.subgoals(expertTrajectories)
    print(subgoals)
    # TODO: this can't get subgoals yet because we have more than one subgoal and that isn't implemented yet
