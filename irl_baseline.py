import numpy as np
from max_entropy_irl import MaxEntropyIRL
import json
from irl_agents_separate import getItemLocations, getTargetLocations, makeGetNextState, load_expert_trajectories, BASKET_LOCATION, REGISTER_LOCATION, START_STATE, FINAL_GOAL_LOCATION, THETA_SIZE, plotSampledTrajectory

def augmentStep(step, targetLocations):
    x, y = step[:2]
    hasBasket, hasItem1, hasItem2, hasItem3, hasPaid = step[2:]

    distToBasket = np.array([abs(x - BASKET_LOCATION[0]), abs(y - BASKET_LOCATION[1])])
    distToRegister = np.array([abs(x - REGISTER_LOCATION[0]), abs(y - REGISTER_LOCATION[1])])
    # only the shelf items
    distToItems = np.array([np.array([abs(x - loc[0]), abs(y - loc[1])]) for loc in targetLocations[1:4]])

    new_features = np.concatenate([
        distToBasket,
        [hasBasket],
        distToItems.flatten(),
        [hasItem1, hasItem2, hasItem3],
        distToRegister,
        [hasPaid]
    ])

    return new_features


def runBaseline(shoppingList, learnMode=False, thetaFile="experiment/baseline_theta.json", verbose=True):
    mask = np.full((THETA_SIZE), True)
    mask[[0,1]] = False # mask only the x,y features
    expertTrajectories = load_expert_trajectories("trajectories.pkl", maskShape=mask)

    itemLocations = getItemLocations()
    targetLocations = getTargetLocations(itemLocations, shoppingList)

    # now our phi is: 2 coordinates for distances to basket, 3 items, and register, plus 5 for the flags
    def phi(state):
        return augmentStep(state, targetLocations)
    
    # size of above phi
    augmentedThetaSize = 15 
    theta = np.random.uniform(0.0, 0.1, size=augmentedThetaSize)

    if not learnMode:
        with open(thetaFile, "r") as f:
            theta = np.array(json.load(f))
    
    getNextState = makeGetNextState(np.array(targetLocations))

    learner = MaxEntropyIRL(
        theta=theta,
        actions=np.arange(5),
        probeNextState=getNextState,
        initialState=START_STATE,
        gameOver = lambda x: np.allclose(x, FINAL_GOAL_LOCATION, atol=0.2),
        phi=phi,
    )

    if learnMode:
        theta_hat, _ = learner.learn(
            expertTrajectories, 
            num_iterations=300,
            alpha=0.01,
            num_samples=100,
            maxTrajectoryLength=300,
            verbose=True
        )
        if verbose:
            print("Learned theta:", theta_hat)
    
        with open("baseline_theta.json", "w") as f:
            json.dump(theta_hat.tolist(), f, indent=2)

    # generate a trajectory for this shopping list
    newTraj, actions = learner.greedy_trajectory(maxLength=500, epsilon=0.1, recordActions=True)
    if verbose:
        print("Final state:", newTraj[-1])
        print("Goal state:", FINAL_GOAL_LOCATION)
        
        # Analyze subgoal completion
        finalLocation = newTraj[-1]
        print(f"Has basket: {finalLocation[2] > 0.5}")
        print(f"Has item 1: {finalLocation[3] > 0.5}")
        print(f"Has item 2: {finalLocation[4] > 0.5}")
        print(f"Has item 3: {finalLocation[5] > 0.5}")
        print(f"Has paid: {finalLocation[6] > 0.5}")

    return newTraj, actions


if __name__ == "__main__":
    learnMode = False
    shoppingList = ['sausage', 'milk', 'banana']
    
    newTraj, actions = runBaseline(shoppingList, learnMode=learnMode)
    
    with open("baseline_trajectory.json", "w") as f:
        json.dump([step.tolist() for step in newTraj], f)
    with open("baseline_actions.json", "w") as f:
        json.dump([int(a) for a in actions], f)
    

    mask = np.full((THETA_SIZE), True)
    mask[[0,1]] = False # mask only the x,y features
    expertTrajectories = load_expert_trajectories("trajectories.pkl", maskShape=mask)
    plotSampledTrajectory(newTraj, expertTrajectories, subgoals=[], startState=START_STATE, imgPath="baseline.png", title="Baseline IRL Agent")




