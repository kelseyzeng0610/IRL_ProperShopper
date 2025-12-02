import numpy as np

def constructDPTable(trajectory, subgoals):
    # Dynamic programming approach to optimally segment the expert trajectories by subgoals.
    # Consider a trajectory T = [t1, t2, ..., tn] and subgoals S = [s1, s2, ..., sm]
    # let dp[i][j] be the minimum cost to segment the first j subgoals (s1...sj) using the first i steps of the trajectory (t1...ti)
    # where s_j is matched with t_i.
    # dp[i][j] therefore is the cost of the best alignment up to that point, ending with matching s_j and t_i.
    # Base case: if j=0, then dp[i][0] = 0 for all i because there is no cost to match zero subgoals.
    # Then for some i,j, we say d[i][j] = cost(s_j, t_i) + min_{k<i} dp[k][j-1]
    # meaning, the cost of matching subgoals 1..j to trajectory steps 1..i (where we also match t_i to s_j)
    # is calculated as the distance between s_j and t_i plus the best possible cost of matching subgoals 1..(j-1) (all previous subgoals)
    # to trajectory steps 1..(k) for some k < i. 
    n, m = len(trajectory), len(subgoals)
    dp = np.full((n, m+1), np.inf)
    parentIdx = np.full((n, m+1), -1) # this will store the parent index we came from, so we can reconstruct the assignment

    # Base case - anything for 0 subgoals is 0 cost
    dp[:, 0] = 0
    # anything for 1 subgoal is just the distance
    dp[:, 1] = [np.linalg.norm(trajectory[i] - subgoals[0]) for i in range(n)]

    # Iterate through i,j
    for j in range(2, m+1):
        for i in range(j-1, n):
            # cost of assigning subgoal j to trajectory step i is just the distance between them
            cost_ij = np.linalg.norm(trajectory[i] - subgoals[j-1])

            # now need to incorporate the previous best cost - look for the best k < i to match with j-1
            bestIdx = np.argmin(dp[:i, j-1])
            best_prev_cost = dp[bestIdx, j-1]
            parentIdx[i, j] = bestIdx
            dp[i][j] = cost_ij + best_prev_cost

    # Then the final optimal cost of assignment is dp[n-1][m]
    # return dp[n-1][m]
    return dp, parentIdx

def segmentTrajectoryBySubgoals(trajectory, subgoals):
    dp_table, parentIdx = constructDPTable(trajectory, subgoals)
    # now compute the actual assignment using the parentIdx table to backtrack
    assignment = np.full(len(subgoals), -1)
    n, m = len(trajectory), len(subgoals)

    # starting from the end, backtrack
    currentSubgoalIdx = m # start from the last subgoal
    trajectoryPoint = np.argmin(dp_table[:, m]) # the trajectory point that we should assign the last subgoal to is the one with minimal cost for this subgoal
    while currentSubgoalIdx > 0:
        assignment[currentSubgoalIdx-1] = trajectoryPoint # assign this trajectory point to the current subgoal
        trajectoryPoint = parentIdx[trajectoryPoint, currentSubgoalIdx] # move to the parent trajectory point
        currentSubgoalIdx -= 1 # move to the previous subgoal

    return assignment


if __name__ == "__main__":
    # Testing the DP segmentation

    # simple case with no ambiguity
    trajectory = np.array([[0.01, 0.0], [1.1, 0.0], [2.0, 0.0], [2.1, 0.95]])
    subgoals = np.array([[1.0, 0.0], [2.0, 1.0]])
    assignment = segmentTrajectoryBySubgoals(trajectory, subgoals)
    print("Assignment for simple case:")
    print(assignment)
    print("\n\n")

    # then a more complicated case with a cycle that more closely resembles the problem we face in our store environment trajectories
    # we saw an issue previously where the greedy approach failed when there was a loop because for the first subgoal,
    # the second time it hit that point in the trajectory was actually closer so it chose that whole segment.
    trajectory = np.array([
        [0.0, 0.0], 
        [1.1, 0.1], # the first time we get close to subgoal 1, we should be a bit "off"
        [2.0, 0.0], 
        [3.0, 0.0], # the second subgoal
        [2.0, 0.0], 
        [1.0, 0.0], # the third subgoal, but this time we are closer to it. our first subgoal should NOT be assigned to this even though this point is closer to it than the second point of the trajectory.
        [1.0, 1.0], # the last subgoal
    ])
    subgoals = np.array([[1.0, 0.0], [3.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assignment = segmentTrajectoryBySubgoals(trajectory, subgoals)
    print("Assignment for complicated case with loop:")
    print(assignment)
    print("\n\n")
