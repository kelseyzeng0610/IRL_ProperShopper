# Overview of our experiment

1. The first thing we did is generate random shopping lists, each with 3 items. We generated 50 shopping lists. This is done with `python experiment.py --generate_lists`. A file is also saved in `experiment/runs` with the start state that can later be loaded, saved as `experiment/runs/start_state_i.txt` for the shopping list at index i.
2. Next, we ran the expert agent on each of these shopping lists in the actual environment 5 times, using epsilon=0.2, and recording the trajectories. These are saved as `experiment/runs/trajectories_i.pkl` for the shopping list at index i.

TODO: Now we have a few items to do.
1. Define some set of metrics that we want to record for each of our agents. We also need to configure the socket/agent such that we can run it and store those metrics somehow, or at least store something that allows us to calculate them later.
2. We need to process each trajectory pkl file, and add noise before running the HIRL pipeline. This can be done ahead of time but doesn't need to be.
3. We need to run the HIRL subgoal & max entropy learning pipeline on each trajectories file. This will produce a list of subgoals and a theta value array (optionally can save this to file as an intermediate). Then, we need to sample `M` trajectories from its learned policy using the epsilon-greedy method. These sampled trajectories need to be saved to file.
4. We can then run those sampled trajectories in the environment, and again record the same metrics.
5. We need to run the baseline IRL method (without HIRL segmentation) on each of the shopping lists and record the same metrics.
6. Optionally, try running the HIRL progress vector version or AIRL or something.
7. Finally, we can collect all of the results which should have been saved to file, and make some tables and plots to display the results and compare how well each method adheres to norms and completes the task.

Questions:
* Should we evaluate the metrics on the expert demos, just to see what our HIRL pipeline is learning from? It certainly has some norm violations because of the epsilon, but we'd have to rerun it because we didn't record violations.
