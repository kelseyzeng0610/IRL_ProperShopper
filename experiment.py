import numpy as np
import json
import subprocess
import time
from experiment_utils import buildRandomLists, getParser
import pickle
from pathlib import Path
from irl_agents_separate import getExpertTrajectoriesWithNoise, getSubgoals, learnSegments, generateLearnedTrajectory, loadLearnedAgents, saveLearnedAgents, plotSampledTrajectory, START_STATE

def generateRandomLists(numLists=50, listSize=3):
    randomLists = buildRandomLists(numLists, listSize)
    with open('experiment/random_shopping_lists.json', 'w') as f:
        json.dump(randomLists, f)

def generateIRLTrajectories(shoppingLists, startIdx, endIdx, headless=False):
    for i in range(startIdx, endIdx):
        shoppingList = shoppingLists[i]
        modifiedList = startState.copy()
        modifiedList['players'][0]['shopping_list'] = shoppingList

        filePath = Path(f'experiment/runs/run_{i}')
        filePath.mkdir(parents=True, exist_ok=True)
        startStateFileName = f'experiment/runs/run_{i}/start_state_{i}.txt'
        trajectoryFileName = f'experiment/runs/run_{i}/trajectories_{i}.pkl'
        
        # write the modified start state to a file so we can start the environment with this
        with open(startStateFileName, 'w') as f:
            f.write(repr(modifiedList))

        # Step 1: start the environment with the provided start state
        args = ['python', 'socket_env.py', '--player_speed=0.25', f'--file={startStateFileName}']
        if headless:
            args.append('--headless')
        envProcess = subprocess.Popen(args)

        # Step 2: Run the expert agent with epsilon > 0 to generate trajectories (they will be saved to file)
        if not headless:
            time.sleep(5)  # wait for the env to start up
        genProcess = subprocess.run([
            'python', 'socket_agent_expert.py',
            '--generate_trajectories', 
            f'--trajectory_output={trajectoryFileName}',
            '--agent_epsilon=0.2',
            f'--metrics_output=experiment/runs/run_{i}/metrics_expert_demos_{i}.json'
        ])

        envProcess.terminate()
        envProcess.wait()

def addNoiseForIRL(startIdx, endIdx):
    for i in range(startIdx, endIdx):
        # add noise to existing trajectories
        noisy = getExpertTrajectoriesWithNoise(f'experiment/runs/run_{i}/trajectories_{i}.pkl')
        
        # save to new file
        with open(f'experiment/runs/run_{i}/noisy_trajectories_{i}.pkl', 'wb') as f:
            pickle.dump(noisy, f)

def trainHIRL(allShoppingLists, startIdx, endIdx):
    for i in range(startIdx, endIdx):
        # load the noisy trajectories file
        with open(f'experiment/runs/run_{i}/noisy_trajectories_{i}.pkl', 'rb') as f:
            noisyTrajectories = pickle.load(f)

        # get subgoals
        subgoals, segments_by_subgoal = getSubgoals(noisyTrajectories)
        learned_agents = learnSegments(subgoals, segments_by_subgoal, allShoppingLists[i], tol=0.2)

        saveLearnedAgents(learned_agents, file=f'experiment/runs/run_{i}/learned_agents_{i}.pkl')

def sampleIRLTrajectories(allShoppingLists, startIdx, endIdx):
    for i in range(startIdx, endIdx):
        learned_agents = loadLearnedAgents(allShoppingLists[i], file=f'experiment/runs/run_{i}/learned_agents_{i}.pkl')
        # TODO: should do this `m` times
        sampleTrajectory = generateLearnedTrajectory(
            learned_agents, 
            trajectoryPath=f'experiment/runs/run_{i}/irl_generated_trajectory_{i}.json',
            actionPath=f'experiment/runs/run_{i}/irl_generated_actions_{i}.json'
        )

        # load the expert trajectories for the plot
        with open(f'experiment/runs/run_{i}/noisy_trajectories_{i}.pkl', 'rb') as f:
            expertTrajectories = pickle.load(f)

        # get the subgoals from learned_agents
        subgoals = np.array([subgoal for (theta, learner, subgoal, initialState) in learned_agents])

        plotSampledTrajectory(sampleTrajectory, expertTrajectories, subgoals, START_STATE, imgPath=f'experiment/runs/run_{i}/irl_sampled_trajectory_{i}.png', showPlot=False)


def runHIRLSamples(allShoppingLists, startIdx, endIdx, headless=False):
    for i in range(startIdx, endIdx):
        print(f"Running sampled trajectory for shopping list: {allShoppingLists[i]}")
        args = ['python', 'socket_env.py', '--player_speed=0.25', f'--file=experiment/runs/run_{i}/start_state_{i}.txt']
        if headless:
            args.append('--headless')
        envProcess = subprocess.Popen(args)

        # load the generated trajectory to check if it succeeded
        success = False
        with open(f'experiment/runs/run_{i}/irl_generated_trajectory_{i}.json', 'r') as f:
            generatedTrajectory = json.load(f)
            flags = generatedTrajectory[-1][2:]
            success = np.all(np.array(flags))

        if not headless:
            time.sleep(5)  # wait for the env to start up
        
        args = [
            'python', 'run-generated-irl-trajectory.py',
            f'--file=experiment/runs/run_{i}/irl_generated_actions_{i}.json',
            f'--output=experiment/runs/run_{i}/irl_final_state_{i}.json',
            f'--run_id={i}',
            f'--metrics_file=experiment/runs/run_{i}/irl_generated_action_metrics_{i}.json'
        ]
        if success:
            args.append('--success')
        subprocess.run(args)

        envProcess.terminate()
        envProcess.wait()

def runExpertForEvaluation(startIdx, endIdx, headless=False):
    for i in range(startIdx, endIdx):
        args = ['python', 'socket_env.py', '--player_speed=0.25', f'--file=experiment/runs/run_{i}/start_state_{i}.txt']
        if headless:
            args.append('--headless')
        envProcess = subprocess.Popen(args)

        if not headless:
            time.sleep(5)  # wait for the env to start up
        genProcess = subprocess.run([
            'python', 'socket_agent_expert.py', 
            '--agent_epsilon=0.0',
            f'--metrics_output=experiment/runs/run_{i}/metrics_expert_evaluation_{i}.json',
        ])

        envProcess.terminate()
        envProcess.wait() 


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()
    
    if args.generate_lists:
        generateRandomLists(numLists=args.num_lists)
    
    # Load shopping lists and start state
    with open('experiment/random_shopping_lists.json', 'r') as f:
        randomLists = json.load(f)
    with open('experiment/start-state.json', 'r') as f:
        startState = json.load(f)

    startIdx = args.start_idx
    endIdx = args.end_idx if args.end_idx != -1 else len(randomLists)

    if args.generate_irl_trajectories:
        generateIRLTrajectories(randomLists, startIdx, endIdx, headless=args.headless)

    if args.add_noise:
        addNoiseForIRL(startIdx=startIdx, endIdx=endIdx)

    if args.run_irl:
        trainHIRL(randomLists, startIdx=startIdx, endIdx=endIdx)

    if args.sample_irl_trajectories:
        sampleIRLTrajectories(randomLists, startIdx=startIdx, endIdx=endIdx)

    if args.run_hirl_samples:
        runHIRLSamples(randomLists, startIdx=startIdx, endIdx=endIdx, headless=args.headless)

    if args.record_expert:
        runExpertForEvaluation(startIdx=startIdx, endIdx=endIdx, headless=args.headless)

    if args.run_baseline:
        raise Exception("Baseline not yet implemented")
        # TODO: this is where we will run our baseline in the environment and record the evaluation metrics



    
        

        
