import numpy as np
import json
import subprocess
import time
from experiment_utils import buildRandomLists, getParser
import pickle
from pathlib import Path
from irl_agents_separate import getExpertTrajectoriesWithNoise, getSubgoals, learnSegments, generateLearnedTrajectory, loadLearnedAgents, saveLearnedAgents, plotSampledTrajectory, START_STATE
import concurrent.futures
import warnings
from irl_baseline import runBaseline

def generateRandomLists(numLists=50, listSize=3):
    randomLists = buildRandomLists(numLists, listSize)
    # make the first random list our default example
    randomLists[0] = ['sausage', 'milk', 'banana']
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

def trainSingleHIRL(i, shoppingList, verbose=False):

    # Suppress numpy warnings about divide by zero/overflow during IRL training - doesn't seem to be causing issues 
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    print(f"Starting training for run {i}: {shoppingList}")
    
    startTime = time.time()
    # load the noisy trajectories file
    with open(f'experiment/runs/run_{i}/noisy_trajectories_{i}.pkl', 'rb') as f:
        noisyTrajectories = pickle.load(f)

    # get subgoals
    subgoals, segments_by_subgoal = getSubgoals(noisyTrajectories)
    learned_agents = learnSegments(subgoals, segments_by_subgoal, shoppingList, tol=0.2, verbose=verbose)

    saveLearnedAgents(learned_agents, file=f'experiment/runs/run_{i}/learned_agents_{i}.pkl')
    
    endTime = time.time()
    trainingTime = endTime - startTime
    
    return {
        'run_id': i,
        'training_time': trainingTime,
        'num_subgoals': len(subgoals),
    }

def trainHIRL(allShoppingLists, startIdx, endIdx, max_workers, parallel=True, verbose=False):
    total_runs = endIdx - startIdx
    if parallel:
        startTime = time.time()
        print(f"\nStarting parallel training for {total_runs} runs with {max_workers} workers...")
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(startIdx, endIdx):
                futures.append(executor.submit(trainSingleHIRL, i, allShoppingLists[i], verbose=verbose))
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                elapsed = time.time() - startTime
                print(f"Progress: {completed}/{total_runs} complete ({100*completed/total_runs:.1f}%) - Run {result['run_id']} finished - Elapsed: {elapsed:.1f}s")
        endTime = time.time()
        
        # Calculate and save aggregate metrics
        total_training_time = sum(r['training_time'] for r in results)
        avg_training_time = total_training_time / len(results) if results else 0
        avg_subgoals = sum(r['num_subgoals'] for r in results) / len(results) if results else 0
        
        metrics = {
            'total_time_all_runs': endTime - startTime,
            'avg_training_time_per_run': avg_training_time,
            'avg_subgoals_per_run': avg_subgoals,
            'run_details': sorted(results, key=lambda x: x['run_id'])
        }
        
        with open('experiment/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"All HIRL training completed in {endTime - startTime:.2f} seconds")
        print(f"Average training time per run: {avg_training_time:.2f}s")
        print(f"Average number of subgoals: {avg_subgoals:.2f}")
    else:
        # Sequential fallback
        results = []
        for i in range(startIdx, endIdx):
            result = trainSingleHIRL(i, allShoppingLists[i], verbose=verbose)
            results.append(result)
            
        # Calculate and save aggregate metrics for sequential run too
        avg_training_time = sum(r['training_time'] for r in results) / len(results) if results else 0
        avg_subgoals = sum(r['num_subgoals'] for r in results) / len(results) if results else 0
        
        metrics = {
            'total_time_all_runs': sum(r['training_time'] for r in results), # approximate for sequential
            'avg_training_time_per_run': avg_training_time,
            'avg_subgoals_per_run': avg_subgoals,
            'run_details': results
        }
        
        with open('experiment/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

def sampleIRLTrajectories(allShoppingLists, startIdx, endIdx, numSamples=10):
    for i in range(startIdx, endIdx):
        # load learned agents
        learned_agents = loadLearnedAgents(allShoppingLists[i], file=f'experiment/runs/run_{i}/learned_agents_{i}.pkl')

        # get the subgoals from learned_agents
        subgoals = np.array([subgoal for (theta, learner, subgoal, initialState) in learned_agents])
        
        # load the expert trajectories for the plot
        with open(f'experiment/runs/run_{i}/noisy_trajectories_{i}.pkl', 'rb') as f:
            expertTrajectories = pickle.load(f)
        
        # Generate m sample trajectories with noise
        sampleTrajectories, sampleActions = [], []
        for m in range(numSamples):
            trajectoryPath = f'experiment/runs/run_{i}/irl_generated_trajectory_{i}_sample_{m}.json'
            actionPath = f'experiment/runs/run_{i}/irl_generated_actions_{i}_sample_{m}.json'
            sampleTrajectory = generateLearnedTrajectory(
                learned_agents, 
                trajectoryPath=trajectoryPath,
                actionPath=actionPath,
                epsilon=0.05,
            )

            # instead of keeping actions and trajectories in separate files, we can just make a list.
            sampleTrajectories.append(sampleTrajectory)
            with open(actionPath, 'r') as f:
                actions = json.load(f)
                sampleActions.append(actions)

            # remove the per-sample files
            Path(trajectoryPath).unlink()
            Path(actionPath).unlink()

            plotSampledTrajectory(sampleTrajectory, expertTrajectories, subgoals, START_STATE, imgPath=f'experiment/runs/run_{i}/irl_sampled_trajectory_{i}_sample_{m}.png', showPlot=False)

        # save all trajectories and actions to one file each
        with open(f'experiment/runs/run_{i}/irl_generated_trajectories_{i}.pkl', 'wb') as f:
            pickle.dump(sampleTrajectories, f)
        with open(f'experiment/runs/run_{i}/irl_generated_actions_{i}.json', 'w') as f:
            json.dump(sampleActions, f, indent=2)

        # then, generate deterministically
        trajectoryPath = f'experiment/runs/run_{i}/irl_generated_trajectory_{i}_deterministic.json'
        actionPath = f'experiment/runs/run_{i}/irl_generated_actions_{i}_deterministic.json'
        deterministicTrajectory = generateLearnedTrajectory(
            learned_agents, 
            trajectoryPath=trajectoryPath,
            actionPath=actionPath,
            epsilon=0.0,
        )
        plotSampledTrajectory(deterministicTrajectory, expertTrajectories, subgoals, START_STATE, imgPath=f'experiment/runs/run_{i}/irl_deterministic_trajectory_{i}.png', showPlot=False)


def runHIRLSamples(allShoppingLists, startIdx, endIdx, headless=False, numSamples=10):
    for i in range(startIdx, endIdx):
        print(f"Running {numSamples} sampled trajectories for shopping list: {allShoppingLists[i]}")
        args = ['python', 'socket_env.py', '--player_speed=0.25', f'--file=experiment/runs/run_{i}/start_state_{i}.txt', '--stay_alive']
        if headless:
            args.append('--headless')
        envProcess = subprocess.Popen(args)

        if not headless:
            time.sleep(5)  # wait for the env to start up
        
        allMetrics = []
        finalStates = []
        actionsFile = f'experiment/runs/run_{i}/irl_generated_actions_{i}.json'
        with open(actionsFile, 'r') as f:
            generatedActionsList = json.load(f)
        
        with open(f'experiment/runs/run_{i}/irl_generated_trajectories_{i}.pkl', 'rb') as f:
            generatedTrajectories = pickle.load(f)
        
        assert len(generatedActionsList) == numSamples
        assert len(generatedTrajectories) == numSamples
        
        for m in range(numSamples):
            generatedTrajectory = generatedTrajectories[m]
            flags = generatedTrajectory[-1][2:]
            success = np.all(np.array(flags))
            
            metricsFile = f'experiment/runs/run_{i}/irl_generated_action_metrics_{i}_sample_{m}.json'
            finalStateFile = f'experiment/runs/run_{i}/irl_final_state_{i}_sample_{m}.json'
            # TODO: generated actions script expects actions to exist in a single file
            tmpFile = f'experiment/runs/run_{i}/tmp_generated_actions_{i}_sample_{m}.json'
            with open(tmpFile, 'w') as f:
                json.dump(generatedActionsList[m], f, indent=2)
            args = [
                'python', 'run-generated-irl-trajectory.py',
                f'--file={tmpFile}',
                f'--output={finalStateFile}',
                f'--run_id={m}',
                f'--metrics_file={metricsFile}',
                f'--shopping_list={",".join(allShoppingLists[i])}'
            ]
            if success:
                args.append('--success')
            
            subprocess.run(args)  # Runs and waits for completion before next iteration

            # remove the temporary actions file
            Path(tmpFile).unlink()

            # it's silly to have a metrics file per sample, so aggregate all of them here
            with open(metricsFile, 'r') as f:
                sampleMetrics = json.load(f)
                allMetrics.append(sampleMetrics)
                # remove the file
                Path(metricsFile).unlink()

            # same deal with the final state jsons
            with open(finalStateFile, 'r') as f:
                finalState = json.load(f)
                finalStates.append(finalState)
                # remove the file
                Path(finalStateFile).unlink()

        # at the end, save all metrics and final states to one file each
        with open(f'experiment/runs/run_{i}/irl_generated_action_metrics_{i}.json', 'w') as f:
            json.dump(allMetrics, f, indent=2)
        with open(f'experiment/runs/run_{i}/irl_final_states_{i}.json', 'w') as f:
            json.dump(finalStates, f, indent=2)

        # evaluate the deterministic ones
        deterministicActionsFile = f'experiment/runs/run_{i}/irl_generated_actions_{i}_deterministic.json'
        deterministicFinalStateFile = f'experiment/runs/run_{i}/irl_deterministic_final_state_{i}.json'
        deterministicMetricsFile = f'experiment/runs/run_{i}/irl_deterministic_action_metrics_{i}.json'
        args = [
            'python', 'run-generated-irl-trajectory.py',
            f'--file={deterministicActionsFile}',
            f'--output={deterministicFinalStateFile}',
            f'--run_id=99',
            f'--metrics_file={deterministicMetricsFile}',
            f'--shopping_list={",".join(allShoppingLists[i])}'
        ]
        subprocess.run(args)

        envProcess.terminate()
        envProcess.wait()

def recomputeHIRLMetrics(allShoppingLists, startIdx, endIdx, numSamples=10):
    # load the final state and compute the success metric again because there was a bug
    for i in range(startIdx, endIdx):
        shoppingList = allShoppingLists[i]
        finalStatesFile = f'experiment/runs/run_{i}/irl_final_states_{i}.json'
        metricsFile = f'experiment/runs/run_{i}/irl_generated_action_metrics_{i}.json'
        with open(finalStatesFile, 'r') as f:
            finalStates = json.load(f)

        with open(metricsFile, 'r') as f:
            allMetrics = json.load(f)

        for m in range(numSamples):
            finalState = finalStates[m]
            purchased_contents = finalState['observation']['baskets'][0]['purchased_contents'] if len(finalState['observation']['baskets']) > 0 else []
            unpurchased_contents = finalState['observation']['baskets'][0]['contents'] if len(finalState['observation']['baskets']) > 0 else []

            # success if everything in the shopping list is purchased
            success = all(item in purchased_contents for item in shoppingList)

            allMetrics[m]['success'] = success
            allMetrics[m]['paid_items'] = purchased_contents
            allMetrics[m]['unpaid_items'] = unpurchased_contents

        with open(metricsFile, 'w') as f:
            json.dump(allMetrics, f, indent=2)

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

def runBaselineEvaluation(allShoppingLists, startIdx, endIdx, headless=True):
    for i in range(startIdx, endIdx):
        args = ['python', 'socket_env.py', '--player_speed=0.25', f'--file=experiment/runs/run_{i}/start_state_{i}.txt']
        if headless:
            args.append('--headless')
        envProcess = subprocess.Popen(args)

        shoppingList = allShoppingLists[i]
        traj, actions = runBaseline(shoppingList=shoppingList, thetaFile="experiment/baseline_theta.json", verbose=False)

        actionsFile = f'experiment/runs/run_{i}/baseline_actions_{i}.json'
        with open(f'experiment/runs/run_{i}/baseline_trajectory_{i}.json', 'w') as f:
            json.dump([step.tolist() for step in traj], f)
        with open(actionsFile, 'w') as f:
            json.dump([int(a) for a in actions], f)

        if not headless:
            time.sleep(5)  # wait for the env to start up

        genProcess = subprocess.run([
            'python', 'run-generated-irl-trajectory.py',
            f'--file={actionsFile}',
            f'--output=experiment/runs/run_{i}/baseline_final_state_{i}.json',
            f'--run_id={i}',
            f'--metrics_file=experiment/runs/run_{i}/baseline_action_metrics_{i}.json',
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
    verbose = args.verbose

    if args.generate_irl_trajectories:
        generateIRLTrajectories(randomLists, startIdx, endIdx, headless=args.headless)

    if args.add_noise:
        addNoiseForIRL(startIdx=startIdx, endIdx=endIdx)

    if args.run_irl:
        trainHIRL(randomLists, startIdx=startIdx, endIdx=endIdx, verbose=verbose, max_workers=12)

    if args.sample_irl_trajectories:
        sampleIRLTrajectories(randomLists, startIdx=startIdx, endIdx=endIdx)

    if args.run_hirl_samples:
        runHIRLSamples(randomLists, startIdx=startIdx, endIdx=endIdx, headless=args.headless)

    if args.recompute_hirl_metrics:
        recomputeHIRLMetrics(randomLists, startIdx=startIdx, endIdx=endIdx)

    if args.record_expert:
        runExpertForEvaluation(startIdx=startIdx, endIdx=endIdx, headless=args.headless)

    if args.run_baseline:
        runBaselineEvaluation(randomLists, startIdx=startIdx, endIdx=endIdx, headless=args.headless)
