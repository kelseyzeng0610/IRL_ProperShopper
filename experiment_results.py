import argparse
import pickle
import json
import numpy as np

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start_idx',
        type=int,
        help="starting index in existing shopping lists for experiment",
        default=0
    )
    parser.add_argument(
        '--end_idx',
        type=int,
        help="ending index in existing shopping lists for experiment",
        default=-1
    )

    return parser

def makeMetrics(successRate, violations, avgNumSteps, avgNumSubgoals, avgStepsBetweenSubgoals):
    return {
        'success_rate': successRate,
        'avg_violations_per_run': np.mean(violations),
        'median_violations_per_run': np.median(violations),
        'percent_violation_free': np.mean(violations == 0),
        'worst_violations': np.max(violations),
        'avg_num_steps': avgNumSteps,
        'avg_num_subgoals': avgNumSubgoals,
        'avg_steps_between_subgoals': avgStepsBetweenSubgoals,
    }

def evaluateExpertDemos(idx):
    # compute metrics on expert demos
    with open(f'experiment/runs/run_{idx}/metrics_expert_demos_{idx}.json', 'r') as f:
        expert_demos_metrics = json.load(f)
    success = np.array([run['success'] for run in expert_demos_metrics])
    successRate = np.mean(success)
    
    numViolations = np.array([run['num_violations'] for run in expert_demos_metrics])

    avgNumSteps = np.mean([run['num_steps'] for run in expert_demos_metrics])
    return makeMetrics(successRate, numViolations, avgNumSteps, -1, -1)

def evaluateExpertDeterministic(idx):
    # computes metrics on the actual expert results where epsilon=0
    with open(f'experiment/runs/run_{idx}/metrics_expert_evaluation_{idx}.json', 'r') as f:
        expert_eval_metrics = json.load(f)

    # this only has one run since it's deterministic, so the averages are just the values from that run
    expert_eval_metrics = expert_eval_metrics[0]
    return makeMetrics(
        successRate=1.0 if expert_eval_metrics['success'] else 0.0,
        violations=np.array([expert_eval_metrics['num_violations']]),
        avgNumSteps=expert_eval_metrics['num_steps'],
        avgNumSubgoals=-1,
        avgStepsBetweenSubgoals=-1
    )

def evaluateIRL(idx):
    # compute metrics on the IRL-generated trajectories
    with open(f'experiment/runs/run_{idx}/irl_generated_action_metrics_{idx}.json', 'r') as f:
        irl_metrics = json.load(f)
    run_violation_types = {}
    for run in irl_metrics:
        types = run.get('violation_types', {})
        for v, count in types.items():
            run_violation_types[v] = run_violation_types.get(v, 0) + count
            
            metrics['violation_types'] = run_violation_types
    return metrics
    # TODO: also need subgoal metrics
    return makeMetrics(
        successRate=np.mean(np.array([run['success'] for run in irl_metrics])),
        violations=np.array([run['num_violations'] for run in irl_metrics]),
        avgNumSteps=np.mean(np.array([run['num_steps'] for run in irl_metrics])),
        avgNumSubgoals=-1,
        avgStepsBetweenSubgoals=-1
    )

def aggregateResults(resultsByRun):
    modelKeys = ['demo_metrics', 'expert_metrics', 'irl_metrics', 'baseline_metrics']
    aggregated = {
        key: {
            'success_rate': np.mean([run[key]['success_rate'] for run in resultsByRun]),
            'avg_violations_per_run': np.mean([run[key]['avg_violations_per_run'] for run in resultsByRun]),
            'median_violations_per_run': np.median([run[key]['median_violations_per_run'] for run in resultsByRun]),
            'percent_violation_free': np.mean([run[key]['percent_violation_free'] for run in resultsByRun]),
            'worst_violations': int(np.max([run[key]['worst_violations'] for run in resultsByRun])),
            'avg_num_steps': np.mean([run[key]['avg_num_steps'] for run in resultsByRun]),
            'avg_num_subgoals': np.mean([run[key]['avg_num_subgoals'] for run in resultsByRun]),
            'avg_steps_between_subgoals': np.mean([run[key]['avg_steps_between_subgoals'] for run in resultsByRun]),
        } if resultsByRun[0][key] != {} else {} for key in modelKeys
    }
total_irl_violations = {}
    for run in resultsByRun:
        # Use .get() to safely access 'violation_types' in case it's missing from some runs
        v_types = run['irl_metrics'].get('violation_types', {})
        for v, count in v_types.items():
            total_irl_violations[v] = total_irl_violations.get(v, 0) + count
            
    if 'irl_metrics' in aggregated and aggregated['irl_metrics']:
        aggregated['irl_metrics']['total_violation_breakdown'] = total_irl_violations

    return aggregated



if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    with open('experiment/random_shopping_lists.json', 'r') as f:
        randomLists = json.load(f)
    numLists = len(randomLists)
    
    startIdx = args.start_idx
    endIdx = args.end_idx if args.end_idx != -1 else numLists

    print(f"Aggregating experiment results from run {startIdx} to {endIdx - 1}")
    results = []
    for i in range(startIdx, endIdx):
        # These will all be computed metrics for a single shopping list / run
        # Then we need to aggregate them at the end to get the overall metrics

        # compute metrics on the demos used for IRL training, from expert with epsilon > 0
        demoMetrics = evaluateExpertDemos(i)

        # compute metrics on expert with epsilon=0
        expertMetrics = evaluateExpertDeterministic(i)

        # compute metrics on irl-generated trajectories
        irlMetrics = evaluateIRL(i)

        # compute metrics on baseline IRL
        baselineMetrics = {} # TODO: need baseline 

        results.append({
            'run_id': i,
            'demo_metrics': demoMetrics,
            'expert_metrics': expertMetrics,
            'irl_metrics': irlMetrics,
            'baseline_metrics': baselineMetrics
        })

    # aggregate to get averages for each model
    aggregatedResults = aggregateResults(results)
    with open('experiment/aggregated_results.json', 'w') as f:
        json.dump(aggregatedResults, f, indent=2)

    print("Aggregated Results saved to experiment/aggregated_results.json")

