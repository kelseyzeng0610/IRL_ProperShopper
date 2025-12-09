import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
from irl_agents_separate import plotSampledTrajectory, START_STATE, THETA_SIZE, getExpertTrajectoriesWithNoise
from enum import Enum

def makePlot(expertTrajectories, startState, sampleTrajectory=None, subgoals=None, title=""):
    # plot the expert demos
    plt.figure(figsize=(10, 8))
    for i, traj in enumerate(expertTrajectories):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], 'b.-', alpha=0.2, linewidth=0.5,
                label='Expert Trajectory' if i == 0 else "")
    
    if sampleTrajectory is not None:
        x, y = [step[0] for step in sampleTrajectory], [step[1] for step in sampleTrajectory]
        plt.plot(x, y, 'o-', color='purple', linewidth=2, markersize=4, label='Per-Subgoal Agent Trajectory')

    if subgoals is not None:    
        plt.scatter(subgoals[:, 0], subgoals[:, 1], c='red', s=250, marker='*',
                label='Subgoals', zorder=10)
    
        # Add numbers to subgoals
        for i, subgoal in enumerate(subgoals):
            plt.text(subgoal[0] + 0.2, subgoal[1] + 0.2, str(i+1), fontsize=12, fontweight='bold',
                    ha='center', va='center', color='black', zorder=11,
                    bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5))
    

    plt.scatter(startState[0], startState[1], c='green', s=200, marker='o',
                label='Start', zorder=10)
    
    plt.title(title)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("presentation-plots/expert_trajectories.png")
    plt.show()

def makeSuccessMetricsPlot(aggregated_results, modelNames):
    # plot 1: success rates, percentage of violation free runs
    modelTypes = [modelNames[k] for k in aggregated_results]
    successRates = [result['success_rate'] for _, result in aggregated_results.items()]
    violationFreeRates = [result['percent_violation_free'] for _, result in aggregated_results.items()]

    barWidth = 0.35

    x = np.arange(len(modelTypes))

    fig, ax = plt.subplots(figsize=(10, 6))
    successBars = ax.bar(x - barWidth/2, successRates, width=barWidth, label='Success Rate')
    violationFreeBars = ax.bar(x + barWidth/2, violationFreeRates, width=barWidth, label='Violation-Free Rate')

    ax.set_ylabel('Rate')
    ax.set_xlabel('Model Type')
    ax.set_title('Success and Violation-Free Rates by Model Type')
    ax.set_xticks(x)
    ax.set_ylim(0, 1.1)
    ax.set_xticklabels(modelTypes, rotation=45, ha='right')
    ax.legend(loc="upper center",)

    for bars in [successBars, violationFreeBars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("presentation-plots/success_metrics.png")
    plt.close()

def makeViolationMetricsPlot(aggregated_results, modelNames):
    # plot 2: avg violations per run, median violations per run, max violations per run
    modelTypes = [modelNames[k] for k in aggregated_results]
    avgViolations = [result['avg_violations_per_run'] for _, result in aggregated_results.items()]
    medianViolations = [result['median_violations_per_run'] for _, result in aggregated_results.items()]
    maxViolations = [result['worst_violations'] for _, result in aggregated_results.items()]
    maxViolationsClipped = [np.clip(v, 0, 50) for v in maxViolations]

    barWidth = 0.25
    x = np.arange(len(modelTypes))

    fig, ax = plt.subplots(figsize=(10, 6))
    avgBars = ax.bar(x - barWidth, avgViolations, width=barWidth, label='Avg Violations per Run')
    medianBars = ax.bar(x, medianViolations, width=barWidth, label='Median Violations per Run')
    maxBarsClipped = ax.bar(x + barWidth, maxViolationsClipped, width=barWidth, label='Max Violations in a Run')
    ax.set_ylabel('Number of Violations')
    ax.set_xlabel('Model Type')
    ax.set_title('Violation Metrics by Model Type')
    ax.set_xticks(x)
    ax.set_xticklabels(modelTypes, rotation=45, ha='right')
    ax.legend(loc="upper center",)

    for bars, values in [(avgBars, avgViolations), (medianBars, medianViolations), (maxBarsClipped, maxViolations)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            clipped = height < value
            if clipped:
                bar.set_hatch('//')  # Indicate clipping with hatch
            text_color = 'red' if clipped else 'black'
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, f"{value:.2f}", ha='center', va='bottom', color=text_color)
    
    plt.tight_layout()
    plt.savefig("presentation-plots/violation_metrics.png")
    plt.close()

def makeEfficiencyPlot(aggregated_results, modelNames):
    # plot 3: avg num steps, number of subgoals
    modelTypes = [modelNames[k] for k in aggregated_results]
    avgNumSteps = [result['avg_num_steps'] for _, result in aggregated_results.items()]
    numSubgoals = [result.get('avg_num_subgoals', 0) for _, result in aggregated_results.items()]

    barWidth = 0.35
    x = np.arange(len(modelTypes))
    fig, ax = plt.subplots(figsize=(10, 6))
    stepsBars = ax.bar(x - barWidth/2, avgNumSteps, width=barWidth, label='Avg Number of Steps')
    # subgoalBars = ax.bar(x + barWidth/2, numSubgoals, width=barWidth, label='Avg Number of Subgoals')

    ax.set_ylabel('Count')
    ax.set_xlabel('Model Type')
    ax.set_title('Efficiency Metrics by Model Type')
    ax.set_xticks(x)
    ax.set_xticklabels(modelTypes)
    ax.legend(loc="upper right",)
    for bar in stepsBars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("presentation-plots/efficiency_metrics.png")
    plt.close()

class ViolationType(Enum):
    NONE = 0,
    RanIntoCheckout = 1,
    RanIntoShelf = 2,
    RanIntoWall = 3,
    RanIntoCartReturn = 4
    RanIntoBasketReturn = 5
    TookItemNotOnList = 6
    PutItemOnShelf = 7
    TookMoreThanNeeded = 8

def getViolationTypeFromString(violationStr):
    if 'ran into a checkout counter' in violationStr:
        return ViolationType.RanIntoCheckout
    elif 'ran into' in violationStr and 'shelf' in violationStr:
        return ViolationType.RanIntoShelf
    elif 'ran into the fresh fish counter' in violationStr:
        return ViolationType.RanIntoShelf
    elif 'ran into a wall' in violationStr:
        return ViolationType.RanIntoWall
    elif 'ran into the cart return' in violationStr:
        return ViolationType.RanIntoCartReturn
    elif 'ran into the basket return' in violationStr:
        return ViolationType.RanIntoBasketReturn
    elif 'that is not on their shopping list' in violationStr:
        return ViolationType.TookItemNotOnList
    elif 'put the' in violationStr and 'on the' in violationStr and 'shelf' in violationStr:
        return ViolationType.PutItemOnShelf
    elif 'more' in violationStr and 'than they needed' in violationStr:
        return ViolationType.TookMoreThanNeeded
    else:
        raise Exception(f"Unknown violation string: {violationStr}")

def getViolationBuckets(details):
    buckets = {
        ViolationType.RanIntoCheckout: 0,
        ViolationType.RanIntoShelf: 0,
        ViolationType.RanIntoWall: 0,
        ViolationType.RanIntoCartReturn: 0,
        ViolationType.RanIntoBasketReturn: 0,
        ViolationType.TookItemNotOnList: 0,
        ViolationType.PutItemOnShelf: 0,
        ViolationType.TookMoreThanNeeded: 0,
    }
    for violation, count in details.items():
        violationType = getViolationTypeFromString(violation)
        buckets[violationType] += count
    return buckets
        

def makeViolationDetailPlot(aggregated_results, modelNames):
    # plot 4: violation details per model
    baselineDetails = aggregated_results['baseline_metrics']['total_violation_breakdown']
    irlDetails = aggregated_results['irl_metrics']['total_violation_breakdown']
    detIRLDetails = aggregated_results['deterministic_irl_metrics']['total_violation_breakdown']

    baselineViolationsByType = getViolationBuckets(baselineDetails)
    irlViolationsByType = getViolationBuckets(irlDetails)
    detIRLViolationsByType = getViolationBuckets(detIRLDetails)

    # Get violation type labels
    violationTypes = [
        'Ran Into\nCheckout',
        'Ran Into\nShelf',
        'Ran Into\nWall',
        'Ran Into\nCart Return',
        'Ran Into\nBasket Return',
        'Took Item\nNot On List',
        'Put Item\nOn Shelf',
        'Took More\nThan Needed'
    ]
    
    # Get counts for each violation type
    baselineCounts = [baselineViolationsByType[vt] for vt in ViolationType if vt != ViolationType.NONE]
    irlCounts = [irlViolationsByType[vt] for vt in ViolationType if vt != ViolationType.NONE]
    detIRLCounts = [detIRLViolationsByType[vt] for vt in ViolationType if vt != ViolationType.NONE]

    barWidth = 0.25
    x = np.arange(len(violationTypes))

    fig, ax = plt.subplots(figsize=(12, 6))
    baselineBars = ax.bar(x - barWidth, baselineCounts, width=barWidth, label=modelNames['baseline_metrics'])
    irlBars = ax.bar(x, irlCounts, width=barWidth, label=modelNames['irl_metrics'])
    detIRLBars = ax.bar(x + barWidth, detIRLCounts, width=barWidth, label=modelNames['deterministic_irl_metrics'])

    ax.set_ylabel('Total Violations')
    ax.set_xlabel('Violation Type')
    ax.set_title('Violation Breakdown by Type')
    ax.set_xticks(x)
    ax.set_xticklabels(violationTypes, rotation=45, ha='right')
    ax.legend(loc="upper right")

    # Add value labels on bars
    for bars in [baselineBars, irlBars, detIRLBars]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{int(height)}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("presentation-plots/violation_details.png")
    plt.close()

def makePartialSuccessPlot(aggregated_results, modelNames):
    irl_summary = aggregated_results['irl_metrics']['error_analysis_summary']['summary']
    
    total_failed_to_pay = irl_summary['total_failed_to_pay']
    total_failed_to_get_all_items = irl_summary['total_failed_to_get_all_items']
    
    avg_items_when_failed_to_pay = irl_summary['avg_items_in_basket_when_failed_to_pay']
    avg_missing_items = irl_summary['avg_missing_items_when_failed_to_get_all_items']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    categories1 = ['Failed to\nPay', 'Failed to Get\nAll Items']
    counts = [total_failed_to_pay, total_failed_to_get_all_items]
    x1 = np.arange(len(categories1))
    
    bars1 = ax1.bar(x1, counts, width=0.6, color=['#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Number of Runs')
    ax1.set_xlabel('Failure Type')
    ax1.set_title('Partial Success: Failure Counts')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(categories1)
    
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{int(height)}", 
                    ha='center', va='bottom', fontsize=10)
    
    categories2 = ['Avg Items in Basket\n(When Failed to Pay)', 'Avg Missing Items\n(When Failed to Get All)']
    averages = [avg_items_when_failed_to_pay, avg_missing_items]
    x2 = np.arange(len(categories2))
    
    bars2 = ax2.bar(x2, averages, width=0.6, color=['#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Average Count')
    ax2.set_xlabel('Metric Type')
    ax2.set_ylim(0, 2.0)
    ax2.set_title('Partial Success: Item Details')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories2)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.05, f"{height:.2f}", 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("presentation-plots/partial_success.png")
    plt.close()


if __name__ == "__main__":
    # load trajectories
    expertTrajectories = getExpertTrajectoriesWithNoise("trajectories.pkl")
    startState = START_STATE

    # just the expert demos
    # makePlot(expertTrajectories, startState, title="Expert Trajectories Visualization")


    # plots from the aggregated results
    with open('experiment/aggregated_results.json', 'r') as f:
        aggregated_results = json.load(f)

    modelNames = {
        "expert_metrics": "Expert Deterministic",
        "demo_metrics": "Expert Demonstrations",
        "baseline_metrics": "Single IRL Agent Baseline",
        "irl_metrics": "HIRL Pipeline with noise",
        "deterministic_irl_metrics": "HIRL Deterministic Pipeline",
    }

    makeSuccessMetricsPlot(aggregated_results, modelNames)
    makeViolationMetricsPlot(aggregated_results, modelNames)
    makeEfficiencyPlot(aggregated_results, modelNames)
    makeViolationDetailPlot(aggregated_results, modelNames)
    makePartialSuccessPlot(aggregated_results, modelNames)

    



    
    