import argparse
import numpy as np
import json

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_lists',
        action='store_true',
        help="whether to generate new shopping lists",
    )
    parser.add_argument(
        '--num_lists',
        type=int,
        help="number of shopping lists to generate",
        default=50
    )
    parser.add_argument(
        '--generate_irl_trajectories',
        action='store_true',
        help="whether to generate noisy expert trajectories for IRL pipeline",
    )
    parser.add_argument(
        '--add_noise',
        action='store_true',
        help="whether to add noise to expert trajectories for IRL",
    )
    parser.add_argument(
        '--run_irl',
        action='store_true',
        help="whether to run the IRL pipeline",
    )
    parser.add_argument(
        '--sample_irl_trajectories',
        action='store_true',
        help="whether to sample trajectories from learned IRL agents",
    )
    parser.add_argument(
        '--run_hirl_samples',
        action='store_true',
        help="whether to run the HIRL-generated trajectories for evaluation",
    )
    parser.add_argument(
        '--record_expert',
        action='store_true',
        help="whether to record expert model behavior for evaluation",
    )
    parser.add_argument(
        '--run_baseline',
        action='store_true',
        help="whether to run the baseline IRL pipeline",
    )
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
    parser.add_argument(
        '--headless',
        action='store_true',
        help="whether to run the environment in headless mode",
    )

    return parser

def getShelfLocations():
    with open('sample-start-state.json', 'r') as f:
        state = json.load(f)

    shelfLocations = {}
    for shelf in state['shelves']:
        shelfName = shelf['food']
        shelfPosition = shelf['position']
        shelfTargetPosition = (shelfPosition[0] + 0.5 * shelf['width'], shelfPosition[1] + shelf['height'])
        shelfLocations[shelfName] = shelfTargetPosition
    
    return shelfLocations

def randomList(size, shelfNames):
    l = []
    seenItems = {}
    while len(l) < size:
        item = np.random.choice(shelfNames)
        if item not in seenItems:
            l.append(str(item))
            seenItems[item] = True

    return l

def buildRandomLists(numLists, listSize=3):
    shelfLocations = getShelfLocations()
    shelfNames = list(shelfLocations.keys())
    seen = {}
    randomLists = []
    while len(randomLists) < numLists:
        r = randomList(listSize, shelfNames)
        r_tuple = tuple(r)
        if r_tuple not in seen:
            randomLists.append(r)
            seen[r_tuple] = True

    return randomLists
