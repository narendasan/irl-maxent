import pandas as pd
from itertools import combinations, product
from sympy import symbols, Eq, solve, sqrt
from typing import List, Callable, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import uuid
import scipy

from canonical_task_generation_sim_exp.lib.arguments import parser, out_path
from canonical_task_generation_sim_exp.canonical_task_search.task import RIRLTask
from canonical_task_generation_sim_exp.generate_canonical_task_archive import load_score_span_tasks as load_canoncial_tasks
from canonical_task_generation_sim_exp.generate_complex_task_archive import load_tasks as load_complex_tasks

def gen_system(A: np.array, B: np.array) -> Callable:
    def system(w: np.array) -> List[np.array]:
        def decision_boundry(W: np.array):
            return A @ W - B @ W

        def on_sphere(W: np.array):
            return np.sum(np.square(W)) - 1

        return [
            decision_boundry(w),
            on_sphere(w)
        ]
    return system

# What if we changed the coordinate bases to something like polar, easy to figure out directions on the unit sphere then
def find_decision_boundries(task: RIRLTask) -> List[List]:
    feats = task.possible_cumulative_features.values()
    #print(len(feats))
    sols = []
    for a,b in combinations(feats,2):
        if all([x > y for x, y in zip(a, b)]) or all([x < y for x, y in zip(a, b)]):
            continue
        else:
            pair = np.stack([a,b])
            # w1 = symbols('w1')
            # When expanding to ND.. need more eqs???
            # https://www.savarese.org/math/hypersphere.html
            sol = scipy.optimize.fsolve(gen_system(a, b), np.eye(pair.shape[1])[0])
            #eq1 = Eq(a[0] * w1 + a[1] * (1- w1 ** 2), b[0] * w1 + b[1] * (1- w1 ** 2))
            #sol = solve((eq1,), (w1,), quick=True, particular=True)

            if all([s >= 0 and s <= 1 for s in sol]):
                sols.append(sol)

    return sols



def plot_decision_boundries(args, canonical_task_def, complex_task_def):
    num_canonical_actions = len(canonical_task_def["features"])
    num_complex_actions = len(complex_task_def["features"])
    print(f"Number of canonical task actions {num_canonical_actions}, complex task actions {num_complex_actions}")

    canonical_task = RIRLTask(features=canonical_task_def["features"],
                              preconditions=canonical_task_def["preconditions"])
    complex_task = RIRLTask(features=complex_task_def["features"],
                            preconditions=complex_task_def["preconditions"])

    canonical_task_dbs = find_decision_boundries(canonical_task)
    complex_task_dbs = find_decision_boundries(complex_task)

    theta = np.linspace(0, 2*np.pi, 150)
    radius = 1
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    plt.plot(a, b)
    if len(canonical_task_dbs) > 0:
        plt.scatter(*zip(*canonical_task_dbs))
    if len(complex_task_dbs) > 0:
        plt.scatter(*zip(*complex_task_dbs))

    plt.gca().set_aspect('equal')
    plt.savefig(out_path(args, kind="figures", owner="decision_boundries")/f"{uuid.uuid4()}_can_act{num_canonical_actions}_comp_act{num_complex_actions}")
    plt.clf()

def dist_fn(A: List[float], B: List[float]):
    return np.sqrt(sum([(b - a) ** 2 for a, b in zip(A, B)]))

def calc_sim_score(can_dbs, comp_dbs) -> Dict[Tuple[List[float], List[float]], float]:
    sim_scores = {}
    for can_db, comp_db in product(can_dbs, comp_dbs):
        if can_db[1][0] < comp_db[0][0] and can_db[1][1] > comp_db[0][1]:
            sim_scores[(can_db, comp_db)] = 0.0
        elif comp_db[1][0] < comp_db[0][0] and comp_db[1][1] > comp_db[0][1]:
            sim_scores[(can_db, comp_db)] = 0.0
        else:
            prob_can_r = dist_fn(can_db[0], can_db[1]) / (np.pi / 2)
            prob_comp_r = dist_fn(comp_db[0], comp_db[1]) / (np.pi / 2)
            sim_scores[(can_db, comp_db)] = (prob_can_r * prob_comp_r) / prob_comp_r

def bets_scoring(canonical_task_def, complex_task_def):
    can_dbs = canonical_task_def["decision_boundries"]
    comp_dbs = complex_task_def["decision_boundries"]

    sen_can_dbs = [[1, 0]] + can_dbs + [[0, 1]]
    sen_comp_dbs =  [[1, 0]] + comp_dbs + [[0, 1]]


    sim_score = calc_sim_score(sen_can_dbs, sen_comp_dbs)

    can_probs = dict(map(lambda a, b: ((a,b), (dist_fn(a, b) / (np.pi / 2))), zip(sen_can_dbs[:-1], sen_can_dbs[1:])))
    comp_probs = dict(map(lambda a, b: (dist_fn(a, b) / (np.pi / 2)), zip(sen_can_dbs[:-1], sen_can_dbs[1:])))

    region_scores = []
    for r in can_probs.keys():
        score = can_probs[r] * sum(s*)

    score = sum(can_probs) * sum([s ** 2] for _,s in sim_score) # p(c_i|c_j) == S(c_i|c_j)


def main(args, canonical_tasks: pd.DataFrame, complex_tasks: pd.DataFrame):
    canonical_task_dbs = []
    for _, con_t in canonical_tasks.iterrows():
        canonical_task = RIRLTask(features=con_t["features"],
                                  preconditions=con_t["preconditions"])
        canonical_task_dbs.append(find_decision_boundries(canonical_task))
    canonical_tasks.insert(loc=len(canonical_tasks.columns), column="decision_boundries", value=canonical_task_dbs)

    print(canonical_tasks)

    complex_task_dbs = []
    for _, comp_t in complex_tasks.iterrows():
        complex_task = RIRLTask(features=comp_t["features"],
                                preconditions=comp_t["preconditions"])
        complex_task_dbs.append(find_decision_boundries(complex_task))
    complex_tasks.insert(loc=len(complex_tasks.columns), column="decision_boundries", value=complex_task_dbs)

    print(complex_tasks)

    for _, comp_t in complex_tasks.iterrows():
        for _, con_t in canonical_tasks.iterrows():
            bets_scoring(con_t, comp_t)
            #plot_decision_boundries(args, con_t, comp_t)

    return

if __name__ == "__main__":
    args = parser.parse_args()
    CANONICAL_TASKS = load_canoncial_tasks("score_spanning", args)
    COMPLEX_TASKS = load_complex_tasks(args)
    #main(CANONICAL_TASKS, COMPLEX_TASKS)
    main(args, CANONICAL_TASKS, COMPLEX_TASKS)
