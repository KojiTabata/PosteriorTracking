"""
Simulation experiments with the following settings:
mu_1, mu_2, ..., mu_{K-M} = np.linspace(0, \xi - \epsilon, K - M)
mu_{K-M+1}, ..., mu_{K} = np.linspace(\xi - \epsilon, 1.0, K - M)
"""


import multiprocessing
from itertools import combinations
import datetime
import os.path
import time
import pickle
import random

import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool

from policy import CTracking, PTracking, DTracking, ThompsonSampling
from policy import UGapE, LUCB, APT, UCB
from policy import StoppingConditionCB, StopByIdentification, BernoulliArm

DEBUG = True
# n_jobs = (multiprocessing.cpu_count() - 1) // 2
n_jobs = multiprocessing.cpu_count() - 2

SAVEDIR = "experiment1_result"


def experiment1(args):
    seed, pname, policy_class, sname, stop_class, delta, theta, K, L, m, mus = args
    policy = policy_class(K, theta, L, delta)
    print(f"{pname}+{sname}(K={K}, L={L}, m={m}, seed={seed}) delta={delta}")

    np.random.seed(seed)
    random.seed(seed)

    selected = []

    arms = [BernoulliArm(mu) for mu in mus]
    stop_cond = stop_class(delta, theta, K, L)
    t = 1

    t0 = time.time()

    extra_info = None

    while True:
        if stop_cond.elimination_type:
            uncertains = stop_cond.uncertains()
            i_t = policy.select(uncertains, nP=stop_cond.n_positive())
        else:
            i_t = policy.select()
        x_t = arms[i_t].draw()
        policy.update(i_t, x_t)
        stop_cond.update(i_t, x_t)

        res = stop_cond.check_finished()
        if res:
            print(f"{pname}(K={K}, L={L}, m={m}, seed={seed}) delta={delta}, t={t} (K,L,m)={(K,L,m)} (Finished! {res})")
            break
        selected.append((i_t, x_t))

        t += 1
        if DEBUG and np.random.random() * t < 1:
            print(pname, sname, t)
            print("draws:", policy.draws)
            print("rewards:", policy.rewards)

    T = t

    if hasattr(policy, "extra_info"):
        extra_info = policy.extra_info

    return T, res, selected, time.time() - t0, extra_info


def main():
    if DEBUG:
        Ks = [10]
    else:
        Ks = sorted([100, 50, 20, 10, 5])

    policies = {
                "DTracking": DTracking, "CTracking": CTracking, "PTracking": PTracking,
                "Thompson sampling": ThompsonSampling, "UGapE": UGapE, "LUCB": LUCB, "APT": APT, "UCB": UCB, }
    stopping = {"StoppingConditionCB": StoppingConditionCB, "StopByIdentification": StopByIdentification,}
    pt_st = [
        ("DTracking", "StoppingConditionCB"),
        ("CTracking", "StoppingConditionCB"),
        ("PTracking", "StoppingConditionCB"),
        ("Thompson sampling", "StopByIdentification"),
        ("UGapE", "StopByIdentification"),
        ("LUCB", "StopByIdentification"),
        ("APT", "StopByIdentification"),
        ("UCB", "StopByIdentification"),
    ]

    eps = 0.05
    theta = 0.5

    for K in Ks:
        if DEBUG:
            nrepeat = 3
        else:
            nrepeat = 100

        pickles = []

        for delta in [1e-16]:
            fp_pickle = f"{SAVEDIR}/result_exp1_[K={K},delta={delta}].pkl"

            args_list = []
            for m in [int(0.3*K), int(0.5*K), int(0.7*K)]:
                for L in [int(0.2*K), int(0.4*K), int(0.6*K), int(0.8*K)]:
                    mus = np.linspace(0, theta - eps, K - m)
                    mus = np.append(mus, np.linspace(theta + eps, 1.0, m))
                    mus = tuple(mus)

                    for i in range(nrepeat):
                        for pname, sname in pt_st:
                            policy_class = policies[pname]
                            stop_class = stopping[sname]

                            args = (i, pname, policy_class, sname, stop_class, delta, theta, K, L, m, mus)
                            args_list.append(args)

            if DEBUG:
                results = []
                for args in args_list:
                    res = experiment1(args)
                    results.append(res)
            else:
                with Pool(n_jobs) as p:
                    results = p.map(experiment1, args_list)

            d = {}
            for args, result in zip(args_list, results):
                i, pname, policy, sname, stop, delta, theta, K, L, m, mus = args
                key1 = (delta, theta, K)
                key2 = L
                key3 = m
                key4 = pname
                key5 = sname
                key6 = (mus, i)

                dd = d
                for key in [key1, key2, key3, key4, key5]:
                    if key not in dd:
                        dd[key] = {}
                    dd = dd[key]
                dd[key6] = result

            with open(fp_pickle, "wb") as f:
                pickle.dump(d, f)
            pickles.append(fp_pickle)

    return pickles



if __name__ == "__main__":
    if not os.path.exists(SAVEDIR):
        os.makedirs(SAVEDIR)

    main()



