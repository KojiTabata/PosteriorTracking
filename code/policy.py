from itertools import combinations
import datetime
import os.path
import time
import pickle
import random
import math
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool
import gurobipy as gp
from util import get_d, BernoulliArm

DEBUG = False


def get_l_star(mus, m, L, theta, ds):
    min_v, min_l = None, L
    for l in range(L, m + 1):
        if mus[l-1] == theta:
            break
        v = 1.0/(l - L + 1) * np.sum(1/ds[:l])
        if min_v is None or v < min_v:
            min_v = v
            min_l = l
    return min_l


def get_w_new(mus, L, theta, eps=0.0):
    s = get_S(mus, L, theta)
    ws = s / np.sum(s)
    K = s.size

    if eps > 0:
        if eps == 1.0/K:
            return np.zeros(K) + 1.0/K
        # L^inf projection
        # minimize_{w^{\epsilon} \in \Sigma_{\epsilon}} ||w^{\epsilon} - w^*||_{\infty}
        ws = np.array(ws)
        while np.any(np.array(ws) < eps):
            ws1 = ws.copy()
            ws1[ws1 < eps] = eps
            ws1[ws1 > eps] -= np.sum(ws1 - ws) / np.sum(ws1 > eps)
            ws = ws1 / np.sum(ws1)
        return ws

    return ws


def get_S(mus, L, theta):
    mus = np.array(mus)
    K = mus.size
    m = np.sum(mus >= theta)
    if m < L:
        # negative case
        L = K - L + 1
        theta = 1 - theta
        mus = 1 - mus
        m = np.sum(mus >= theta)
    sort_order = np.argsort(mus)[::-1]
    sort_inverse = np.argsort(sort_order)
    mus = mus[sort_order]
    ds = np.array([get_d(mu, theta) for mu in mus])
    l_star = get_l_star(mus, m, L, theta, ds)

    s = np.zeros(K)
    for i in range(l_star):
        if ds[i] == 0:
            s[i] = sys.float_info.max / K
        else:
            s[i] = 1.0 / (l_star - L + 1) / ds[i]

    return s[sort_inverse]


#%% Stopping Conditions

class StoppingConditionCB:
    def __init__(self, delta, theta, K, L):
        self.delta = delta
        self.K = K
        self.ndraw = [0]*K
        self.reward = [0.0]*K
        self.theta = theta
        self.L = L
        self.t = 1
        self.elimination_type = False

    def update(self, i, x):
        self.ndraw[i] += 1
        self.reward[i] += x
        self.t += 1

    def check_oneside(self, theta, L, mus):
        beta = math.log(math.log(1+self.t)/self.delta)
        K = self.K

        Zs = []
        for i in range(K):
            mu = mus[i]
            n = self.ndraw[i]
            if mu >= theta:
                Zs.append(n * get_d(mu, theta))
            else:
                Zs.append(0)

        Zs = np.array(Zs)
        ord_index = np.argsort(Zs)[::-1]

        if DEBUG:
            print(Zs)

        if np.sum(Zs[ord_index[L - 1:K]]) >= beta:
            return True

        return False

    def check_finished(self):
        mus = [self.reward[i] / self.ndraw[i] if self.ndraw[i] > 0 else self.theta for i in range(self.K)]
        if self.check_oneside(self.theta, self.L, mus):
            return "P"

        if self.check_oneside(1 - self.theta, self.K - self.L + 1, [1 - v for v in mus]):
            return "N"

        return False


class StopByIdentification:
    def __init__(self, delta, theta, K, L):
        self.delta = delta
        self.K = K
        self.ndraw = [0]*K
        self.reward = [0.0]*K
        self.theta = theta
        self.L = L
        self.t = 1
        self.elimination_type = True
        self.lil = True
        self.remain = list(range(K))
        self.P = set()
        self.N = set()

    def uncertains(self):
        return self.remain

    def n_positive(self):
        return len(self.P)

    def update(self, i, x):
        self.ndraw[i] += 1
        self.reward[i] += x
        self.t += 1
        if self.t >= self.K:
            n = self.ndraw[i]
            beta = np.log(np.log(self.t + 1)/self.delta)
            mu = self.reward[i] / self.ndraw[i]
            if get_d(mu, self.theta) * n > beta:
                if mu > self.theta:
                    self.P.add(i)
                    self.remain.remove(i)
                else:
                    self.N.add(i)
                    self.remain.remove(i)

    def check_finished(self):
        if len(self.P) >= self.L:
            return "P"

        if len(self.N) > self.K - self.L:
            return "N"

        return False


#%% Arm Selection Methods

class PTracking:
    def __init__(self, K, theta, L, delta):
        self.K = K
        self.rewards = [0.0 for i in range(K)]
        self.draws = [0 for i in range(K)]
        self.theta = theta
        self.L = L
        self.t = 1
        self.delta = delta
        self.abs = [[1.0, 1.0] for i in range(K)]
        self.extra_info = []

    def get_mu(self, i):
        n = self.draws[i]
        if n > 0:
            mu = self.rewards[i] / n
        else:
            mu = 0
        return mu

    def get_mus(self):
        return [self.get_mu(i) for i in range(self.K)]

    def update(self, i, x):
        self.rewards[i] += x
        self.draws[i] += 1
        self.abs[i][0] += x
        self.abs[i][1] += 1 - x
        self.t += 1

    def beta(self):
        return math.log(math.log(1 + self.t) / self.delta)

    def select(self):
        # select each arm once
        if self.t <= self.K:
            return self.t - 1

        arms = list(range(self.K))
        mus_tilde = np.array([np.random.beta(*self.abs[i]) for i in arms])
        S = get_S(mus_tilde, self.L, self.theta)
        i_t = np.argmax(S/self.draws)

        return i_t


class DTracking:
    def __init__(self, K, theta, L, delta):
        self.delta = delta
        self.K = K
        self.rewards = [0.0 for i in range(K)]
        self.draws = [0 for i in range(K)]
        self.theta = theta
        self.L = L
        self.t = 1

    def get_mu(self, i):
        n = self.draws[i]
        if n > 0:
            mu = self.rewards[i] / n
        else:
            mu = 0
        return mu

    def update(self, i, x):
        self.rewards[i] += x
        self.draws[i] += 1
        self.t += 1

    def select(self, arms=None, nP=0):
        if arms is None:
            arms = list(range(self.K))

        for i in arms:
            if self.draws[i] == 0:
                return i

        if len(arms) == 1:
            return list(arms)[0]

        n = np.sqrt(self.t) - self.K / 2
        U = [(self.draws[i], i) for i in arms if self.draws[i] < n]
        if U:
            return min(U)[1]

        mus = [self.get_mu(i) for i in arms]
        S = get_S(mus, self.L - nP, self.theta)
        ws = S / np.sum(S)
        # ws = get_w_new(mus, self.L - nP, self.theta)
        vs = [(self.t * ws[i] - self.draws[j], j) for i, j in enumerate(arms)]

        return max(vs)[1]


class CTracking:
    def __init__(self, K, theta, L, delta):
        self.delta = delta
        self.K = K
        self.rewards = [0.0 for i in range(K)]
        self.draws = [0 for i in range(K)]
        self.ws_sum = np.zeros(self.K)
        self.theta = theta
        self.L = L
        self.t = 1

    def get_mu(self, i):
        n = self.draws[i]
        if n > 0:
            mu = self.rewards[i] / n
        else:
            mu = 0
        return mu

    def get_mus(self):
        return [self.get_mu(i) for i in range(self.K)]

    def update(self, i, x):
        self.rewards[i] += x
        self.draws[i] += 1

        self.t += 1

    def select(self, arms=None, nP=0):
        if arms is None:
            arms = list(range(self.K))

        for i in arms:
            if self.draws[i] == 0:
                return i

        arms = list(arms)
        mus = [self.get_mu(i) for i in arms]
        eps = 0.5 / (self.K*self.K + self.t)**0.5

        ws = get_w_new(mus, self.L - nP, self.theta, eps)
        self.ws_sum[arms] += ws

        if len(arms) == 1:
            return list(arms)[0]

        return max([(self.ws_sum[i] - self.draws[i], i) for i in arms])[1]


class ThompsonSampling:
    def __init__(self, K, theta, L, delta):
        self.delta = delta
        self.K = K
        self.abs = np.ones((K, 2))
        self.theta = theta
        self.rewards = [0.0 for i in range(K)]
        self.draws = [0 for i in range(K)]
        self.L = L

    def update(self, i, x):
        self.rewards[i] += x
        self.draws[i] += 1
        self.abs[i, 0] += x
        self.abs[i, 1] += 1 - x

    def select(self, arms=None, nP=0):
        if arms is None:
            arms = list(range(self.K))
        psamples = [(np.random.beta(*self.abs[i, :]), i) for i in arms]
        p = [i for v, i in psamples if v > self.theta]
        m = len(p)

        if m + nP >= self.L:
            return max(psamples)[1]
        else:
            return min(psamples)[1]


class UGapE:
    def __init__(self, K, theta, L, delta):
        self.K = K
        self.L = L
        self.delta = delta
        self.b = 1
        self.c = 0.5
        self.rewards = [0.0 for i in range(K)]
        self.draws = [0 for i in range(K)]
        self.t = 1

    def get_lucb(self, mu, n):
        # solve for increasing function f(x) = 0
        # this code assumes f(x) < 0, f(y) > 0
        def bisection_search(f, x, y):
            while True:
                m = (x + y) / 2
                if f(m) > 0:
                    y = m
                else:
                    x = m
                if abs(y - x) < 1e-10:
                    return m

        def f(x):
            return get_d(mu, x) - np.log(np.log(self.t + 1)/self.delta) / n

        lcb = bisection_search(lambda x: - f(x), 0.0, mu)
        ucb = bisection_search(f, mu, 1.0)
        return lcb, ucb

    def lucb(self, i):
        if self.draws[i] == 0:
            return 0, 1
        mu, n = self.rewards[i] / self.draws[i], self.draws[i]
        lcb, ucb = self.get_lucb(mu, n)
        return lcb, ucb

    def update(self, i, x):
        self.rewards[i] += x
        self.draws[i] += 1
        self.t += 1

    def select(self, arms=None, nP=0):
        if arms is None:
            arms = list(range(self.K))

        for i in arms:
            if self.draws[i] == 0:
                return i

        ucbs = {}
        lcbs = {}
        for i in arms:
            mu = self.rewards[i] / self.draws[i]
            lcb, ucb = self.lucb(i)
            ucbs[i] = ucb
            lcbs[i] = lcb
        ucb_sorted_index = [i for i, v in sorted(ucbs.items(), key=lambda x: x[1], reverse=True)]

        L = self.L - nP
        K = len(arms)
        if K == 1:
            return arms[0]
        if L >= K:
            # 例えば、K=20, L=8のとき、12個負腕が見つかったとき、
            # まだ判定はできないが、残り腕が8個で上位8腕を見つける問題を
            # 考えないといけないが、この問題設定ではtop-L arm identificationの
            # アルゴリズムは動かない
            #raise RuntimeError("L >= K")
            assert L == K
            L = K - 1

        topL = ucb_sorted_index[:L]

        # upper bound of simple regret
        Bks = []
        for i in arms:
            if i in topL:
                Bks.append((ucbs[ucb_sorted_index[L]] - lcbs[i], i))
            else:
                Bks.append((ucbs[ucb_sorted_index[L-1]] - lcbs[i], i))

        Bk_sorted_index = [i for v, i in sorted(Bks)]
        u_t = max([(ucbs[i], i) for i in Bk_sorted_index[L:]])[1]
        l_t = min([(lcbs[i], i) for i in Bk_sorted_index[:L]])[1]

        if self.draws[l_t] < self.draws[u_t]:
            return l_t
        else:
            return u_t


class LUCB:
    def __init__(self, K, theta, L, delta):
        self.K = K
        self.L = L
        self.delta = delta
        self.b = 1
        self.c = 0.5
        self.rewards = [0.0 for i in range(K)]
        self.draws = [0 for i in range(K)]
        self.t = 1
        self.pending = []

    def get_lucb(self, mu, n):
        # solve for increasing function f(x) = 0
        # this code assumes f(x) < 0, f(y) > 0
        def bisection_search(f, x, y):
            while True:
                m = (x + y) / 2
                if f(m) > 0:
                    y = m
                else:
                    x = m
                if abs(y - x) < 1e-10:
                    return m

        def f(x):
            return get_d(mu, x) - np.log(np.log(self.t + 1)/self.delta) / n

        lcb = bisection_search(lambda x: - f(x), 0.0, mu)
        ucb = bisection_search(f, mu, 1.0)
        return lcb, ucb

    def update(self, i, x):
        self.rewards[i] += x
        self.draws[i] += 1
        self.t += 1

    def select(self, arms=None, nP=0):
        if arms is None:
            arms = list(range(self.K))

        for i in arms:
            if self.draws[i] == 0:
                return i

        if self.pending:
            i = self.pending.pop()
            return i

        L = self.L - nP
        K = len(arms)
        if K == 1:
            return arms[0]
        if L >= K:
            L = K - 1

        stats = {}
        for i in arms:
            lcb, ucb = self.get_lucb(self.rewards[i] / self.draws[i], self.draws[i])
            stats[i] = (lcb, self.rewards[i]/self.draws[i], ucb)
        sorted_index_by_mean = sorted(stats, key=lambda i: stats[i][1], reverse=True)
        max_u = max([(stats[i][2], i) for i in sorted_index_by_mean[L:]])[1]
        min_l = min([(stats[i][0], i) for i in sorted_index_by_mean[:L]])[1]

        self.pending = [min_l]

        return max_u


class APT:
    def __init__(self, K, theta, L, delta):
        self.K = K
        self.L = L
        self.delta = delta
        self.theta = theta
        self.b = 1
        self.c = 0.5
        self.rewards = [0.0 for i in range(K)]
        self.draws = [0 for i in range(K)]
        self.t = 1
        self.pending = []

    def update(self, i, x):
        self.rewards[i] += x
        self.draws[i] += 1
        self.t += 1

    def select(self, arms=None, nP=0):
        if arms is None:
            arms = list(range(self.K))

        for i in arms:
            if self.draws[i] == 0:
                return i

        epsilon = 0
        Bk = {}
        for i in arms:
            mu = self.rewards[i] / self.draws[i]
            Bk[i] = self.draws[i]**0.5 * (abs(mu - self.theta) + epsilon)

        return max([(v, -i, i) for i, v in Bk.items()])[-1]


class UCB:
    def __init__(self, K, theta, L, delta):
        self.K = K
        self.L = L
        self.delta = delta
        self.b = 1
        self.c = 0.5
        self.rewards = [0.0 for i in range(K)]
        self.draws = [0 for i in range(K)]
        self.t = 1

    def update(self, i, x):
        self.rewards[i] += x
        self.draws[i] += 1
        self.t += 1

    def select(self, arms=None, nP=0):
        if arms is None:
            arms = list(range(self.K))

        for i in arms:
            if self.draws[i] == 0:
                return i

        Bk = {}
        for i in arms:
            mu = self.rewards[i] / self.draws[i]
            Bk[i] = mu + (0.5 * np.log(self.t) / self.draws[i])

        return max([(v, -i, i) for i, v in Bk.items()])[-1]

