import numpy as np
import sys
import random
import math

#%% utility functoins

def get_d(xs, ys, eps=1e-16):
    """
    xs, ys: float, ary
    doubleは 仮数部52bit 5*3+1=16桁くらいなので eps=1e-16 とする
    """
    if type(xs) == np.ndarray:
        xs = np.maximum(np.minimum(xs, 1 - eps), eps)
        ys = np.maximum(np.minimum(ys, 1 - eps), eps)
        ret = xs * np.log(xs / ys) + (1 - xs) * np.log((1 - xs) / (1 - ys))
    else:
        xs = max(min(xs, 1 - eps), eps)
        ys = max(min(ys, 1 - eps), eps)
        ret = xs * np.log(xs / ys) + (1 - xs) * np.log((1 - xs) / (1 - ys))
    return ret


#%% arm with Bernoulli reward
class BernoulliArm:
    def __init__(self, mu):
        self.mu = mu
        self.samples = []

    def draw(self):
        r = 1.0 if random.random() < self.mu else 0.0
        self.samples.append(r)
        return r

    def mean(self):
        if self.samples:
            return np.sum(self.samples) / len(self.samples)
        return None

