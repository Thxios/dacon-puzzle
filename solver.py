
import os
import numpy as np
from patch_util import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

INF = 1e+6
V, H = 0, 1


class Solver:
    fill_order_base = [
        0, 1, 4, 5,
        2, 3, 6, 7,
        8, 9, 12, 13,
        10, 11, 14, 15
    ]

    def __init__(self, logit: np.ndarray, temperature=2., eps=1e-8, max_branches=4):
        assert logit.shape == (n_patches, n_patches, 2)

        prob = sigmoid(logit / temperature)
        log_prob = np.log(prob + eps)
        self.cost = -log_prob
        self.cost[np.eye(n_patches, dtype=bool)] = INF
        self.min = INF
        self.tile = [-1] * n_patches
        self.min_tile = None
        self.remain = [*range(n_patches)]
        self.fill_order = [self.fill_order_base.index(i) for i in range(16)]
        self.cnt = 0
        self.max_branches = max_branches
    def calc_score(self, pos, indices):
        r, c = idx2coord(pos)
        assert r != 0 or c != 0
        ret = np.zeros((len(indices,)))
        if r > 0:
            up = self.tile[coord2idx(r - 1, c)]
            ret += self.cost[up, indices, V]
        if c > 0:
            left = self.tile[coord2idx(r, c - 1)]
            ret += self.cost[left, indices, H]
        return ret

    def solve(self):
        not_tl = np.sum(np.min(self.cost, axis=0), axis=-1)
        cands = [*zip(not_tl, range(n_patches))]
        cands.sort(reverse=True)
        for ntl, cand in cands:
            self.tile[self.fill_order[0]] = cand
            self.remain.remove(cand)
            self.find(1, 0)
            self.remain.append(cand)

        if self.min_tile is not None:
            ret = self.min_tile[:]
        else:
            ret = [*range(n_patches)]

        return ret, self.min


    def find(self, idx: int, score: float):
        if idx >= n_patches:
            self.cnt += 1
            if self.cnt % 10000 == 0 and self.max_branches > 2:
                self.max_branches -= 1
            if self.min > score:
                self.min = score
                self.min_tile = self.tile[:]
            return
        if score > self.min:
            return

        pos = self.fill_order[idx]
        cands = [*zip(self.calc_score(pos, self.remain), self.remain)]
        cands.sort()
        for cnt, (cost, tidx) in enumerate(cands):
            if cnt > self.max_branches:
                break
            self.tile[pos] = tidx
            self.remain.remove(tidx)
            self.find(idx + 1, score + cost)
            self.remain.append(tidx)






