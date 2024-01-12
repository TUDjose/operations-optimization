# Variables to be changed:
# 1. w values
# 2. C values
# 3. QC values
# 4. P values

""""
Sensitivity of w very dependent on the distribution of P -> overall w is quite insensitive
"""

import numpy as np
import random
import pickle
from tqdm import tqdm
from collections import Counter
from Model.data_generator import DataGenerator
from Model.model import RSPModel


def count_i(lst):
    k = []
    for tup in lst:
        k.append(tup[0])
    counts = Counter(k)
    return list(counts.values())

def w_sensitivity():
    res = []
    l = []
    c = 0

    with open('Model/data/5-8-4-1.pkl', 'rb') as f:
        loaded_array = pickle.load(f)
    dg = DataGenerator.recover(loaded_array)
    RSP = RSPModel(dg)
    RSP.solve()
    res.append([list(RSP.w), [RSP.L[i].x for i in range(RSP.M)], count_i([(i, k) for (i, k) in RSP.x.keys() if RSP.x[i, k].x == 1.])])

    n = 500
    with (tqdm(total=n) as pbar):
        for _ in range(n):
            with open('Model/data/5-8-4-1.pkl', 'rb') as f:
                loaded_array = pickle.load(f)
            dg = DataGenerator.recover(loaded_array)
            RSP = RSPModel(dg)
            random_numbers = [random.randint(-10, 11) / 10 for _ in range(len(RSP.w))]
            for i in range(len(RSP.w)):
                RSP.w[i] += random_numbers[i]
            RSP.solve()
            l.append([RSP.L[i].x for i in range(RSP.M)])
            res.append([list(RSP.w), [RSP.L[i].x for i in range(RSP.M)], count_i([(i, k) for (i, k) in RSP.x.keys() if RSP.x[i, k].x == 1.])])
            if count_i([(i, k) for (i, k) in RSP.x.keys() if RSP.x[i, k].x == 1.]) == res[0][2]:
                c += 1
            pbar.update(1)

    return c/n, res, list(set(tuple(sublist) for sublist in l))


def p_sensitivity():
    res = []
    c = 0
    C = []

    with open('Model/data/5-8-4-1.pkl', 'rb') as f:
        loaded_array = pickle.load(f)
    dg = DataGenerator.recover(loaded_array)
    RSP = RSPModel(dg)
    RSP.solve()
    res.append(count_i([(i, k) for (i, k) in RSP.x.keys() if RSP.x[i, k].x == 1.]))

    n = 500
    ps = [5, 10, 20, 30, 40, 50, 75, 100]
    with tqdm(total=n*len(ps)) as pbar:
        for p in ps:
            c=0
            for _ in range(n):
                with open('Model/data/5-8-4-1.pkl', 'rb') as f:
                    loaded_array = pickle.load(f)
                dg = DataGenerator.recover(loaded_array)
                for i in range(dg.M):
                    for j in range(dg.N):
                        dg.P[i, j] += random.randint(-p, p+1)
                        if dg.P[i, j] < 0:
                            dg.P[i, j] = 0
                RSP = RSPModel(dg)
                RSP.solve()
                res.append(count_i([(i, k) for (i, k) in RSP.x.keys() if RSP.x[i, k].x == 1.]))
                if count_i([(i, k) for (i, k) in RSP.x.keys() if RSP.x[i, k].x == 1.]) == res[0]:
                    c += 1
                pbar.update(1)
            C.append(c/n)

    return C


if __name__ == '__main__':
    print(p_sensitivity())
