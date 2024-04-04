#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bart
"""

import glob, gc, os, re
import numpy as np
import itertools as itert

def combos_prod(l):
    return list(itert.product(*l))
def combos(l, r):
    return [a for a in powerset(l) if len(a) == r]

def powerset(iterable):
    s = list(iterable)
    res =  [list(r2) for r2 in [list(itert.chain.from_iterable(itert.combinations(s, r) for r in range(len(s)+1)))]][0]
    res2 = []
    for r in res:
        res2 += [list(r)]
    return res2

def pairs(l):
    res = []
    for i in np.arange(1,len(l)-1, 1):
        res += [[l[i-1],l[i]]]
    return res

# flatten a list of lists
def flatten(l):
    res = []
    if type(l)==list:
        for l2 in l:
            if type(l2)==list:
                res += flatten(l2)
            else:
                res.append(l2)
    else:
        res.append(l)
    return res

def flatten1(l):
    return list(itert.chain.from_iterable(l))
def random_sign():
    return 1 if np.random.rand() < 0.5 else -1

def unique_all(l):
    _,idx = np.unique([str(x) for x in l], return_index=True)
    return [tuple(x) for x in np.array(l)[idx]]

def dedup(l):
    seen = set()
    smaller_l = []
    for x in l:
        if frozenset(x) not in seen:
            smaller_l.append(x)
            seen.add(frozenset(x))
    return smaller_l
