#-*-coding=utf-8 -*-

import numpy as np

from function import nlfunc
from target import Target
from rand import ensure_rng


target = Target(nlfunc, {"x":(np.array([0.]),np.array([10.]))}, rs=1)
print target.random_state
print target.func
print target.keys
x = target.random_points(10)
print x

'''
x = target.order_points(10)
print x
print target.get_output(**x)
print target.dict_to_array(**x)
'''
