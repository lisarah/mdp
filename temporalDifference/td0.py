#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 17:23:08 2019

@author: sarahli
"""

import util as ut
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
N = 3; 
M = 3;
S = N*M; A = 4;
gamma = 0.5;
P = ut.rectangleMD(M,N, 0.7)

C = np.random.rand(S,A);

#SARSA implementation

T = 3000;
Q = np.zeros((S, A, T))
for t in range(T):
    