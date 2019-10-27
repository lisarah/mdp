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
stateVec = np.linspace(0,S,S,endpoint = False);
P = ut.rectangleMD(M,N, 0.7)

C = np.random.rand(S,A);
# generate random list
#SARSA implementation

T = 3000;
Q = np.zeros((S, A, T));
s = np.random.randint(0,S);
for t in range(T):
    aNext = np.argmax(Q[s, :, t]);
    transition = P[:,s*A + aNext];
    nextS  = np.random.choice(stateVec, 1, p = transition);
    nextR = 1*C(aNext, s);