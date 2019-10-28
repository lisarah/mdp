#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 17:23:08 2019


This is a SARSA - TD implementation

@author: sarahli
"""

import util as ut
import matplotlib.pyplot as plt
import numpy as np
import dynamicProg as dP

plt.close('all')
N = 2; 
M = 2;
S = N*M; A = 4;
gamma = 0.4;
alpha = 1.0;#step size of algorithm
eps = 0.2; # for the epsilon greedy algorithm
stateVec = np.linspace(0,S,S,endpoint = False);
P = ut.rectangleMDP(M,N, 0.7)

C = np.random.rand(S,A);
#print (C);
# generate random list
#SARSA implementation

T = 100000;
Q = np.zeros((S, A, T));
s = np.random.randint(0,S);
curA = np.random.randint(0,A)
for t in range(T-1):
    alpha = 1./(t+1);
    # transition
    transition = P[:,s*A + curA];
    nextS  = int(np.random.choice(stateVec, 1, p = transition)[0]);
    nextR = 1.*C[s, curA];
    
    # next action
    newPol = dP.epsGreedy(Q[:,:,t], nextS, eps);
#    print (new1sPol)
    nextA = np.random.choice(np.arange(0,A), p = newPol)
    
    
    # value function update
    lastQ = 1.0*Q[:,:,t];
    Q[:,:,t+1] = lastQ;
    Q[s, curA, t+1] += alpha*(nextR + gamma*lastQ[nextS,nextA] - lastQ[s,curA]);
    
    # action update
    curA = nextA;
    s = nextS;
    
    
# value iteration: 
VTrue = dP.valueIteration(P,C, returnV = True, g = gamma);

plt.figure();
plt.plot(np.linspace(0,T,T), np.linalg.norm(np.min(Q, axis = 1), ord = 2, axis = 0))
plt.plot(np.linspace(0,T,T), (np.ones((1,T))*np.linalg.norm(VTrue,ord=2)).T);
plt.grid();
plt.show();



