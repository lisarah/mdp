# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:51:37 2019

@author: craba
"""

import util as ut
import numpy as np
import matplotlib.pyplot as plt
import dynamicProg as dp

N = 3; 
M = 5;
S = N*M; A = 4;
gamma = 0.9;
P = ut.rectangleMDP(N,M,0.7);
C = np.random.rand(S,A);
Q = np.zeros((S,A));

T = 1000;
xt = np.random.randint(0,S-1);
exploration = 0.05;
VStar = np.zeros((S,T));
timeLine = np.arange(0,T);
for i in range(T):
    alpha =  2./(i+2);
    aNext = np.argmax(Q[xt,:]);
    
    if np.random.rand() <= exploration:
        aNext = np.random.randint(0,A);
    
    transition = P[:,xt*A +aNext];   
#    print (transition.shape);
#    print (np.arange(0,S).shape)
    xNext = np.random.choice(np.arange(0,S), p = transition);
    
    # Q update
    maxPrev = np.max(Q[xNext, :]);
    lastQ = 1.0*Q[xt,aNext];
    Q[xt, aNext] += alpha*(gamma*maxPrev + C[xt, aNext ] - lastQ);
    
    xt = xNext;
    
    VStar[:,i] = np.max(Q,axis = 1);
    
VTrue = dp.valueIteration(P,C,minimize = False, returnV = True,g = gamma);
VTrueMat = np.ones((S,T));
for i in range(S):
    VTrueMat[i,:] = VTrueMat[i,:]*VTrue[i];
#-----------qlearning Output --------------#
plt.figure();
plt.plot(timeLine, VStar.T);
plt.plot(timeLine, VTrueMat.T, alpha = 0.5, linewidth = 3)
plt.grid();
plt.show()

    