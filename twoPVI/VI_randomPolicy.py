# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:30:46 2019

@author: craba
"""
import util as ut
import numpy as np
import matplotlib.pyplot as plt
import dynamicProg as dp
import cvxpy as cvx
plt.close('all')
N = 5; 
M = 4;
S = N*M; A = 4;
gamma = 0.5;
P = ut.rectangleMDP(N,M,0.7);
"""
    Cost model:
        player 1: C = C1 + x2
        player 2: C = C2 + x1
"""
C1 = np.random.rand(S,A);
C2 = 0.1*np.random.rand(S*A,S*A);
#C_lower = 1.0*C1; C_upper = C1+C2;
T = 100;
piX = np.zeros(S); piY = np.zeros(S);
xDensity = np.ones(S)/S; yDensity = np.ones(S)/S;


Vx = np.zeros((S,T)); Vy = np.zeros((S,T));
timeLine = np.arange(0,T);
for t in range(T-1):
    # calculate resulting state-action density    
    x = np.zeros(S*A); y = np.zeros(S*A);
    for i in range(S):
        x[i*A + int(piX[i])] = xDensity[i];
        y[i*A + int(piY[i])] = yDensity[i];
    # calculate value function
    newCost = C1 + np.reshape(C2.dot(y), (S,A));

    Vx[:,t+1] = np.min(newCost + gamma*np.reshape(Vx[:,t].T.dot(P), (S,A)), axis = 1);

    piX = np.argmin(newCost + gamma*np.reshape(Vx[:,t].T.dot(P), (S,A)), axis = 1);
    piY = np.random.randint(0, A, S);

    # calculate new density after propagation
    xDensity = P.dot(x); yDensity = P.dot(y);

plt.figure();
plt.plot(timeLine, np.linalg.norm(Vx, ord = 2, axis = 0));
plt.show()