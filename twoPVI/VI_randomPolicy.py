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
N = 3; 
M = 3;
S = N*M; A = 4;
gamma = 0.5;
P = ut.rectangleMDP(N,M,0.7);
"""
    Cost model:
        player x: C = C1 + C2.dot(y)
        player y: C = C1 + C2.dot(x)
"""
C1 = 0.1*np.random.rand(S,A);
C2 = 0.02*np.diag(np.random.rand(S*A));
T = 100;
Samples = 100;
timeLine = np.arange(0,T);
Vx = np.zeros((S,T,Samples)); 
for sample in range(Samples):
    #-------------------- propagating a random policy -------------------#
    piX = np.zeros(S); piY = np.zeros(S);
    xDensity = np.ones(S)/S; yDensity = np.ones(S)/S;
    
    for t in range(T-1):
        # calculate resulting state-action density    
        x = np.zeros(S*A); y = np.zeros(S*A);
        for i in range(S):
            x[i*A + int(piX[i])] = xDensity[i];
            y[i*A + int(piY[i])] = yDensity[i];
        # calculate value function
        newCost = C1 + np.reshape(C2.dot(y), (S,A));
    
        Vx[:,t+1, sample] = np.min(newCost + gamma*np.reshape(Vx[:,t, sample].T.dot(P), (S,A)), axis = 1);
    
        piX = np.argmin(newCost + gamma*np.reshape(Vx[:,t, sample].T.dot(P), (S,A)), axis = 1);
        piY = np.random.randint(0, A, S);
    
        # calculate new density after propagation
        xDensity = P.dot(x); yDensity = P.dot(y);
#--------------- If player y is also doing value iteration -------------#
piX = np.zeros(S); piY = np.zeros(S);
xDensity = np.ones(S)/S; yDensity = np.ones(S)/S;
Vy = np.zeros((S,T));Vx_VI = np.zeros((S,T));
for t in range(T-1):
    # calculate resulting state-action density    
    x = np.zeros(S*A); y = np.zeros(S*A);
    for i in range(S):
        x[i*A + int(piX[i])] = xDensity[i];
        y[i*A + int(piY[i])] = yDensity[i];
    # calculate value function
    newCostx = C1 + np.reshape(C2.dot(y), (S,A));
    newCosty = C1 + np.reshape(C2.dot(x), (S,A));
    Vx_VI[:,t+1] = np.min(newCostx + gamma*np.reshape(Vx_VI[:,t].T.dot(P), (S,A)), axis = 1);
    Vy[:,t+1] = np.min(newCosty + gamma*np.reshape(Vy[:,t].T.dot(P), (S,A)), axis = 1);
    piX = np.argmin(newCostx + gamma*np.reshape(Vx_VI[:,t].T.dot(P), (S,A)), axis = 1);
    piY = np.argmin(newCosty + gamma*np.reshape(Vy[:,t].T.dot(P), (S,A)), axis = 1);

    # calculate new density after propagation
    xDensity = P.dot(x); yDensity = P.dot(y);
#--------------- If player y is doing policy iteration -------------#
piX = np.zeros(S); piY = np.zeros(S);
xDensity = np.ones(S)/S; yDensity = np.ones(S)/S;
Vy = np.zeros((S,T));Vx_PI = np.zeros((S,T));
for t in range(T-1):
    # calculate resulting state-action density    
    x = np.zeros(S*A); y = np.zeros(S*A);
    for i in range(S):
        x[i*A + int(piX[i])] = xDensity[i];
        y[i*A + int(piY[i])] = yDensity[i];
    # calculate value function
    newCostx = C1 + np.reshape(C2.dot(y), (S,A));
    newCosty = C1 + np.reshape(C2.dot(x), (S,A));
    Vx_PI[:,t+1] = np.min(newCostx + gamma*np.reshape(Vx_PI[:,t].T.dot(P), (S,A)), axis = 1);
    piMaty = np.zeros((S,S*A));
    newCosty_vec = np.zeros(S)
    for state in range(S):
        piMaty[state, int(piY[state])] = 1.;
        newCosty_vec[state] = newCosty[state, int(piY[state])];
    Vy[:,t+1] = np.linalg.inv((np.eye(S) - gamma*P.dot(piMaty.T))).dot(newCosty_vec);
    piX = np.argmin(newCostx + gamma*np.reshape(Vx_PI[:,t+1].T.dot(P), (S,A)), axis = 1);
    piY = np.argmin(newCosty + gamma*np.reshape(Vy[:,t+1].T.dot(P), (S,A)), axis = 1);

    # calculate new density after propagation
    xDensity = P.dot(x); yDensity = P.dot(y);
#------------------ propagating upper and lower bounds -----------------#
C_lower = 1.0*C1; 
C_upper = C1+np.reshape(C2.dot(np.ones(S*A)), (S,A));
T = 100;
Vupper = np.zeros((S,T)); Vlower = np.zeros((S,T));
for t in range(T-1):
    Vupper[:,t+1] = np.min(C_upper+ gamma*np.reshape(Vupper[:,t].T.dot(P), (S,A)), axis = 1);
    Vlower[:,t+1] = np.min(C_lower+ gamma*np.reshape(Vlower[:,t].T.dot(P), (S,A)), axis = 1); 
#------------------- graph of results ---------------------------_#
plt.figure();
for sample in range(Samples):
    plt.plot(timeLine, np.linalg.norm(Vx[:,:,sample], ord = 2, axis = 0));
plt.plot(timeLine, np.linalg.norm(Vupper, ord = 2, axis = 0), label= "upper bound")
plt.plot(timeLine, np.linalg.norm(Vlower, ord = 2, axis = 0), label= "lower bound");
plt.plot(timeLine, np.linalg.norm(Vx_VI, ord = 2, axis = 0), label= "value iteration");
plt.plot(timeLine, np.linalg.norm(Vx_PI, ord = 2, axis = 0), label= "policy iteration");
plt.legend(); plt.grid();
plt.show()

#def variance()
