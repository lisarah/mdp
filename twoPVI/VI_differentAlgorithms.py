#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:36:33 2019

@author: sarahli
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
C1 = np.random.rand(S,A);
C2 = 0.3*np.random.rand(S,A);
T = 100;


Samples = 10;
timeLine = np.arange(0,T);

#--------------- If player y is also doing value iteration -------------#
piX = np.random.rand(S,A); piY = np.random.rand(S,A);
for s in range(S):
    piY[s,:] = piY[s,:]/ np.sum(piY[s,:]);
    piX[s,:] = piX[s,:]/ np.sum(piX[s,:]);
Vy = np.zeros((S,T));Vx_VI = np.zeros((S,T));
for t in range(T-1):
        # calculate cost C # evaluate next policy x
        C = C1 + np.multiply(C2, piY);
        newPi = np.argmin(C + gamma*np.reshape(Vx_VI[:,t].T.dot(P), (S,A)), axis = 1);
        
        piX = np.zeros((S,A));
        for state in range(S):
            piX[state, int(newPi[state])] = 1.;
        # evaluate next policy y
        Cnext = C1 + np.einsum('ij, ij->ij', C2, piX);
        newPi = np.argmax(C + gamma*np.reshape(Vy[:,t].T.dot(P), (S,A)), axis = 1);
        Vy[:,t+1]= np.max(C + gamma*np.reshape(Vy[:,t].T.dot(P), (S,A)), axis = 1);
        piY = np.zeros((S,A));
        for state in range(S):
            piY[state, int(newPi[state])] = 1.;   
        
        # calculate value function  
        Vx_VI[:,t+1] = np.min(C + gamma*np.reshape(Vx_VI[:,t].T.dot(P), (S,A)), axis = 1);
##--------------- If player y is also doing value iteration (minimizing)-------------#
piX = np.random.rand(S,A); piY = np.random.rand(S,A);
for s in range(S):
    piY[s,:] = piY[s,:]/ np.sum(piY[s,:]);
    piX[s,:] = piX[s,:]/ np.sum(piX[s,:]);
Vy = np.zeros((S,T)); Vx_VI_min = np.zeros((S,T));
for t in range(T-1):
        # calculate cost C # evaluate next policy x
        C = C1 + np.multiply(C2, piY);
        newPi = np.argmin(C + gamma*np.reshape(Vx_VI_min[:,t].T.dot(P), (S,A)), axis = 1);
        
        piX = np.zeros((S,A));
        for state in range(S):
            piX[state, int(newPi[state])] = 1.;
        # evaluate next policy y
        Cnext = C1 + np.einsum('ij, ij->ij', C2, piX);
        newPi = np.argmin(C + gamma*np.reshape(Vy[:,t].T.dot(P), (S,A)), axis = 1);
        Vy[:,t+1]= np.min(C + gamma*np.reshape(Vy[:,t].T.dot(P), (S,A)), axis = 1);
        piY = np.zeros((S,A));
        for state in range(S):
            piY[state, int(newPi[state])] = 1.;   
        
        # calculate value function  
        Vx_VI_min[:,t+1] = np.min(C + gamma*np.reshape(Vx_VI_min[:,t].T.dot(P), (S,A)), axis = 1);
###--------------- If player y is doing policy iteration -------------#
#piX = np.zeros(S); piY = np.zeros(S);
#for s in range(S):
#    piY[s,:] = piY[s,:]/ np.sum(piY[s,:]);
#    piX[s,:] = piX[s,:]/ np.sum(piX[s,:]);
#Vy = np.zeros((S,T));Vx_PI = np.zeros((S,T));
#for t in range(T-1):
#    # calculate cost C # evaluate next policy x
#    C = C1 + np.multiply(C2, piY);
#    newPi = np.argmin(C + gamma*np.reshape(Vx[:,t, sample].T.dot(P), (S,A)), axis = 1);
#    
#    piX = np.zeros((S,A));
#    for state in range(S):
#        piX[state, int(newPi[state])] = 1.;
#    # evaluate next policy y
#    Cnext = C1 + np.einsum('ij, ij->ij', C2, piX);
#    newPi = np.argmin(C + gamma*np.reshape(Vy[:,t].T.dot(P), (S,A)), axis = 1);
#    Vy[:,t+1]= np.min(C + gamma*np.reshape(Vy[:,t].T.dot(P), (S,A)), axis = 1);
#    piY = np.zeros((S,A));
#    for state in range(S):
#        piY[state, int(newPi[state])] = 1.; 
#        
#    Vx_PI[:,t+1] = np.min(newCostx + gamma*np.reshape(Vx_PI[:,t].T.dot(P), (S,A)), axis = 1);
#    # calculate value function  
#    Vx_VI_min[:,t+1] = np.min(C + gamma*np.reshape(Vx[:,t, sample].T.dot(P), (S,A)), axis = 1);
#    
#    newCostx = C1 + np.reshape(C2.dot(y), (S,A));
#    newCosty = C1 + np.reshape(C2.dot(x), (S,A));
#    Vx_PI[:,t+1] = np.min(newCostx + gamma*np.reshape(Vx_PI[:,t].T.dot(P), (S,A)), axis = 1);
#    piMaty = np.zeros((S,S*A));
#    newCosty_vec = np.zeros(S)
#    for state in range(S):
#        piMaty[state, int(piY[state])] = 1.;
#        newCosty_vec[state] = newCosty[state, int(piY[state])];
#    Vy[:,t+1] = np.linalg.inv((np.eye(S) - gamma*P.dot(piMaty.T))).dot(newCosty_vec);
#    piX = np.argmin(newCostx + gamma*np.reshape(Vx_PI[:,t+1].T.dot(P), (S,A)), axis = 1);
#    piY = np.argmin(newCosty + gamma*np.reshape(Vy[:,t+1].T.dot(P), (S,A)), axis = 1);
#
#    # calculate new density after propagation
#    xDensity = P.dot(x); yDensity = P.dot(y);
##------------------ propagating upper and lower bounds -----------------#
C_lower = 1.0*C1; 
C_upper = C1+np.multiply(C2, np.ones((S,A)));
T = 100;
Vupper = np.zeros((S,T)); Vlower = np.zeros((S,T));
for t in range(T-1):
    Vupper[:,t+1] = np.min(C_upper+ gamma*np.reshape(Vupper[:,t].T.dot(P), (S,A)), axis = 1);
    Vlower[:,t+1] = np.min(C_lower+ gamma*np.reshape(Vlower[:,t].T.dot(P), (S,A)), axis = 1); 
#------------------- graph of results ---------------------------_#
plt.figure();

plt.plot(timeLine, np.linalg.norm(Vupper, ord = 2, axis = 0),  linewidth = 5, alpha = 0.3, color = 'k', label= "upper bound")
plt.plot(timeLine, np.linalg.norm(Vlower, ord = 2, axis = 0),  linewidth = 5, alpha = 0.3, color = 'k', label= "lower bound")
plt.plot(timeLine, np.linalg.norm(Vx_VI, ord = 2, axis = 0), linewidth = 2, label= "maximizing Opponent");
plt.plot(timeLine, np.linalg.norm(Vx_VI_min, ord = 2, axis = 0), linewidth = 2, label= "minimizing Opponent");
plt.legend(); plt.grid();
plt.show()

#def variance()
