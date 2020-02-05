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
T = 20;


Samples = 5;
timeLine = np.arange(0,T);

Vx_VI = np.zeros((S,T, Samples));
Vx_VI_min = np. zeros((S,T,Samples));
for sample in range(Samples):
    Vx_VI[:,0, sample] = np.random.rand(S);
    #--------------- If player y is also doing value iteration -------------#
    piX = np.random.rand(S,A); piY = np.random.rand(S,A);
    for s in range(S):
        piY[s,:] = piY[s,:]/ np.sum(piY[s,:]);
        piX[s,:] = piX[s,:]/ np.sum(piX[s,:]);
    Vy = np.zeros((S,T));
#    Vx_VI = np.zeros((S,T));
    for t in range(T-1):
            # calculate cost C # evaluate next policy x
            C = C1 + np.multiply(C2, piY);
            newPi = np.argmin(C + gamma*np.reshape(Vx_VI[:,t,sample].T.dot(P), (S,A)), axis = 1);
            
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
            Vx_VI[:,t+1, sample] = np.min(C + gamma*np.reshape(Vx_VI[:,t, sample].T.dot(P), (S,A)), axis = 1);
    ##--------------- If player y is also doing value iteration (minimizing)-------------#
    Vx_VI_min[:,0, sample] = np.random.rand(S);
    piX = np.random.rand(S,A); piY = np.random.rand(S,A);
    for s in range(S):
        piY[s,:] = piY[s,:]/ np.sum(piY[s,:]);
        piX[s,:] = piX[s,:]/ np.sum(piX[s,:]);
    Vy = np.zeros((S,T)); 
#    Vx_VI_min = np.zeros((S,T));
    for t in range(T-1):
            # calculate cost C # evaluate next policy x
            C = C1 + np.multiply(C2, piY);
            newPi = np.argmin(C + gamma*np.reshape(Vx_VI_min[:,t,sample].T.dot(P), (S,A)), axis = 1);
            
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
            Vx_VI_min[:,t+1,sample] = np.min(C + gamma*np.reshape(Vx_VI_min[:,t,sample].T.dot(P), (S,A)), axis = 1);
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
#    T
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

Vupper = np.zeros((S,T)); Vlower = np.zeros((S,T));
Vupper[:,0] = np.ones(S);
for t in range(T-1):
    Vupper[:,t+1] = np.min(C_upper+ gamma*np.reshape(Vupper[:,t].T.dot(P), (S,A)), axis = 1);
    Vlower[:,t+1] = np.min(C_lower+ gamma*np.reshape(Vlower[:,t].T.dot(P), (S,A)), axis = 1); 
    
hausDD = np.zeros((2, Samples, T)); # first is the maximizing agent, second is the minimizing agent
for sample in range(Samples):
    for t in range(T):
        upperNorm = np.linalg.norm(Vx_VI[:,t,sample] - Vupper[:,t], ord = np.inf);
        lowerNorm = np.linalg.norm(Vx_VI[:,t,sample] - Vupper[:,t], ord = np.inf);
        hausDD[0, sample, t] = np.max([upperNorm, lowerNorm]);
        upperNorm = np.linalg.norm(Vx_VI_min[:,t,sample] - Vupper[:,t], ord = np.inf);
        lowerNorm = np.linalg.norm(Vx_VI_min[:,t,sample] - Vupper[:,t], ord = np.inf);
        hausDD[1, sample, t] = np.max([upperNorm, lowerNorm]);
    
plt.figure();
plt.plot(timeLine, hausDD[0,:,:].T, linewidth = 3);
#plt.plot(timeLine, hausDD[1,:,:].T, ':', linewidth = 4);
plt.show();
##------------------- graph of results ---------------------------_#
textFont = 10;
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure();
plt.fill_between(timeLine, np.linalg.norm(Vupper, ord = np.inf, axis = 0),
                 np.linalg.norm(Vlower, ord = np.inf, axis = 0),
                 linewidth = 4,
                 color = 'k', alpha = 0.1)
#plt.plot(timeLine, np.linalg.norm(Vupper, ord = np.inf, axis = 0),  linewidth = 5, alpha = 0.3, color = 'k', label= "upper bound")
#plt.plot(timeLine, np.linalg.norm(Vlower, ord = np.inf, axis = 0),  linewidth = 5, alpha = 0.3, color = 'k', label= "lower bound")
plt.plot(timeLine, np.linalg.norm(Vx_VI, ord = np.inf, axis = 0), linewidth = 2.5, label= "maximizing Opponent");
plt.plot(timeLine, np.linalg.norm(Vx_VI_min, ord = np.inf, axis = 0),  ':', linewidth = 2.5, label= "minimizing Opponent");
plt.xlabel('Iterations (k)', fontsize=textFont);
plt.ylabel("$\|| V \||_{\infty}$", fontsize=textFont);
plt.tight_layout();
#plt.legend(); 
#plt.grid();
plt.show()

#def variance()
