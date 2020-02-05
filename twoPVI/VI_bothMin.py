#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:24:31 2019

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
gamma = 0.7;
P = ut.rectangleMDP(N,M,0.7);
"""
    Cost model:
        player x: C = C1 + C2.dot(y)
        player y: C = C1 + C2.dot(x)
"""
C1 = np.random.rand(S,A);
C2 = 0.3*np.random.rand(S,A);
T =100;

Samples = 10;
timeLine = np.arange(0,T);

Vx = np.zeros((S,T,Samples)); 
Vy_varyingGamma = np.zeros((S,T,Samples)); 
for sample in range(Samples):
    gammaY = 1./Samples*(sample+1);
    piX = np.random.rand(S,A); piY = np.random.rand(S,A);
    for s in range(S):
        piY[s,:] = piY[s,:]/ np.sum(piY[s,:]);
        piX[s,:] = piX[s,:]/ np.sum(piX[s,:]);
    for t in range(T-1):
        # calculate cost C
        C = C1 + np.multiply(C2, piY);
        # evaluate next policy x
        newPi = np.argmin(C + gamma*np.reshape(Vx[:,t, sample].T.dot(P), (S,A)), axis = 1);
        
        piX = np.zeros((S,A));
        for state in range(S):
            piX[state, int(newPi[state])] = 1.;
        # evaluate next policy y
        Cnext = C1 + np.einsum('ij, ij->ij', C2, piX);
        newPi = np.argmin(C + gammaY*np.reshape(Vy_varyingGamma[:,t,sample].T.dot(P), (S,A)), axis = 1);
        Vy_varyingGamma[:,t+1,sample]= np.min(C + gammaY*np.reshape(Vy_varyingGamma[:,t,sample].T.dot(P), (S,A)), axis = 1);
        piY = np.zeros((S,A));
        for state in range(S):
            piY[state, int(newPi[state])] = 1.;         
        # calculate value function  
        Vx[:,t+1, sample] = np.min(C + gamma*np.reshape(Vx[:,t, sample].T.dot(P), (S,A)), axis = 1);

##--------------- If player y is also doing value iteration -------------#
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
Vupper = np.zeros((S,T)); Vlower = np.zeros((S,T));
for t in range(T-1):
    Vupper[:,t+1] = np.min(C_upper+ gamma*np.reshape(Vupper[:,t].T.dot(P), (S,A)), axis = 1);
    Vlower[:,t+1] = np.min(C_lower+ gamma*np.reshape(Vlower[:,t].T.dot(P), (S,A)), axis = 1); 
#------------------- graph of results ---------------------------_#
plt.figure();
for sample in range(Samples):
    plt.plot(timeLine, np.linalg.norm(Vx[:,:,sample], ord = 2, axis = 0), linewidth = 3, label = "%0.2f" %(1./Samples*(sample+1)));
plt.plot(timeLine, np.linalg.norm(Vupper, ord = 2, axis = 0), linewidth = 5, alpha = 0.3, color = 'b', label= "upper bound")
plt.plot(timeLine, np.linalg.norm(Vlower, ord = 2, axis = 0), linewidth = 5, alpha = 0.3, color = 'b',  label= "lower bound");
#plt.plot(timeLine, np.linalg.norm(Vx_VI, ord = 2, axis = 0), label= "value iteration");
#plt.plot(timeLine, np.linalg.norm(Vx_VI_min, ord = 2, axis = 0), label= "value iteration_ minimize");
plt.xlabel('Iterations')
plt.ylabel("$\||V^k\||_2$")
plt.legend(); plt.grid();
plt.show()


# scatter plots
plotTime = [1, int((T/2)-1), T-1];
textFont = 10;
fig, axes = plt.subplots(3,1, sharex = True); axInd = 0;
maxY = 1.05*np.max(Vupper);
minY = 0.99*np.min(Vlower);
for t in plotTime:
    stateArr = np.linspace(0,S,S, endpoint = False)+1;
#    ax = plt.subplot(111);
    axes[axInd].set_title("k = %d"%t, fontsize=textFont);
    axes[axInd].errorbar(stateArr, Vx[:, t, 0], lolims = True, 
                 yerr = Vupper[:,t] - Vx[:, t, 0], 
                 fmt='none', 
                 ecolor='lightblue', elinewidth=5, capsize=0, capthick = 0, dash_capstyle = 'round');
    axes[axInd].errorbar(stateArr, Vx[:, t, 0], uplims = True, 
                 yerr =Vx[:, t, 0] - Vlower[:,t], 
                 fmt='none',
                 ecolor='lightblue', elinewidth=5, capsize=0, capthick = 0, dash_capstyle = 'round');
    for sample in range(0,Samples):
        axes[axInd].errorbar(stateArr, Vx[:, t, sample], fmt = 'o:', label = "$\gamma = $%0.2f"%(1./Samples*(sample+1)));
    axes[axInd].set_ylim([minY, maxY]);
    axInd += 1;
axes[2].set_xlabel('States', fontsize=textFont);
axes[1].set_ylabel('$V(s)$', fontsize=textFont);
fig.tight_layout()
axes[1].legend(loc='center left', borderaxespad=0.1, bbox_to_anchor=(1, 0.5));
plt.subplots_adjust(right=0.8)
plt.show();

