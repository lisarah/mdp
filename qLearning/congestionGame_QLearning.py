# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:30:25 2019

@author: craba
"""

import util as ut
import numpy as np
import matplotlib.pyplot as plt
import dynamicProg as dp
plt.close('all')
N = 3; 
M = 3;
S = N*M; A = 4;
Players = 3;
gamma = 0.5;
P = ut.rectangleMDP(N,M,0.7);
"""
    Cost model:
        player 1: C = C1 + x2
        player 2: C = C2 + x1
"""
C = np.zeros((S,A,Players));
C1 = np.random.rand(S,A);
for i in range(Players):
    C[:,:,i] = 1.0*C1;

"""
    Each player has Q function of size S x SA
"""
Q = np.zeros((S,A, Players));
for i in range(Players):
    Q[:,:,i] = 1.0*C1;

"""
    Each player needs to keep track of its own step indices
"""
stepInd = np.zeros((S,A,Players));


"""
    saDistr keeps track of each player's current state-action distribution, 
    always discrete
"""
xsaDistrTrue = np.zeros((S,A, Players));
"""
    total number of iterations and Q learning exploration factor
"""
T = 5000;
timeLine = np.arange(0,T);
exploration = 0.1;
"""
    current state of each player
"""
xCur = np.zeros(Players);
for i in range(Players):
    xCur[i] = int(np.random.randint(0,S));  
# I don't actually need this
VHist = np.zeros((S,T,Players));

for i in range(T):
#    alpha =  2./(i+2);
    aNext = np.zeros((Players));
    xNext = np.zeros((Players));
    transition = np.zeros((S, Players));
    pi = np.zeros((S, S*A, Players));
    maxNext = np.zeros((Players));
    lastQ = np.zeros((Players));
    alphaInd  = np.zeros((Players));
    for player in range(Players):
#        player = int(player);
        xCur[player]  = int(xCur[player]);
        if np.random.rand() > exploration:
            aNext[player] = np.argmax(Q[int(xCur[player]), :, player]);
        else:
            aNext[player] = np.random.randint(0,A);
        transition[:,player] = 1*P[:,int(xCur[player]*A + aNext[player])];    
        for j in range(S):
            Aj  = np.argmax(Q[:,:, player], axis = 1);
            pi[:,int(A*j + Aj[j]),player] = 1.;
            

        SA = ut.stationaryDist(P, pi[:,:,player]);
        xsaDistrTrue[:,:,player] = np.reshape(SA.T.dot(pi[:,:,player]), (S,A));      
        xNext[player] = np.random.choice(np.arange(0,S), p = transition[:,player]);
        maxNext[player] = np.max(Q[int(xNext[player]), :, player]);
        lastQ[player] = 1.0*Q[int(xCur[player]), int(aNext[player]), player];
        alphaInd[player] = 1.0*stepInd[int(xCur[player]), int(aNext[player]), player];
        stepInd[int(xCur[player]), int(aNext[player]), player] += 1;
    
    # Q update
    for player in range(Players):
        for neighbours in range(Players):
            coupleCost = 1.0*C[int(xCur[player]), int(aNext[player]), player]; 
            if player != neighbours:
                coupleCost += xsaDistrTrue[int(xCur[player]), int(aNext[player]), player];
                
        alpha = 2./(2. + alphaInd[player]);
        Q[int(xCur[player]), int(aNext[player]), player] += alpha*(gamma*maxNext[player]
                + coupleCost - lastQ[player]);

    for player in range(Players):
        xCur[player] = 1.0*xNext[player];
        VHist[:, i, player] = np.max(Q[:,:, player], axis = 1);
    
    
#policy = dp.discounted_valueIteration(P,C,minimize = False, returnV = False,g = gamma);
#VTrue1 = dp.discounted_valueIteration(P,C1 + yTrue,minimize = False, g = gamma);
#VTrue2 = dp.discounted_valueIteration(P,C2 + xTrue,minimize = False, g = gamma);
#VTrueMat1 = np.ones((S,T));
#VTrueMat2 = np.ones((S,T));
#for i in range(S):
#    VTrueMat1[i,:] = VTrueMat1[i,:]*VTrue1[i];
#    VTrueMat2[i,:] = VTrueMat2[i,:]*VTrue2[i];
    
    
#-----------qlearning Output --------------#
for i in range(Players):
    plt.figure();
    plt.plot(timeLine, VHist[:,:,i].T);
#    plt.plot(timeLine, VTrueMat1.T, alpha = 0.8, linewidth = 5);
#    plt.xscale('log');
    plt.grid();
    plt.show();


#-----------qlearning Output --------------#
#plt.figure();
#plt.plot(timeLine, VStar2.T);
#plt.plot(timeLine, VTrueMat2.T, alpha = 0.8, linewidth = 5)
#plt.xscale('log');
#plt.grid();
#plt.show();

# convex solver
    