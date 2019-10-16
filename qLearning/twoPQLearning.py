# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:25:45 2019

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
        player 1: C = C1 + x2
        player 2: C = C2 + x1
"""
C1 = np.random.rand(S,A);
C2 = np.random.rand(S,A);
Q1 = 1.0*C1; Q2 = 1.0*C2;
stepInd1 = np.zeros((S,A));
stepInd2 = np.zeros((S,A));
xTrue = None; yTrue = None;
T = 3000;
xCur1 = np.random.randint(0,S); xCur2 = np.random.randint(0,S);
exploration = 0.1;
VStar1 = np.zeros((S,T)); VStar2 = np.zeros((S,T));
timeLine = np.arange(0,T);
for i in range(T):
#    alpha =  2./(i+2);
    aNext1 = np.argmax(Q1[xCur1,:]);
    aNext2 = np.argmax(Q2[xCur2,:]);
    if np.random.rand() < exploration:
        aNext1 = np.random.randint(0,A);
    if np.random.rand() < exploration:
        aNext2 = np.random.randint(0,A);
    transition1 = P[:,xCur1*A +aNext1];   
    transition2 = P[:,xCur2*A +aNext2];
    
    # two player Q learning code
    pi1 = np.zeros((S,S*A)); pi2 = np.zeros((S,S*A));
    A1 = np.argmax(Q1,axis = 1);
    A2 = np.argmax(Q2,axis = 1);
    for j in range(S):
        pi1[j, int(A*j + A1[j])] =1.;
        pi2[j, int(A*j + A2[j])] =1.;
        
    SA1 = ut.stationaryDist(P, pi1);
    SA2 = ut.stationaryDist(P, pi2);
    xTrue = np.reshape(SA1.T.dot(pi1), (S,A)); yTrue = np.reshape(SA2.T.dot(pi2), (S,A));
#    print (transition.shape);
#    print (np.arange(0,S).shape)
    xNext1 = np.random.choice(np.arange(0,S), p = transition1);
    xNext2 = np.random.choice(np.arange(0,S), p = transition2);
    
    # Q update
    maxNext1 = np.max(Q1[xNext1, :]);
    maxNext2 = np.max(Q2[xNext2,:])
    lastQ1 = 1.0*Q1[xCur1,aNext1];
    lastQ2 = 1.0*Q2[xCur2,aNext2];
    alphaInd1 = 1.0*stepInd1[xCur1, aNext1];
    alphaInd2 = 1.0*stepInd2[xCur2, aNext2];
    stepInd1[xCur1,aNext1] += 1;
    stepInd2[xCur2,aNext2] += 1;
    
    alpha1 = 2./(2.+alphaInd1); alpha2 = 2./(2. + alphaInd2);
    Q1[xCur1, aNext1] += alpha1*(gamma*maxNext1 + yTrue[xCur1, aNext1] + C1[xCur1, aNext1] - lastQ1);
    Q2[xCur2, aNext2] += alpha2*(gamma*maxNext2 + xTrue[xCur2, aNext2] + C2[xCur2, aNext2] - lastQ2);
    
    xCur1 = xNext1; 
    xCur2 = xNext2; 
    
    VStar1[:,i] = np.max(Q1,axis = 1);
    VStar2[:,i] = np.max(Q2,axis = 1);
    
#policy = dp.discounted_valueIteration(P,C,minimize = False, returnV = False,g = gamma);
VTrue1 = dp.discounted_valueIteration(P,C1 + yTrue,minimize = False, g = gamma);
VTrue2 = dp.discounted_valueIteration(P,C2 + xTrue,minimize = False, g = gamma);
VTrueMat1 = np.ones((S,T));
VTrueMat2 = np.ones((S,T));
for i in range(S):
    VTrueMat1[i,:] = VTrueMat1[i,:]*VTrue1[i];
    VTrueMat2[i,:] = VTrueMat2[i,:]*VTrue2[i];
    
    
#-----------qlearning Output --------------#
plt.figure();
plt.plot(timeLine, VStar1.T);
plt.plot(timeLine, VTrueMat1.T, alpha = 0.8, linewidth = 5);
plt.xscale('log');
plt.grid();
plt.show();


#-----------qlearning Output --------------#
plt.figure();
plt.plot(timeLine, VStar2.T);
plt.plot(timeLine, VTrueMat2.T, alpha = 0.8, linewidth = 5)
plt.xscale('log');
plt.grid();
plt.show();

# convex solver
    