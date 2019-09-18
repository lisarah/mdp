# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:02:34 2019

@author: craba
"""
import numpy as np
import dynamicProg as pI
import cvxpy as cvx
import util as ut
# need to generate MDP 
# P : S x SA, column stochastic
N = 3; # columns
M = 5; # rows
S = N*M;
A = 4;
p = 0.7;
SMin = []; SMax = [];
for i in range(S):
    if i%3 == 0:
        SMin.append(i);
    else:
        SMax.append(i);
        
P = ut.rectangleMDP(M,N,p);
C = np.random.rand(N*M,A)*100.;

#-------------------------- value iteration -----------------------------------#
avgCost = pI.game_VI(P,C, SMin, SMax);

#------------ checking with cvx -----------------------------------#
y = cvx.Variable((S,A));
constraints = [];
ones = np.ones(A);

for i in range(S):
    constraints.append(ones*(y[i,:]) == 
                       sum([sum([P[i,s*A +a]*y[s,a] for s in range(S)])
                       for a in range(A)]));
constraints.append(y >= 0);
constraints.append( sum([sum([y[s,a] for s in range(S)]) for a in range(A)]) == 1.);
gamma= 1.0;
cost = 0;
for s in range(S):
    if s in SMin:
        cost += sum([y[s,a]*C[s,a] for a in range(A)]);
    else:
        cost += -sum([y[s,a]*C[s,a] for a in range(A)]);
        
objective = cvx.Minimize(cost)
mdp = cvx.Problem(objective, constraints);
mdpRes = mdp.solve();
print ("expected cost from LP formulation: ",mdpRes)
print ("expected cost from Value Iteration: ",avgCost)