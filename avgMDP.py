# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:47:40 2019

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

P = ut.rectangleMDP(M,N,p);
C = np.random.rand(N*M,A)*100.;
#S = 6; A = 2;
#P,C =   ut.multipleMECMDP();
#------------------------ value iteration 
policy = pI.valueIteration(P,C, minimize = False);

Markov = P.dot(policy.T);
xk = np.ones(S) /S;
#xk[0] = 1.;
xNext = (Markov).dot(xk);
#print ("xNext = ", xNext);
#print ("xk = ", xk);
it = 0;
while (np.linalg.norm(xk - xNext, 2) >= 5e-2) and  it < 1000 :
    print ("in while loop iteration ", it, "  norm difference ", np.linalg.norm(xk - xNext, 2)  );
    xk = 1.0*xNext;
    xNext = (Markov).dot(xk);
    it += 1;

expectedCost = np.sum(xNext.dot(policy).dot(np.reshape(C,(S*A))));
print ("expected cost from stationary distribution: ", expectedCost);  

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
objective = cvx.Minimize(sum([sum([y[s,a]*C[s,a] for s in range(S)]) for a in range(A)]))
mdp = cvx.Problem(objective, constraints);
mdpRes = mdp.solve();
print ("expected cost from LP formulation: ",mdpRes)