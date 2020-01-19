# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:52:53 2019

@author: craba
"""
import numpy as np
import dynamicProg as pI
import cvxpy as cvx
import util as ut
# need to generate MDP 
# P : S x SA, column stochastic
N = 5; # columns
M = 5; # rows
A = 4;
p = 0.7;

P = ut.rectangleMDP(M,N,p);
C = np.random.rand(N*M,A)*100.;
#----------------------- policy iteration                    
policy = pI.policyIteration(P,C);
#------------------------ value iteration 
#policy = pI.valueIteration(P,C);

Markov = P.dot(policy.T);
xk = np.ones(N*M) /(N*M);
#xk[0] = 1.;
xNext = (Markov).dot(xk);
#print ("xNext = ", xNext);
#print ("xk = ", xk);
it = 0;
while (np.linalg.norm(xk - xNext, 2) >= 5e-2) and  it < 100 :
    print ("in while loop iteration ", it, "  norm difference ", np.linalg.norm(xk - xNext, 2)  );
    xk = 1.0*xNext;
    xNext = (Markov).dot(xk);
    it += 1;

expectedCost = 0.9*np.sum(xNext.dot(policy).dot(np.reshape(C,(N*M*A))));
print ("expected cost from stationary distribution: ", expectedCost);  

y = cvx.Variable((N*M,A));
constraints = [];
ones = np.ones(A);

for i in range(N*M):
    constraints.append(ones*(y[i,:]) == 
                       sum([sum([P[i,s*A +a]*y[s,a] for s in range(N*M)])
                       for a in range(A)]));
constraints.append(y >= 0);
constraints.append( sum([sum([y[s,a] for s in range(N*M)]) for a in range(A)]) == 1.);
gamma= 0.9;
objective = cvx.Minimize(gamma*sum([sum([y[s,a]*C[s,a] for s in range(N*M)]) for a in range(A)]))
mdp = cvx.Problem(objective, constraints);
mdpRes = mdp.solve();
print ("expected cost from LP formulation: ",mdpRes)