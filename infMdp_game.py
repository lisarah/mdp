# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:52:53 2019

@author: craba
"""
import numpy as np
import policyIteration as pI
import cvxpy as cvx
# need to generate MDP 
# P : S x SA, column stochastic
N = 3; # columns
M = 5; # rows
A = 4;
P = np.zeros((N*M, N*M*A));
p = 0.7;
# ell  = Ay + b
A = np.random.rand(N*M,A)*100.;
b = np.random.rand(N*M,A)*10.;



# create the MDP
def assignP(P, p, valid, lookup,s ):
    if lookup[a] not in valid:
        newp = 1./(len(valid));
        for neighbour in valid:
            P[neighbour, SA] = newp;
    else:
        P[lookup[a], SA] = p;
        pBar = (1. - p) /(len(valid)-1);
        for neighbour in valid:
            if neighbour != lookup[a]:
                P[neighbour, SA] = pBar;
    return P;
            
for i in range(M):
    for j in range(N):
        s = i*N + j;
        print (s)
        left = i*N + j-1;
        right = i*N + j + 1;
        top = (i-1)*N + j;
        bottom = (i+1)*N + j;

        valid = [];
        if s%N != 0:
            valid.append(left);
        if s%N != N-1:
            valid.append(right);
        if s >= N:
            valid.append(top);
        if s < (M*N - N):
            valid.append(bottom);

        lookup = {0: left, 1: right, 2: top, 3: bottom};
        for a in range(A):
            SA = s*A+ a; 
            P = assignP(P,p, valid, lookup, s);


# random initial distribution
pi0 = np.random.rand((N*M, N*M*A));
                    
policy = pI.policyIteration(P,C);

Markov = P.dot(policy.T);
xk = np.ones(N*M) /(N*M);
#xk[0] = 1.;
xNext = (Markov).dot(xk);
print ("xNext = ", xNext);
print ("xk = ", xk);
it = 0;
while (np.linalg.norm(xk - xNext, 2) >= 1e-8) and  it < 100 :
    print ("in while loop",np.linalg.norm(xk - xNext, 2)  );
    xk = 1.0*xNext;
    xNext = (Markov).dot(xk);
    it += 1;

expectedCost = np.sum(xNext.dot(policy).dot(np.reshape(C,(N*M*A))));
print ("expected cost from stationary distribution: ",expectedCost);  


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
objective = cvx.Minimize(sum([sum([y[s,a]*C[s,a] for s in range(N*M)]) for a in range(A)]))
mdp = cvx.Problem(objective, constraints);
mdpRes = mdp.solve();
print ("expected cost from LP formulation: ",mdpRes)