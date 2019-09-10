# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:21:26 2019

@author: craba
"""
import numpy as np
#import matplotlib.pyplot as plt
#----------------------------------------------------------------------------#
"""
Policy Iteration
Input:  P   - probability kernel - S x SA
        c   - cost - S x A
Output: pi  - optimal policy - S x A
        
uses policy iteration to compute the optimal policy defined by (P,c)
"""
gamma = 0.9;
def policyIteration(P, c):
    S, A = c.shape;
#    print (S, "    ", A)
    # generate a policy that always chooses the first action
    pik = np.zeros((S, S*A));

    newpi = np.zeros((S,S*A));
    for i in range(S):
        newpi[i, i*A] = 1.;
    Vk = newpi.dot(np.reshape(c, (S*A,1)));
    it = 0;
    while np.min(np.equal(pik, newpi)) == False:
        pik = 1.0*newpi;
        newpi = np.zeros((S,S*A));
        it += 1;
        Vk = np.reshape(c,(1,S*A)).dot(pik.T).dot(np.linalg.inv(np.eye(S) - gamma*P.dot(pik.T)))
        BO = c + np.reshape(Vk.dot(P), (S,A));
        print (Vk)
        piS = np.argmin(BO, axis=1);
        for s in range(S):
            if (it %1000) == 0:
                print(s*A + piS[s]);
            newpi[s, s*A + piS[s]] = 1.;
    print ("converged at iteration: ", it) 
    return pik;

def BVI(P, sP, sN, eps):
    print ("------------- BVI -------------");
    maxiter = 100;
    S, SA = P.shape;
    A = int(SA/S);
    L = np.zeros((S));
    U = np.ones((S));
    L[sP] = 1.; U[sP] = 1.;
    U[sN] = 0; L[sN] = 0.;
    
    
    delta = np.zeros(maxiter);
    delta[0] = np.max(abs(L - U));
    it = 0;
    while delta[int(it)] >= eps and it < maxiter - 1:
        it += 1.;
        nextL = np.min(np.reshape(L.dot(P), (S,A)), axis=1);
        nextU = np.min(np.reshape(U.dot(P), (S,A)), axis=1);
        nextL[sP] = 1.; nextU[sP] = 1.;
        nextU[sN] = 0.; nextL[sN] = 0.;
        L = nextL;
        U = nextU;
        delta[int(it)] = np.max(abs(L - U));
        if it % 1 == 0:
            print ("Iteration ", it, " Value Inf Norm: ", delta[int(it)]);
#    plt.figure();
#    plt.plot(delta); 
#    plt.yscale('log');
#    plt.show();       
    print ("--------- end of BVI ---------------");
    return L, U;
def valueIteration(P,c):
    print ("------------- value iteration -------------")
    S, A = c.shape;
#    print (S, "    ", A)
    # generate a policy that always chooses the first action
    pik = np.zeros((S));
    newpi = np.zeros((S,S*A));
    Vk = np.min(c, axis  =1);
    Vnext = np.zeros(S);
    it = 0;
    eps = 1e-5;
    while np.linalg.norm(Vk - Vnext, 2) >= eps:
        Vk  = 1.0*Vnext;
        it += 1;
#        Vk = np.reshape(c,(1,S*A)).dot(pik.T).dot(np.linalg.inv(np.eye(S) - gamma*P.dot(pik.T)))
        BO = c + gamma*np.reshape(Vk.dot(P), (S,A));
        Vnext = np.min(BO, axis = 1);
        pik= np.argmin(BO, axis = 1);
        if it%10 == 0:
            print ("norm ", np.linalg.norm(Vk - Vnext, 2));
    for s in range(S):
        newpi[s, s*A + pik[s]] = 1.;

    print ("converged at iteration: ", it) 
    print ("--------- end of value iteration ---------------")
    return newpi;

