# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:21:26 2019

@author: craba
"""
import numpy as np
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

def valueIteration(P,c):
    S, A = c.shape;
#    print (S, "    ", A)
    # generate a policy that always chooses the first action
    pik = np.zeros((S, S*A));

    newpi = np.zeros((S,S*A));
    for i in range(S):
        newpi[i, i*A] = 1.;
    Vk = 1.0*c;
    Vnext = np.zeros(S);
    it = 0;
    eps = 9e-3;
    while np.linalg.norm(Vk - Vnext, 2) >= eps:
        Vk  = 1.0*Vnext;
        it += 1;
#        Vk = np.reshape(c,(1,S*A)).dot(pik.T).dot(np.linalg.inv(np.eye(S) - gamma*P.dot(pik.T)))
        BO = c + gamma*np.reshape(Vk.dot(P), (S,A));
        Vnext = np.min(BO, axis = 1);

    print ("converged at iteration: ", it) 
    return pik;