# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:21:26 2019

@author: craba
"""
import numpy as np
import matplotlib.pyplot as plt


def epsGreedy(Q, s, eps):
#    print ("----------- in eps Greedy ---------------")
    S,A = Q.shape;   
    m = np.max(Q[s,:], axis = 0);
    maxAction = [i for i, j in enumerate(Q[s,:]) if j == m];
    mSize = len(maxAction);
    
    pol = np.ones(A)*eps/(A-mSize);
    if mSize ==A:
#        print ("mSize is equal to A")
        pol = np.ones(A)/A;
    else:
        for optAction in maxAction:
            pol[optAction] = (1-eps)/mSize;
#    print (pol)
#    print ("------------ leaving eps greedy ---------------")
    return pol;
     

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
        print ((np.eye(S) - gamma*P.dot(pik.T)).shape)
        print (pik.shape);
        Qk = (np.linalg.inv(np.eye(S) - gamma*P.dot(pik.T))).dot(pik).dot(c.T)
        Vk = np.zeros(S);
        for i in range(S):
            Vk[i] = sum(Qk[i*S:(i+1)*S]);
        BO = c + gamma*np.reshape(Vk.dot(P), (S,A));
        print (Vk)
        piS = np.argmin(BO, axis=1);
        for s in range(S):
            if (it %1000) == 0:
                print(s*A + piS[s]);
            newpi[s, s*A + piS[s]] = 1.;
    print (" Converged at iteration: ", it) 
    print (" ")
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
        nextL[sP] = 1.; 
        nextU[sP] = 1.;
        nextU[sN] = 0.;
        nextL[sN] = 0.;
        L = nextL;
        U = nextU;
        delta[int(it)] = np.max(abs(L - U));
        if it % 10 == 1:
            print ("Iteration ", it, " Value Inf Norm: ", delta[int(it)]);
            print (U);
            print (L)
#    plt.figure();
#    plt.plot(delta); 
#    plt.yscale('log');
#    plt.show();       
    print ("--------- end of BVI ---------------");
    return L, U;

def BPI(P, sP, sN, eps):
    print ("------------- BPI -------------");
    maxiter = 100;
    S, SA = P.shape;
    A= int(SA/S);
#    print ("number of states: ", S);
#    print ("number of actions: ", A);
#    print ("Probability ", P);
    print ("States: ", S, " Actions: ", A);
    pkU = np.zeros((S, S*A));
    pkL = np.zeros((S, S*A));
    newU = np.zeros((S,S*A));
    newL = np.zeros((S,S*A));
    for i in range(S):
        newL[i, i*A] = 1.;
        newU[i, i*A] = 1.;

    it = 0;
    while np.min(np.equal(pkL, newL)) == False \
        and np.min(np.equal(pkU, newU)) == False \
        and it < maxiter:
        pkU = 1.0*newU;
        pkL = 1.0*newL;
        it += 1;
        
        wU, eigU = np.linalg.eig(P.dot(pkU.T));
        wL, eigL = np.linalg.eig(P.dot(pkL.T));
        oneEigU = np.where(wU >=1-1e-9)[0];
        oneEigL = np.where(wL >=1-1e-9)[0];
#        print ("oneEigU: ", oneEigU);
#        print ("oneEigL: ", oneEigL);
#        print ("wU: ", wU );
#        print ("wL: ", wL );
        stationaryU = []; 
        for i in oneEigU:
            stationaryU.append(eigU[:,i]);
        VkU = stationaryU[0];
        for eigVec in stationaryU:
            if eigVec[sP] > VkU[sP]:
                VkU = 1.0*eigVec;
        
        stationaryL = [];
        for i in oneEigL:
            stationaryL.append(eigL[:,i]);
        VkL = stationaryL[0];
        for eigVec in stationaryL:
            if eigVec[sP] < VkL[sP]:
                VkL = 1.0*eigVec;
        print ("VkU : ", VkU);
        print ("VkL : ", VkL);
        piU = np.argmax(np.reshape(VkU.dot(P), (S,A)), axis=1);
        piL = np.argmin(np.reshape(VkL.dot(P), (S,A)), axis=1);
        newU = np.zeros((S,S*A));
        newL = np.zeros((S,S*A));
        if (it%1) == 0:
                print ("Iteration ",it);
        for s in range(S):
#            if (it%1) == 0:
#                print ("Iteration ",it);
#                print(piU[s]);
#                print(piL[s]);
            newU[s, s*A + piU[s]] = 1.;
            newL[s, s*A + piL[s]] = 1.;
#        print(np.sum(newU, axis = 0));
#        print (np.sum(P.dot(newU.T), axis = 0));
#        print (np.sum(newL, axis = 0));
#        print (np.sum(P.dot(newL.T), axis = 0));
    print ("converged at iteration: ", it); 
#    print ("Maximum probability: ", VkU );
#    print ("Minimum probability: ", VkL);
       
    print ("--------- end of BPI ----------");
    return newL, newU;

def valueIteration(P,c, minimize = True, returnV = False, g = 1.):
    print ("------------- value iteration -------------")
    plt.close('all');
    S, A = c.shape;
    Iterations = 100000;
    VHist  = np.zeros((S,Iterations));
#    print (S, "    ", A)
    # generate a policy that always chooses the first action
    pik = np.zeros((S));
    newpi = np.zeros((S,S*A));
    Vk = np.zeros(S);
    Vnext = np.min(c, axis  =1);
    it = 1;
    eps = 1e-4;
    while stoppingCriterion(Vnext/it, Vk) >= eps and it < Iterations:
        Vk  = 1.0*Vnext/it;
        VHist[:, it-1]= Vk;
        it += 1;
#        Vk = np.reshape(c,(1,S*A)).dot(pik.T).dot(np.linalg.inv(np.eye(S) - gamma*P.dot(pik.T)))
        BO = c + g*np.reshape(Vnext.dot(P), (S,A));
        if minimize:
            Vnext = np.min(BO, axis = 1);
            pik= np.argmin(BO, axis = 1);
        else:
            Vnext = np.max(BO, axis = 1);
            pik= np.argmax(BO, axis = 1);
        if it%100 == 0:
            print ("stopping criteria ", stoppingCriterion(Vnext/it, Vk));
#            print ("V difference ", Vnext - Vk);
    for s in range(S):
        newpi[s, s*A + pik[s]] = 1.;

#    plt.figure();
#    plt.plot(np.linalg.norm(VHist, ord = 2, axis = 0));
#    plt.yscale('log');
#    plt.grid();
#    plt.show();
#    print ("converged at iteration: ", it) 
#    print ("Final value function: ", Vk)
    print ("--------- end of value iteration ---------------")
    if returnV:
        return Vk*it;
    else:
        return newpi;
def discounted_valueIteration(P,c, minimize = True, g = 1.):
#    print ("------------- value iteration -------------")
    plt.close('all');
    S, A = c.shape;
    Iterations = 100000;
    VHist  = np.zeros((S,Iterations));
#    print (S, "    ", A)
    # generate a policy that always chooses the first action
    pik = np.zeros((S));
    newpi = np.zeros((S,S*A));
    Vk = np.zeros(S);
    Vnext = np.min(c, axis  =1);
    it = 1;
    eps = 1e-10;
    while np.linalg.norm(Vk - Vnext, ord = 2) >= eps and it < Iterations:
        Vk  = 1.0*Vnext;
        VHist[:, it-1]= Vk;
        it += 1;
        BO = c + g*np.reshape(Vnext.dot(P), (S,A));
        if minimize:
            Vnext = np.min(BO, axis = 1);
            pik= np.argmin(BO, axis = 1);
        else:
            Vnext = np.max(BO, axis = 1);
            pik= np.argmax(BO, axis = 1);
        if it%100 == 0:
#            print ("stopping criteria ", stoppingCriterion(Vnext/it, Vk));
            print ("V difference ", Vnext - Vk);
    for s in range(S):
        newpi[s, s*A + pik[s]] = 1.;

#    print ("--------- end of value iteration ---------------")
    return Vk;
def stoppingCriterion(V, VLast):
    w =   V - VLast;
#    if np.min(w)< 0:
#        print ("Something's wrong in stopping criterion");
    return np.max(w) - np.min(w);

def game_VI(P, c, S1, S2):
    print ("------------- value iteration -------------")
    plt.close('all');
    S, A = c.shape;
    VHist  = np.zeros((S,1000));
#    print (S, "    ", A)
    # generate a policy that always chooses the first action
    pik = np.zeros((S));
    newpi = np.zeros((S,S*A));
    Vk = np.zeros(S);
    Vnext = np.min(c, axis=1);
#    print (Vnext);
    it = 1;
    eps = 1e-4;
    while stoppingCriterion(Vnext/it, Vk) >= eps and it < 1000:
        Vk  = 1.0*Vnext/it;
        VHist[:, it-1]= Vk;
        it += 1;
#        Vk = np.reshape(c,(1,S*A)).dot(pik.T).dot(np.linalg.inv(np.eye(S) - gamma*P.dot(pik.T)))
        BO = c + gamma*np.reshape(Vnext.dot(P), (S,A));
        for s in range(S):
#            print (Vnext)
            if s in S1:
                Vnext[s] = np.min(BO[s, :]);
                pik[s] = np.argmin(BO[s, :]);
            else: # s is maximizing state
                Vnext[s] = np.max(BO[s, :]);
                pik[s] = np.argmax(BO[s, :]);
        if it%100 == 0:
            print ("stopping criteria ", stoppingCriterion(Vnext/it, Vk));
#            print ("V difference ", Vnext - Vk);
    for s in range(S):
        newpi[s, s*A + int(pik[s])] = 1.;

    plt.figure();
    plt.plot(np.linalg.norm(VHist, ord = 2, axis = 0));
    plt.yscale('log');
    plt.grid();
    plt.show();
    print ("converged at iteration: ", it) 
    print ("Final value function: ", Vk)
    print ("--------- end of value iteration ---------------")
    return (np.sum(Vk)/S);