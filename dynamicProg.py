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
def value_iteration_finite(P,c, minimize = True, g = 1.):
    plt.close('all')
    S, A, T_over = c.shape
    T = T_over - 1
    pik = np.zeros((S, T_over));
    newpi = np.zeros((S,S*A, T));
    Vk = np.zeros((S, T_over));
    BO = 1*c
    Vk[:, T]= np.min(BO[:,:,T], axis=1)
    for t in range(T):
        t_ind = T - 1 - t   # T-1 ... 0
        BO[:,:,t_ind] +=  g*np.reshape(Vk[:,t_ind+1].dot(P), (S,A))
        Vk[:,t_ind] = np.min(BO[:,:,t_ind], axis=1)
        pik[:,t_ind] = np.argmin(BO[:,:,t_ind], axis=1)
        
    for s in range(S):
        for t in range(T):
            newpi[s, int(s*A + pik[s,t]), t] = 1.

    return Vk, newpi

def value_iteration(P,c, minimize = True, g = 1.):
    plt.close('all')
    S, A = c.shape
    Iterations = 1000 # 1e3
    pik = np.zeros((S));
    newpi = np.zeros((S,S*A));
    Vk = np.zeros(S)
    if minimize:
        Vnext = np.min(c, axis=1)
    else:
        Vnext = np.max(c, axis=1)
    it = 1;
    eps = 1e-10;
    while np.linalg.norm(Vk - Vnext, ord = 2) >= eps and it < Iterations:
        Vk  = Vnext # - np.min(Vnext);
        it += 1
        BO = c + g*np.einsum('ijk,i',P,Vnext)
        if minimize:
            Vnext = np.min(BO, axis = 1)
            pik= np.argmin(BO, axis = 1)
        else:
            Vnext = np.max(BO, axis = 1)
            pik= np.argmax(BO, axis = 1)
        
    for s in range(S):
        newpi[s, s*A + pik[s]] = 1.

    return Vk, newpi


def value_iteration_polytopes(P, C, gamma = 0.9):
    S, S, A, Ip = P.shape
    V_min =  float("-inf")*np.ones(S)
    V_max =  float("-inf")*np.ones(S)
    C_max = np.max(C, axis=2)
    C_min = np.min(C, axis=2) 
    V_next_min = np.zeros(S) # float("-inf")*np.ones(S)
    V_next_max = np.zeros(S) # float("-inf")*np.ones(S)
    it = 0
    pi_opt = np.zeros(S)
    pi_rbt = np.zeros(S)
    Iterations = 1e3
    while np.linalg.norm(V_min - V_next_min, ord = 2) >= 1e-5 and \
          np.linalg.norm(V_max - V_next_max, ord = 2) >= 1e-5 and \
          it < Iterations:
        print(f'\r it {it}         ', end = '')
        V_min = 1 * V_next_min
        V_max = 1 * V_next_max
        for s in range(S):
            q_min = np.zeros(A)
            q_max = np.zeros(A)
            for a in range(A):
                q_min_a = C_min[s,a] + gamma*np.einsum('ij,i',P[:, s, a, :], V_min)
                q_max_a = C_max[s,a] + gamma*np.einsum('ij,i',P[:, s, a, :], V_max)
                q_min[a] = np.min(q_min_a)
                q_max[a] = np.max(q_max_a)
            V_next_min[s] = np.min(q_min)
            V_next_max[s] = np.min(q_max)
            # print(f' s = {s}, previous V: {V_min[s]} new V: {V_next_min[s]}')
        # print (f'errors {np.linalg.norm(V_min - V_next_min, ord = 2) } {np.linalg.norm(V_max - V_next_max, ord = 2)}')
        it += 1
    for s in range(S):
        q_min = np.zeros(A)
        q_max = np.zeros(A)
        for a in range(A):
            q_min_a = C_min[s,a] + gamma*np.einsum('ij,i',P[:, s, a, :], V_next_min)
            q_max_a = C_max[s,a] + gamma*np.einsum('ij,i',P[:, s, a, :], V_next_max)
            q_min[a] = np.min(q_min_a)
            q_max[a] = np.max(q_max_a)
        pi_opt[s] = np.argmin(q_min)
        pi_rbt[s] = np.argmin(q_max)
    return V_min, V_max, pi_opt, pi_rbt

#----------------------------------------------------------------------------#
"""
Policy Iteration
Input:  P   - probability kernel - S x SA
        c   - cost - S x A
Output: pi  - optimal policy - S x A
        
uses policy iteration to compute the optimal policy defined by (P,c)
"""
def policyIteration(P, c, gamma):
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
#        print (Vk)
        piS = np.argmin(BO, axis=1);
        for s in range(S):
            if (it %1000) == 0:
                print(s*A + piS[s]);
            newpi[s, s*A + piS[s]] = 1.;
    print ("converged at iteration: ", it) 
    return pik, Vk;

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
    Vnext = np.min(c, axis=1);
    it = 1;
    eps = 1e-4;
    while stoppingCriterion(Vnext/it, Vk) >= eps and it < Iterations:
        Vk  = 1.0*Vnext/it;
        VHist[:, it-1]= Vk;
        it += 1;
        # BO = c + g*np.reshape(Vnext.dot(P), (S,A));
        BO = c + g*np.einsum('ijk,i',P,Vnext)
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

def game_VI(P, c, S1, S2, gamma):
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