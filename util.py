# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:30:56 2019

@author: craba
"""
import numpy as np
"""
Determine the stationary distribution for a given policy by using 
SVD decomposition

"""
def stationaryDist_SVD(P, pi, state = None, isMax = None):
    # eigen value decomposition to find this 
    w, eig = np.linalg.eig(P.dot(pi.T));
    oneEig = np.where(w >=1-1e-9)[0];
    stationary = []; 
    for i in oneEig:
        stationary.append(eig[:,i]);
        print (eig[:,i]);
    stationaryDist = stationary[0];
    if state is not None:
        for eigVec in stationary:
            if isMax:
                if eigVec[state] > stationaryDist[state]:
                    stationaryDist = 1.0*eigVec;
            else:
                if eigVec[state] < stationaryDist[state]:
                    stationaryDist = 1.0*eigVec;
    return stationaryDist;
"""
Determine the stationary distribution for a given policy from definition
of stationary distributions -- iterate Ppi.T for a very long time 
and see what the final distribution is

"""
def stationaryDist(P,pi, T = 100):
    S,SA = P.shape;
    Markov = P.dot(pi.T);
    xk = np.ones(S) /S;
    #xk[0] = 1.;
    xNext = (Markov).dot(xk);
#    print ("xNext = ", xNext);
#    print ("xk = ", xk);
    it = 0;
    while (np.linalg.norm(xk - xNext, 2) >= 1e-8) and  it < T :
#        print ("in while loop",np.linalg.norm(xk - xNext, 2)  );
        xk = 1.0*xNext;
        xNext = (Markov).dot(xk);
        it += 1;
        
    return xNext;
def value(P,pi,C, gamma, N = 100):
    # pi : S \times SA
    S, SA = P.shape;
    M = P.dot(pi.T);

    Vt = np.zeros(S);
     
    for i in range(N):
        VNext = np.reshape(C, (SA)).dot(pi.T) + gamma*Vt.dot(M);
        Vt  = VNext;
        
    return Vt;

def simpleMDP():
    P = np.zeros((2, 4));
    #(s1, a1) = (0.1, 0.9)
    P[0, 0] = 0.1; P[1, 0] = 0.9;
    #(s1, a2) = (0.9, 0.1)
    P[0, 1] = 0.9; P[1, 1] = 0.1;
    #(s2, a1) = (0.1, 0.9)
    P[0, 2] = 0.1; P[1, 2] = 0.9;
    #(s2, a2) = (0.5, 0.5)
    P[0, 3] = 0.5; P[1, 3] = 0.5;
    
    C = np.array([[1, 2], [3,4]]);
    lowEps = np.array([[-0.6, -0.7], [-0.5, -1.]]);
    highEps = np.array([[0.6, 0.7], [0.5, 1.]]);
    
    return P, C, lowEps, highEps;
    
def multipleMECMDP():
    S = 6;
    A = 2;   
    P = np.zeros((S, S*A));
    P[2, 0] = 0.999; P[4,0] = 0.001; P[1,1] = 1.; P[1,2] = 1.; P[3,3] = 1.; 
    P[2,4] = 0.5; P[3,4] = 0.5; P[2,5] = 0.5; P[3,5] = 0.5;
    P[2,6] = 0.5; P[3,6] = 0.5; P[2,7] = 0.5; P[3,7] = 0.5;  
    P[4,8] = 0.5; P[5,8] = 0.5; P[4,9] = 0.5; P[5,9] = 0.5;
    P[4,10] = 0.5; P[5,10] = 0.5; P[4,11] = 0.5; P[5,11] = 0.5;
    C = np.array([[0, 0], [4, 5], [10, 10], [0, 0], [20, 20], [0, 0]]);  
    return P, C;      
"""
    Returns a 3 state toy example that is non-ergodic. 
    3 states S1 S2 S3, one action in each state
    A1: S1 -> S1 probability 1
    A2: S2 -> S1 p=0.2, S2-> S3 p = 0.8
    A3: S3 -> S3 p = 1
"""
def nonErgodicToy():
    P = np.array([[1.0, 0.2, 0], [0 , 0, 0], [0, 0.8, 1.]]);
    return P;
"""
    Returns a rectangular MDP that is non-ergodic
    Grid with row = M, column = N, 
    p = main probability of going down a direction
"""
def nonErgodicMDP(M, N, p):
    A = 4;
    P = np.zeros((N*M, N*M*A));
    for i in range(M):
        for j in range(N):
            s = i*N + j;
#            print (s)
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
#                print (SA)
#                if SA >=56:
#                    print ("--------valid out states ----------")
#                    print (valid)
#                    print ("curr action ", a);
#                    print ("self state: ", s);
#                    print ("i: ", i);
#                    print ("j: ", j);
#                    print ("left: ", left);
#                    print ("right: ", right);
#                    print ("top: ", top);
#                    print ("bottom: ", bottom);
                P = nonErgodic_assignP(a, SA, P,p, valid, lookup, s);   
    return P; 

"""
    Returns a rectangular MDP that is ergodic
    Grid with row = M, column = N, 
    p = main probability of going down a direction
""" 
def rectangleMDP(M,N, p):
    A = 4;
    P = np.zeros((N*M, N*M*A));
    for i in range(M):
        for j in range(N):
            s = i*N + j;
#            print (s)
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
    #            print (SA)
    #            if SA == 25:
    #                print ("--------valid out states ----------")
    #                print (valid)
    #                print ("i: ", i);
    #                print ("j: ", j);
    #                print ("left: ", left);
    #                print ("right: ", right);
    #                print ("top: ", top);
    #                print ("bottom: ", bottom);
                P = assignP(a, SA, P,p, valid, lookup);
                
    
    return P;
"""
    given direction and look up table, derive the correct probability column
    associated with action a, state s, state-action index SA, 
    and probability p of taking valid action
    
    ** this one assigns probability 1 to return to state s if 
    direction is not valid
    lookup = the dictionary to look up direction corresponding to action
             is usually lookup = {0: left, 1: right, 2: top, 3: bottom}
    
    valid = valid directions to go for this state
"""   
def nonErgodic_assignP(a, SA, P, p, valid, lookup,s):
    if lookup[a] not in valid:
        P[s, SA] = 1.;
    else:
        P[lookup[a], SA] = p;
        pBar = (1. - p) /(len(valid)-1);
        for neighbour in valid:
            if neighbour != lookup[a]:
                P[neighbour, SA] = pBar;
    return P;
"""
    given direction and look up table, derive the correct probability column
    associated with action a, state s, state-action index SA, 
    and probability p of taking valid action
    
    ** if direction is not valid, returns equal probability of going
    to neighbouring states
    lookup = the dictionary to look up direction corresponding to action
             is usually lookup = {0: left, 1: right, 2: top, 3: bottom}
    
    valid = valid directions to go for this state
"""   
def assignP(a, SA, P, p, valid, lookup):
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