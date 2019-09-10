# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:30:56 2019

@author: craba
"""
import numpy as np

def nonErgodicToy():
    P = np.array([[1.0, 0.2, 0], [0 , 0, 0], [0, 0.8, 1.]]);
    return P;
"""
Returns a rectangular MDP that is non-ergodic
"""
def nonErgodicMDP(M, N, p):
    A = 4;
    P = np.zeros((N*M, N*M*A));
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

def rectangleMDP(M,N, p):
    A = 4;
    P = np.zeros((N*M, N*M*A));
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
