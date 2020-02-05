#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:16:53 2019

@author: sarahli
"""
import util as ut
import numpy as np
import matplotlib.pyplot as plt
import dynamicProg as dp
plt.close('all')
N = 3; 
M = 3;
S = N*M; A = 4;
gamma = 0.5;
P = ut.rectangleMDP(N,M,0.7);
"""
    Cost model:
        player x: C = C1 + C2.dot(y)
        player y: C = C1 + C2.dot(x)
"""
C1 = 0.1*np.random.rand(S,A);
C2 = 0.2*np.random.rand(S,A);
C3 = 0.3*np.random.rand(S,A);
C4 = 0.4*np.random.rand(S,A);

Vbound1 = dp.valueIteration(P,C1,g=gamma);
Vbound2 = dp.valueIteration(P,C2,g=gamma);
Vbound3 = dp.discounted_valueIteration(P,C3,g=gamma);
Vbound4 = dp.discounted_valueIteration(P,C4,g=gamma);
T = 100;
Samples = 100;
timeLine = np.arange(0,T,T);
Vx = np.zeros((S,Samples)); 
for sample in range(Samples):
    # cost
    theta = np.random.rand(4);
    theta = theta/np.sum(theta);
#    print (np.sum(theta))
    C = theta[0]*C1 + theta[1]*C2 + theta[2]*C3 + theta[3]*C4;
    if ((C1 - C).all() >= 0 or
        (C2 - C).all() >= 0 or
        (C3 - C).all() >= 0 or
        (C4 - C).all() >= 0):
        print ("Is in polyhedral set")
    else:
        print ("Not upperbounded")
    Vx[:, sample] = dp.discounted_valueIteration(P,C,g=gamma);




plt.figure();
for sample in range(Samples):
    plt.bar(sample, np.linalg.norm(Vx[:,sample], ord = 2, axis = 0));
plt.bar(Samples, np.linalg.norm(Vbound1, ord = 2, axis = 0), label= "upper bound")
plt.bar(Samples+1, np.linalg.norm(Vbound2, ord = 2, axis = 0), label= "lower bound");
plt.bar(Samples+2, np.linalg.norm(Vbound3, ord = 2, axis = 0), label= "value iteration");
plt.bar(Samples+3, np.linalg.norm(Vbound4, ord = 2, axis = 0), label= "policy iteration");
#plt.legend(); 
plt.grid();
plt.show()

#def variance()

