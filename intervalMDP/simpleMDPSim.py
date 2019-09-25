# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:36:41 2019

@author: craba
"""

import numpy as np
import util as ut
import matplotlib.pyplot as plt

P, C, lowEps, highEps = ut.simpleMDP();
#print ("Cost is ", C);
T = 20;
S,A = C.shape;
plt.close('all');

Vk = np.zeros((S));
VHist = np.zeros((2, T));
for i in range(T):
    print ("-------------- iteration ", i, " -------------------")
    BO = C + np.reshape(Vk.dot(P), (S,A));
    print ("Bellman Operator:");
    print (BO);
    Vk = np.min(BO, axis = 1); 
    print ("Value function ", Vk);
    VHist[:, i] = 1*Vk;
plt.figure();
plt.title("Converging value function")
plt.plot(VHist[0,:], VHist[1,:]);
plt.show();

plt.figure();
plt.title("Converging V^\star function")
plt.plot(VHist[0,1:T] - VHist[0,0:T-1], VHist[1,1:T] - VHist[1,0:T-1]);
plt.show()


#--------- choose random starting points --------------------#
print ("Sampling random cost functions");
T = 500;
Samples = 100;
VRes = np.zeros((S,Samples));
for sample in range(Samples):
    Cs = 1.0*C;
    for s in range(S):
        for a in range(A):
            ran = highEps[s,a] - lowEps[s,a];
            Cs[s,a] += ran*np.random.random() + lowEps[s,a];
    
    Vk = np.zeros((S));        
    for i in range(T):
        BO = Cs + np.reshape(Vk.dot(P), (S,A));
        Vk = np.min(BO, axis = 1); 
    
    VRes[:,sample] = 1.0*Vk; 
VBound = np.zeros((S, 16));
for i in [0,1]:
    for j in [0,1]:
        for k in [0,1]:
            for l in [0,1]:
                Cs = 1.0*C;
                if i == 0:
                    Cs[0,0] += lowEps[0,0];
                else:
                    Cs[0,0] += highEps[0,0];
                if j == 0:
                    Cs[0,1] += lowEps[0,1];
                else:
                    Cs[0,1] += highEps[0,1];
                if k == 0:
                    Cs[1,0] += lowEps[1,0];
                else:
                    Cs[1,0] += highEps[1,0]; 
                if l == 0:
                    Cs[1,1] += lowEps[1,1];
                else:
                    Cs[1,1] += highEps[1,1];
                    
                #Run the value iteration
                Vk = np.zeros((S));        
                for t in range(T):
                    BO = Cs + np.reshape(Vk.dot(P), (S,A));
                    Vk = np.min(BO, axis = 1); 
#                print (Vk)
                boundNum = 8*i + 4*j + 2*k + 1*l;    
                VBound[:,boundNum] = 1.0*Vk;        

plt.figure();
plt.title("Samples of cost function")
plt.scatter(VRes[0,:], VRes[1,:]);
plt.scatter(VBound[0,:], VBound[1, :], marker = 'o')
plt.grid();
plt.show();    
    
