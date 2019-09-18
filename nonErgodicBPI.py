# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:45:43 2019

@author: craba
"""
import dynamicProg as dP
import util as ut
import numpy as np
print ("----- Minimum reachability on a toy example ---- ");
P = ut.nonErgodicToy();
LV, UV = dP.BPI(P, 0, 2, 1e-2);
print ("Over approximation");
print (LV);
print ("Under approximation");
print (UV);

print ("---- Minimum reachability under and over approximation ----");
N = 5; M = 3;
p = 0.9;
P = ut.rectangleMDP(M, N, p);
LV, UV = dP.BPI(P, 0, 14, 1e-2);
print ("Over approximation");
print (np.sum(UV, axis = 0));
print ("Under approximation");
print (np.sum(LV, axis = 0));

#xU = ut.stationaryDist(P, UV, state = 0, isMax = True);

#xL =  ut.stationaryDist(P, UV, state = 0, isMax = False);

# eigen value decomposition to find this 
wU, eigU = np.linalg.eig(P.dot(LV.T));
wL, eigL = np.linalg.eig(P.dot(UV.T));
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
    if eigVec[0] > VkU[0]:
        VkU = 1.0*eigVec;
        
stationaryL = [];
for i in oneEigL:
    stationaryL.append(eigL[:,i]);
VkL = stationaryL[0];
for eigVec in stationaryL:
    if eigVec[0] < VkL[0]:
        VkL = 1.0*eigVec;
        
print ("upper bounding stationary distribution ")
print ((xU));
print ("Lower bounding stationray distribuiton ")
#print ((xL));