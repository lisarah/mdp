# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:55:07 2019

@author: craba
"""

import dynamicProg as dP
import util as ut

print ("----- Minimum recheability on a toy example ---- ");
P = ut.nonErgodicToy();
LV, UV = dP.BVI(P, 0, 2, 1e-2);
print ("Over approximation");
print (LV);
print ("Under approximation");
print (UV);

print ("---- Minimum recheability under and over approximation ----");
N = 5; M = 3;
p = 0.9;
P = ut.nonErgodicMDP(M, N, p);
LV, UV = dP.BVI(P, 0, 14, 1e-2);
print ("Over approximation");
print (LV);
print ("Under approximation");
print (UV);


        