# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 17:17:25 2020

@author: craba
"""

import util as ut
import numpy as np
import dynamicProg as dp
row = 5; col = 3; A = 4;
P = ut.rectangleMDP(row, col, p = 0.6);
C = np.random.rand(row*col, A);
gamma = 0.7;

print ("----------------Value iteration ---------------");
v_VI = dp.discounted_valueIteration(P,C, True, gamma);
print ("value function = ", v_VI);
print ("----------------Policy iteration ---------------");
pi_PI, v_PI = dp.policyIteration(P,C, gamma);
print ("value function = ", v_PI);