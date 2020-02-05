# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:29:55 2019

@author: craba
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



def makeWheatStone():
    costParam = 0.1;
    A = np.diag([9., costParam, costParam, 9., costParam, costParam, costParam]);
    b = np.array([costParam, 1.0, 0, costParam, 1.]);
    # ell = Ay + b + eps
    F = np.array([[ 1.,  0.,  0.,   0.,  1., -1.], 
              [ -1.,  1.,  1.,   0.,  0.,  0.], 
              [  0.,  0.,  -0.9,   1., -1.,  0.], 
              [  0., -1.,   -0.1,   -1., 0.,  1.]]);
    #N = np.array([[ 1.,  0.,  0.,   0.,  1.], 
    #              [ -1.,  1.,  1.,   0.,  0.], 
    #              [  0.,  0.,  -0.6, 1., -1.], 
    #              [  0., -1.,  -0.4, -1., 0.],]);
    N = np.array([[ 1.,  0.,  0.,   0.,  1., -1.], 
                  [ -1.,  1.,  1.,   0.,  0.,  0.], 
    #              [  0.,  0.,  -0.6, 1., -1.,  0.], 
                  [  0., -1.,  -0.1, -1., 0.,  1.],
                  [  1.,  1.,  1.,   1.,  1.,  1.]]);
    GInv = np.linalg.inv(A);
    NGN = np.linalg.inv(N.dot(GInv).dot(N.T))
