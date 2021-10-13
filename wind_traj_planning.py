# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 06:14:28 2021

@author: craba
"""
import numpy as np
import wind_mdp.wind_generator as wind
import dynamicProg as dp
import matplotlib.pyplot as plt

np.random.seed(456)
length = 50
width = 50
P, R = wind.mdp_gen(length, width, 0.5)   
values, policy = dp.value_iteration(P, R, minimize=False, g=0.98)

print(f'values shape {values.shape}')
value_grid = values.reshape((length, width))
print(f'value grid shape {value_grid.shape}')

plt.imshow(value_grid, interpolation='nearest')
plt.colorbar()
plt.show() 
            