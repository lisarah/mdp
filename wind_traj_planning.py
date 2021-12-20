# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 06:14:28 2021

@author: craba
"""
import numpy as np
import wind_mdp.wind_generator as wind
import dynamicProg as dp
import matplotlib.pyplot as plt
import visualization as vs


# np.random.seed(456)
length = 9
width = 9
N = 100
mag_bounds = (np.ones(N), np.ones(N) * 2)
ang_bounds = (-np.pi * np.ones(N), np.pi * np.ones(N))
P, R = wind.mdp_gen(length, width, 0.1, mag_bound=mag_bounds, 
                    ang_bound=ang_bounds)   
values, policy = dp.value_iteration(P, R, minimize=False, g=0.99)

S = length * width
_, A = R.shape 
print(f'values shape {values.shape}')
value_grid = values.reshape((width, length))
print(f'value grid shape {value_grid.shape}')

plt.figure()
plt.imshow(value_grid, interpolation='nearest')
plt.colorbar()
plt.show() 


value_list = value_grid.flatten()
cost_plot, val_grids, _ = vs.init_grid_plot(width, length, value_list)

original_policy = []
for s in range(S):
    pol, = np.argwhere(policy[s, s*A:(s+1)*A] == 1) 
    # print(pol[0])
    original_policy.append(pol[0])
    
    
wind.draw_policies(width, length, original_policy, cost_plot)
plt.show()