# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:20:37 2021

@author: Sarah Li
"""
import numpy as np
import visualization as vs
import util as ut
import matplotlib.pyplot as plt
import dynamicProg as dp

Columns = 10
Rows = 5
A = 4
P = ut.nonErgodicMDP(Rows, Columns, p=0.8)
C = np.zeros((Rows* Columns, A))
target_col = [4,5]
target_row = Rows-1
for a in range(A):
    C[target_row*Columns + target_col[0], a] = -1. 
    C[target_row*Columns + target_col[1], a] = -1.
# C[ 0, :] = np.array([-1 for c in range(Columns)])

player_num = 2
gamma = 0.9
step_size = 0.01
opponent_list = [1,0]
policy = ut.random_initial_policy(Rows, Columns, A, player_num)
            
Iterations = 1000
V_hist= [[], []]
for i in range(Iterations):
    next_policy = []
    costs = []
    for p in range(player_num):
        opponent = opponent_list[p]
        # p_cost = C
        p_cost = vs.update_cost(C, P, policy[:,:,opponent], Rows, Columns, 1.25)
        V, pol_new = dp.value_iteration(P, p_cost, g=gamma)
        next_policy.append(pol_new)
        costs.append(p_cost)
    for p in range(player_num):
        policy[:,:,p] = (1 - step_size) * policy[:,:,p] + step_size * next_policy[p]
        V_hist[p].append(list(ut.value(P, policy[:,:,p], costs[p], gamma)))
        
V_hist_array = np.array(V_hist)
plt.figure()
for s in range(Rows*Columns):
    plt.plot(V_hist_array[0,:, s])
plt.show()

color_map, norm = vs.color_map_gen(V_hist_array[0,-1,:]) 

ax, value_grids = vs.init_grid_plot(Rows, Columns, V_hist_array[0, -1, :])


p1_init = np.random.randint(0, Columns - 1) # start randomly in the top row
p2_init = np.random.randint(0, Columns - 1) # start randomly in the top row
vs.simulate(p1_init, p2_init, policy, V_hist_array[0, -1, :], value_grids, A, 
            Rows, Columns, P,  Time = 100)


# v_max_1 = np.max(V_hist_array[0, -1, :])
# v_min_1 = np.min(V_hist_array[0, -1, :])
# # v_max_2 = np.max(V_hist_array[1, -1, :])
# # v_min_2 = np.min(V_hist_array[1, -1, :])
# norm_1 = mpl.colors.Normalize(vmin=v_min_1, vmax=v_max_1)
# # norm_2 = mpl.colors.Normalize(vmin=v_min_2, vmax=v_max_2)
# color_map = plt.get_cmap('coolwarm')    
# f_1, axis_1 = plt.subplots(1)
# value_grids_1 = []
# for x_ind in range(Rows):
#     value_grids_1.append([])
#     for y_ind in range(Columns):
#         R,G,B,_ = color_map(norm_1((V_hist_array[0, -1, x_ind*Columns+ y_ind])))
#         color = [R,G,B]  
#         value_grids_1[-1].append(plt.Rectangle((y_ind, x_ind), 1, 1, 
#                                        fc=color, ec='xkcd:greyish blue'))
#         axis_1.add_patch(value_grids_1[-1][-1])
# plt.axis('scaled')
# axis_1.xaxis.set_visible(False)  
# axis_1.yaxis.set_visible(False)