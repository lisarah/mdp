# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:22:30 2021

@author: Sarah Li
"""
import numpy as np
import visualization as vs
import util as ut
import matplotlib.pyplot as plt
import dynamicProg as dp
import random


np.random.seed(121)
Columns = 10
Rows = 5
A = 4
T = 100
P = ut.nonErgodicMDP(Rows, Columns, p=0.98)
player_num = 3

C = [np.ones((Rows* Columns, A, T+1)) for _ in range(player_num)]


def update_cost(y):
    congestion_cost = np.zeros((Rows*Columns, A, T+1))
    
    for x_ind in range(Columns):
        for y_ind in range(Rows):
            unwrap_ind = (y_ind * Columns + x_ind)
            for t in range(T+1):
                congestion_cost[unwrap_ind,:, t] += np.sum(
                    y[unwrap_ind * A:(unwrap_ind + 1) * A, t])
    return congestion_cost

def update_y(policy, x_0): # policy should be 1D array
    x = np.zeros((Rows*Columns, T+1))
    x[:, 0] = x_0
    y = np.zeros((Rows*Columns*A, T+1))
    for t in range(T):
        markov_chain = P.dot(policy[:,:,t].T)
        x[:, t+1] = markov_chain.dot(x[:, t]) 
        for s in range(Rows*Columns):
            y[s*A:(s + 1)*A, t] = x[s, t] * policy[s, s*A:(s + 1)*A, t]
    for s in range(Rows*Columns):
         y[s*A:(s + 1)*A, T] = x[s, T] / A
    # print(np.round(x[:,T],2))
    # print(x[:,T].shape)
    # print(np.sum(x[:,T]))
    return y

target_col = [9, 7, 2]# , 4, 0, 5, 1
target_row = Rows - 1
for p in range(player_num):
    C[p][target_row*Columns + target_col[p], :, :] = 0.

policy = ut.random_initial_policy_finite(Rows, Columns, A, T, player_num)
initial_x = [np.zeros(Rows*Columns) for _ in range(player_num)]
x_init_state  = random.sample(range(Columns), player_num)
x = [[] for _ in range(player_num)]
for p in range(player_num):
    initial_x[p][x_init_state[p]] = 1.
    x[p].append(update_y(policy[:,:,:,p], initial_x[p]))


# col_div = int(Columns/player_num* A)
# for p in range(player_num):   
#     initial_x[p][col_div*p:col_div *(p+1)] = 1./col_div  
#     x[p].append(update_y(policy[:,:,:,p], initial_x[p]))

actions = [[] for _ in range(player_num)]
for p in range(player_num):
    for s in range(Rows*Columns):
        p_action,  = np.where(policy[s, s*A:(s+1)*A, 0, p] == 1.)
        # print(f' action is {policy[s, s*A:(s+1)*A, 0, p] }')
        if len(p_action) > 0:
            actions[p].append(p_action[0])
        else:
            actions[p].append(0)


# draw initial policy for player zero
axis, value_grids, _ = vs.init_grid_plot(Rows, Columns, list(np.sum(C[0][:,:,0],axis=1))+[4])
vs.draw_policies(Rows, Columns, actions[0], axis)



Iterations = 30
V_hist= [[] for _ in range(player_num)]
costs = [[] for _ in range(player_num)]

gamma = 0.9
step_size = []
# y_lists = [[y[p]]]
# y_1_list = [y_1]
# y_2_list = [y_2]
for i in range(Iterations):
    step_size.append(1/(i+1))
for i in range(Iterations):
    next_distribution = []
    next_policy = []
    
    for p in range(player_num):
        y = sum([x[p][-1] for p in range(player_num)])
        p_cost = C[p] + 1.25 * update_cost(y) # 1.25 
        costs[p].append(1*p_cost)
        V, pol_new = dp.value_iteration_finite(P, 1*p_cost, g=gamma)
        next_policy.append(pol_new)
                       
    for p in range(player_num):
        policy[:,:,:,p] = (1 - step_size[i]) * policy[:,:,:,p] + step_size[i] * next_policy[p]
        V_hist[p].append(list(ut.value_finite(P, policy[:,:,:,p], 1*p_cost, gamma)))
    # new_y1 =    
    # new_y2 = update_y(policy[:,:,:,1], initial_x_2)  
    [x[p].append(update_y(policy[:,:,:,p], initial_x[p])) for p in range(player_num)]
    # y_2_list.append(1*new_y2)
# ---------------- if plotting initial steps of frank-wolfe ------------#
plot_frank_wolfe = False # this only works if Iterations = 1
player = 1
plot_time = T - 1
if plot_frank_wolfe:
    cost_plot, val_grids, _ = vs.init_grid_plot(
        Rows, Columns, list(np.sum(costs[-1][:,:,plot_time],axis=1)))
    dp_policy = []
    original_policy = []
    for s in range(Rows*Columns):
        
        pol, = np.where(policy[s, s*A:(s+1)*A,plot_time, player] == 1)
        original_policy.append(pol[0])
        pol, = np.where(next_policy[player][s, s*A:(s+1)*A,plot_time] == 1)
        dp_policy.append(pol[0])
    vs.draw_policies(Rows, Columns,next_policy, cost_plot)
    # vs.draw_policies_interpolate(Rows, Columns,dp_policy, actions_1, cost_plot)
    plt.show()
    
        
V_hist_array = np.array(V_hist) # player, Iterations(alg), states, Timesteps+1
plt.figure()
plt.title('state values')
for s in range(Rows*Columns):
    plt.plot(V_hist_array[0,-1, s, :]) 
plt.show()

cost_array = [np.array(costs[p]) for p in range(player_num)]
for p in range(player_num):
    plt.figure()
    plt.title(f'player {p} costs')
    for s in range(Rows*Columns):
        plt.plot(np.sum(cost_array[p][:, s, :,T-1], axis=1))
    plt.show()

# p1_costs = list(np.sum(cost_array[33, :,:,T-1],axis=1))
p1_costs = list(np.sum(cost_array[0][Iterations - 1, :,:,T],axis=1))
p1_values = V_hist_array[0,Iterations - 1, :, T-1]

total_player_costs = np.zeros(Columns * Rows)
target_col = [9, 7, 2, 4, 0] #, 1, 5
target_row = Rows - 1
for t_col in target_col:
    total_player_costs[target_row*Columns + t_col] = 1.
color_map, norm, _ = vs.color_map_gen(total_player_costs) 

ax, value_grids, f = vs.init_grid_plot(Rows, Columns, total_player_costs)
plt.show()

p_inits = [np.random.randint(0, Columns - 1) for p in range(player_num)]
# p1_init = np.random.randint(0, Columns - 1) # start randomly in the top row
# p2_init = np.random.randint(0, Columns - 1) # start randomly in the top row
# vs.simulate(p1_init, p2_init, policy, p1_values, value_grids, A, 
#             Rows, Columns, P,  Time = T)

print('visualizing now')
vs.animate_traj('traj_ouput.mp4', f, p_inits, policy, total_player_costs, 
                value_grids, A, Rows, Columns, P,  Time = T)
# # v_max_1 = np.max(V_hist_array[0, -1, :])
# # v_min_1 = np.min(V_hist_array[0, -1, :])
# # # v_max_2 = np.max(V_hist_array[1, -1, :])
# # # v_min_2 = np.min(V_hist_array[1, -1, :])
# # norm_1 = mpl.colors.Normalize(vmin=v_min_1, vmax=v_max_1)
# # # norm_2 = mpl.colors.Normalize(vmin=v_min_2, vmax=v_max_2)
# # color_map = plt.get_cmap('coolwarm')    
# # f_1, axis_1 = plt.subplots(1)
# # value_grids_1 = []
# # for x_ind in range(Rows):
# #     value_grids_1.append([])
# #     for y_ind in range(Columns):
# #         R,G,B,_ = color_map(norm_1((V_hist_array[0, -1, x_ind*Columns+ y_ind])))
# #         color = [R,G,B]  
# #         value_grids_1[-1].append(plt.Rectangle((y_ind, x_ind), 1, 1, 
# #                                         fc=color, ec='xkcd:greyish blue'))
# #         axis_1.add_patch(value_grids_1[-1][-1])
# # plt.axis('scaled')
# # axis_1.xaxis.set_visible(False)  
# # axis_1.yaxis.set_visible(False)