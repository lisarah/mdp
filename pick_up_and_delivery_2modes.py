# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:24:02 2022

Pick up drop off 2 modes

@author: Sarah Li
"""
import numpy as np
import visualization as vs
import util as ut
import matplotlib.pyplot as plt
import pandas as pd
import MDP_algorithms.mdp_state_action as st
import MDP_algorithms.value_iteration as dp
import random, time
import seaborn as sns
import matplotlib as mpl
import time


""" Format matplotlib output to be latex compatible with gigantic fonts."""
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 15})
mpl.rc('legend', fontsize='small')


""" Pick up state is along the bottom. Drop off state is along the top."""
Columns = 10
Rows = 5
T = 60
player_num = 3
# set up space 
pick_up_col = [8, 7, 2] #, 0, 4, 0, 5, 1
pick_up_row = Rows - 1
pick_ups = [pick_up_row*Columns + col for col in pick_up_col]
drop_off_col = [4, 5, 8]
drop_off_row = 0
drop_offs = [drop_off_row*Columns + col for col in drop_off_col]
# Poisson for package inter-arrival time with lambda = 0.5
rate = np.exp(-0.2 * 1) 

alpha = [0.5, 1., 1.25]
P = [st.pick_up_delivery_dynamics(Rows, Columns, rate, [drop_offs[p]], 
                                  [pick_ups[p]], p=1.) 
     for p in range(player_num)]
P_tensor = [st.transition_mat_to_tensor(p_mat) for p_mat in P]

S, SA = P[0].shape
A = int(SA/S)
mode_num = 2


C = [st.pick_up_delivery_cost(Rows, Columns, A, T, pick_ups[p], 
                                    [drop_offs[p]], p, minimize=False) 
     for p in range(player_num)]

pols = ut.random_initial_policy_finite(S, A, T+1, player_num)
x, initial_x = st.policy_list(pols, P, T, player_num, Columns)
initial_x  = st.start_at_pick_up(player_num, drop_offs, S)

# run frank-wolfe
Iterations = 20 # number of Frank wolf iterations
V_hist= [[] for _ in range(player_num)]
costs = [[] for _ in range(player_num)]

gamma = 0.99
steps = [1/(i+1) for i in range(Iterations)]
begin_time = time.time()
for i in range(Iterations):
    print(f'\r on iteration {i}', end='   ')
    next_distribution = []
    y = sum([alpha[p] * x[p][-1] for p in range(player_num)])
    for p in range(player_num):        
        p_cost = C[p] - alpha[p] * st.state_congestion_faster(
            Rows, Columns, mode_num, A, T, y) - 1e-2 * np.reshape(x[p][-1], (S, A, T+1)) # 1.25 
        costs[p].append(1*p_cost)
        V, pol_new = dp.value_iteration_tensor(P_tensor[p], 1*p_cost, T, minimize=False)
        V_hist[p].append(V)
        pols[:,:,:,p] = (1-steps[i])*pols[:,:,:,p] + steps[i]*pol_new
        x[p].append(st.pol2dist(pols[:,:,:,p],initial_x[p], P[p], T))     
print(f'total time is {time.time() - begin_time}')   
    
# # -------------  plot results  ---------------
entries = 100
res = {'Collisions': [], 't': [], 'player': []}
wait_times = {'Average wait' : [], 'Max Wait': [],
        'Player': [], 'Collisions': []}

for ent in range(entries):
    collisions, min_time = st.execute_policy(initial_x, P, pols, T, pick_ups)
    # print(f'number of collisions is {collisions}')
    # print(f'minimum time to target is {min_time}')
    for p in range(player_num):
        if len(min_time[p]) == 0:
            average_wait = 0
            max_wait = 0
        else:
            average_wait = sum(min_time[p])/len(min_time[p]) 
            max_wait = max(min_time[p])
        wait_times['Average wait'].append(average_wait)
        wait_times['Max Wait'].append(max_wait)
        wait_times['Player'].append(p)
        wait_times['Collisions'].append(sum(collisions[p]))
    # for t in range(T):
    #     res['Collisions'].append(collisions[t])
    #     res['t'].append(t)
trials = pd.DataFrame.from_dict(res)

sns.set_style("darkgrid")
columns = ['Collisions'] # , 'Time'
fig, axs = plt.subplots(figsize=(5,3), nrows=len(columns))
# axs = axs.flatten()
k = 0
for column in columns:
    # print(f'visualizing {column}')
    sns.lineplot(ax=axs, data=trials, x='t', y=column)
    axs.set(ylabel=column)
    k += 1
plt.show()

columns = ['Average wait', 'Max Wait', 'Collisions'] # , 'Time'
fig, axs = plt.subplots(figsize=(10,5), ncols=len(columns))
axs = axs.flatten()
k = 0
for column in columns:
    # print(f'visualizing {column}')
    sns.lineplot(ax=axs[k], data=wait_times, x='Player', y=column)
    axs[k].set(ylabel=column)
    k += 1
plt.show()

V_hist_array = np.array(V_hist) # player, Iterations(alg), states, Timesteps+1
# plot the value history as a function of states
plt.figure()
# plt.title('target pick up  state values')
for p in range(player_num):
    plt.plot(V_hist_array[p, 2:, pick_ups[p], 0], label=f'player {p}')
plt.legend()
plt.show()    

plt.figure()
plt.title('State values')
for s in range(Rows*Columns):
    plt.plot(V_hist_array[0,-1, s, :]) 
plt.show()



# cost_array = [np.array(costs[p]) for p in range(player_num)]
# for p in range(player_num):
#     plt.figure()
#     plt.title(f'player {p} costs')
#     for s in range(Rows*Columns):
#         plt.plot(np.sum(cost_array[p][:, s, :, T-1], axis=1))
#     plt.show()

# # p1_costs = list(np.sum(cost_array[33, :,:,T-1],axis=1))
# p1_costs = list(np.sum(cost_array[0][Iterations - 1, :,:,T],axis=1))
# p1_values = V_hist_array[0,Iterations - 1, :, T-1]

total_player_costs = np.zeros(Columns * Rows)

for tar in pick_ups:
    total_player_costs[pick_ups] = 1.
color_map, norm, _ = vs.color_map_gen(total_player_costs) 

ax, value_grids, f = vs.init_grid_plot(Rows, Columns, total_player_costs)
plt.show()

p_inits = list(random.sample(range(0, Columns), player_num))
 
print('visualizing now')
vs.animate_traj(f'traj_ouput_{int(time.time())}.mp4', f, p_inits, pols, 
                total_player_costs, value_grids, A, Rows, Columns, P, Time=T)
    
    










    