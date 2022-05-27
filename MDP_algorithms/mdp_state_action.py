# -*- coding: utf-8 -*-

import numpy as np
import util as ut
import random


def pick_up_delivery_dynamics(Rows, Columns, rate):
    P_0 = ut.nonErgodicMDP(Rows, Columns, p=0.98,with_stay=True)
    S_half, S_halfA = P_0.shape
    A = int(S_halfA/S_half)
    S = S_half*2
    P = np.zeros((S, S*A))
    P[:S_half, :S_halfA] = 1*P_0
    P[S_half:, S_halfA:] = 1*P_0
    # transition from pickup to delivery
    last_row_1 = range(S_half - Columns, S_half)
    for s in last_row_1:
        # sa = slice(s*A,(s+1)*A)
        orig_probability = P[s, :]*1
        P[s, :] = rate * orig_probability
        P[s + S_half, :] += (1 - rate) * orig_probability
        
    first_row_2 = range(S_half,  S_half+Columns)
    for s in first_row_2:
        P[s - S_half, :] += P[s, :]*1
        P[s, :] = 0
        
    return P

def pick_up_multi_delivery_dynamics(Rows, Columns, rate, delivery_dist, drop_off):
    P_0 = ut.nonErgodicMDP(Rows, Columns, p=0.98,with_stay=True)
    modes = len(delivery_dist)
    S_sec, S_secA = P_0.shape
    # A = int(S_secA/S_sec)
    # S = S_sec*2
    P = np.kron(np.eye(modes+1), P_0)
    # print(f'transition_sizes are {P.shape}')
    # print(f'transition current summation {np.ones(S_sec*(modes+1)).T.dot(P)}')
    # transition from pickup to delivery
    last_row_pick_up = range(S_sec - Columns,  S_sec)
    for s in last_row_pick_up:
        # sa = slice(s*A,(s+1)*A)
        stay_probability = rate * P[s, :]
        P[s, :] = stay_probability
        drop_off_probability = (1 - rate) * P[s, :]
        for mode_int in range(modes):
            P[s + S_sec * (mode_int + 1), :] += delivery_dist[mode_int] * drop_off_probability
    
    # transition back to pick up from each mode
    for mode in range(modes):
        drop_off_row = range(S_sec*(mode+1), S_sec*(mode+1)+Columns)
        for s in drop_off_row:
            P[s % S_sec, :] += P[s, :]*1
            P[s, :] = 0
        
    return P



def pick_up_delivery_cost(Rows, Columns, A, T, pick_up_state, deliveries, p_num, 
                         minimize=True):
    targ_rew = 0 if minimize else 1.
    S_sec = Rows*Columns
    S = S_sec * (len(deliveries) + 1)
    if minimize:
        C = np.ones((S, A, T+1))
    else:
        C = np.zeros((S, A, T+1))
    # cost for agents in pick up mode
    for p in range(p_num):
        C[pick_up_state, :, :] = targ_rew
    # cost for agents delivery mode
    for delivery_state in deliveries:
        C[delivery_state, :, :] = targ_rew    
    return C

def set_up_cost(Rows, Columns, A, T, target_col, target_row,  p_num, 
                minimize=True):
    targ_rew = 0 if minimize else 1.
    S = Rows * Columns
    if minimize:
        C = [np.ones((S, A, T+1)) for _ in range(p_num)]
    else:
        C = [np.zeros((S, A, T+1)) for _ in range(p_num)]
    for p in range(p_num):
        C[p][target_row*Columns + target_col[p], :, :] = targ_rew
    return C
    
def pol2dist(policy, x, P, T): 
    # policy is a 3D array for player
    # x is player p's initial state distribution at t = 0
    # returns player P's final distribution
    S, SA, Tp1 = policy.shape
    T = Tp1 - 1
    x_arr = np.zeros((len(x), T+1))
    x_arr[:, 0] = x
        
    markov_chains = np.einsum('ij, kjl->ikl', P, policy)
    # print(f'x_shape {x_arr.shape}')
    # print(f' markov chain shape {markov_chains[:, :, 0].shape}')
    for t in range(T):
        # x is the time state_density
        # print(f'x shape is {markov_chains[:, :, t].dot(x_arr[:, t]).shape}')
        x_arr[:, t+1] = markov_chains[:, :, t].dot(x_arr[:, t]) 
    y = np.einsum('sat, st->at', policy, x_arr)

    return y



def state_congestion(Rows, Columns, modes, A, T, y):
    c_cost = np.zeros((modes*Rows*Columns, A, T+1))
    
    for x_ind in range(Columns):
        for y_ind in range(Rows):
            unwrap_ind = (y_ind * Columns + x_ind)
            for t in range(T+1):
                delivery_ind = unwrap_ind + Rows*Columns
                density = np.sum(y[unwrap_ind*A:(unwrap_ind+1)*A, t]) + \
                          np.sum(y[delivery_ind*A:(delivery_ind+1)*A, t])
                congestion = 5* np.exp(5 * (density - 1))
                c_cost[unwrap_ind, :, t] += congestion
                c_cost[delivery_ind, :, t] += congestion
    return c_cost

def policy_list(policies, P, T, p_num, Columns):
    # policy = ut.random_initial_policy_finite(Rows, Columns, A, T+1, p_num)
    S, SA = P[0].shape
    initial_x = [np.zeros(S) for _ in range(p_num)]
    x_init_state  = random.sample(range(Columns), p_num)
    x = [[] for _ in range(p_num)]
    for p in range(p_num):
        initial_x[p][x_init_state[p]] = 1.
        x[p].append(pol2dist(policies[:,:,:,p], initial_x[p], P[p], T+1))
    return x, initial_x


def execute_policy(initial_x, P, pols, T, targets):
    S, SA = P.shape 
    S_half = int(S/2)
    A = int(SA/S)
    # ind of first position the players are in
    trajs = [[np.where(x == 1)[0][0]] for x in initial_x] 
    # print(f' traj is {trajs}')
    
    _, _, T, p_num = pols.shape
    flat_pols = np.sum(pols, axis=0)
    collisions = {p: [] for p in range(p_num)}
    drop_off_counter = [[] for _ in range(p_num)]
    for t in range(T):
        for p_ind in range(p_num):
            cur_s = trajs[p_ind][-1]
            next_a = np.random.choice(
                np.arange(0,A),p=flat_pols[cur_s*A:(cur_s+1)*A, t, p_ind])
            next_s = np.random.choice(np.arange(0,S), p=P[:, cur_s*A+next_a])
            # print(f' next state {next_s}')
            trajs[p_ind].append(1*next_s)
            # check pick up time
            if cur_s >= S_half and next_s < S_half:
                drop_off_counter[p_ind].append(1*t)
        # collision detection
        cur_pos = [trajs[p_ind][-1] for p_ind in range(p_num)]
        for p in range(p_num):  
            collisions[p].append(cur_pos.count(cur_pos[p])-1)
        # collisions.append(len(cur_pos) - len(set(cur_pos)))
        # if len(cur_pos) - len(set(cur_pos)) > 0:
        #     print(f'time {t} current positions {cur_pos}')
        # collisions += 
        drop_off_time = []
        for i in range(p_num):
            drop_offs = drop_off_counter[i]
            drop_off_time.append([
                drop_offs[j+1] - drop_offs[j] for j in range(len(drop_offs)-1)])
    return collisions, drop_off_time
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
