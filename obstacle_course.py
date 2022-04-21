# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 02:12:21 2021

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
N = 20
mag_bounds = (np.ones(N), np.ones(N) * 2)

def check_region(s_x, s_y, square):
    region = 0 # 0 = calm, 1 = center, 2 = bad wind

    if s_x < 0 or s_y < 0: # negative row or column
        print(f'invalid square at {s_x}, {s_y}')
    elif s_y < square: # first row
        if s_x >= square and s_x < 3 * square:
            region = 2
    elif s_y < 2 * square:
        if s_x >= square and s_x < 2 * square:
            region = 1
        elif s_x >= 2 * square and s_x < 3 * square:
            region = 2
    
    return region
        
def kernel(s_x, s_y, s_length, s_width, a_ind):
    """ Returns neighbors in counterclock sequence, starting from the 
        rightmost neighbour.
    """
    s_ind = s_y * s_length + s_x
    ns = [s_ind + 1,  # right 0
          s_ind + s_length + 1, # upper right 1
          s_ind + s_length, # up 2
          s_ind + s_length - 1, # upper left 3
          s_ind - 1, # left 4
          s_ind - s_length - 1, # bottom left 5
          s_ind - s_length, # bottom 6
          s_ind - s_length + 1 # bottom right 7
          ]
    # no wrap
    unreachable = []
    if s_x == 0:
        unreachable += [3, 4, 5]
    elif s_x == s_width - 1:
        unreachable += [0, 1, 7]
    if s_y == 0:
        unreachable += [5, 6, 7]
    elif s_y == s_length - 1:
        unreachable += [1, 2, 3]
        
    for i in unreachable:
        ns[i] = None
        
    # bad region
    blown_ns = []
    region = check_region(s_x, s_y, s_length/3)
    reacheable_neighbors = []
    if region == 2:
        if a_ind  == 0:
            reacheable_neighbors = [7]
            # reacheable_neighbors = [i for i in range(8)]
            # blown_ns.append(s_ind)
        elif a_ind == 1:
            reacheable_neighbors = [0, 6, 7]
        elif a_ind == 2:
            reacheable_neighbors = [1,0, 7]
        elif a_ind == 3:
            blown_ns.append(s_ind)
            reacheable_neighbors = [2, 0, 7]
        elif a_ind == 4:
            blown_ns.append(s_ind)
            reacheable_neighbors = [3, 5, 0,7]
        elif a_ind == 5:
            reacheable_neighbors = [4, 5, 6]
        elif a_ind == 6:
            reacheable_neighbors = [5, 6, 7]
            # blown_ns.append(s_ind)
        elif a_ind == 7:
            reacheable_neighbors = [6,7]
            # blown_ns.append(s_ind)
        elif a_ind == 8:
            reacheable_neighbors = [7]
    # wild region (center)
    elif region == 1:
        if a_ind == 0:
            blown_ns.append(s_ind)
        reacheable_neighbors = [i for i in range(8)]
    # origin, destination, and left
    elif region == 0:
        # blown_ns.append(s_ind)
        if a_ind == 0:
            blown_ns.append(s_ind)
            # reacheable_neighbors = [i for i in range(8)]
        elif a_ind == 1:
            reacheable_neighbors = [0,1,7]
        elif a_ind == 2:
            reacheable_neighbors = [0,1,2]
        elif a_ind == 3:
            reacheable_neighbors = [1,2,3]
        elif a_ind == 4:
            reacheable_neighbors = [2,3,4]
        elif a_ind == 5:
            reacheable_neighbors = [3,4,5]
        elif a_ind == 6:
            reacheable_neighbors = [4,5,6]
        elif a_ind == 7:
            reacheable_neighbors = [5,6,7]
        elif a_ind == 8:
            reacheable_neighbors = [6,7,0]
            
    
    [blown_ns.append(ns[i]) for i in reacheable_neighbors]
    # remove all non-neighbors
    blown_ns = list(filter((None).__ne__, blown_ns))       
    return blown_ns

def wind_bound_gen(length, wind_mag = 1):
    # mean is optimal through the middle
    square = int(length / 3)
    angles_max = np.zeros((length, length))
    angles_min = np.zeros((length, length))
    wind_min = np.zeros((length, length))
    wind_max = np.zeros((length, length))
    origin = (slice(0, square), slice(0, square))
    destination = (slice(2*square, 3*square), slice(2*square, 3*square))
    l_box = [(slice(square, 2 * square),slice(0, square)),
             (slice(2 * square, 3 * square), slice(0, square)),
             (slice(2 * square, 3 * square), slice(square, 2 * square))]

    bad_box = [(slice(0,square), slice(square, 3*square)),
               (slice(square, 2*square), slice(2*square, 3*square))]
    center = (slice(square, 2 * square), slice(square, 2 * square))
    # beginning has uniform wind
    angles_min[origin] = 0
    angles_max[origin] = 2 * np.pi
    wind_min[origin] = 0
    wind_max[origin] = 1
    for left in l_box:
        # left area has uniform box
        angles_min[left] = 0 
        angles_max[left] = 2 * np.pi
        wind_min[left] = 0
        wind_max[left] = 1
    # center has highly variable wind
    angles_min[center] = 0 
    angles_max[center] = 2 * np.pi
    wind_min[center] = 0
    wind_max[center] = 10
    # right side has uniformly south west wind
    for right in bad_box:
        # right area has only  goes to the upper right
        angles_min[right] = 3/2*np.pi # np.pi / 4 + 
        angles_max[right] = np.pi / 4 + 3/2*np.pi
        wind_min[right] = 0
        wind_max[right] = 5
    # destination has uniform wind
    angles_min[destination] = 0
    angles_max[destination] = 2 * np.pi
    wind_min[destination] = 0
    wind_max[destination] = 1
    
    return wind_min, wind_max, angles_min, angles_max
# --------- generate max/min bounds on the polytopic MDP ------------
S = length * width
A = 9
P, C = wind.polytope_mdp_gen(S, A, length, kernel)

v_min, v_max, pi_opt, pi_rbt = dp.value_iteration_polytopes(P, C, gamma = 0.9)
    
value_grid_min = v_min.reshape((width, length))
value_list_min = value_grid_min.flatten()
cost_plot, val_grids, _ = vs.init_grid_plot(width, length, value_list_min)
wind.draw_policies(width, length, pi_opt, cost_plot)
plt.show()

value_grid_max = v_max.reshape((width, length))
value_list_max = value_grid_max.flatten()
cost_plot, val_grids, _ = vs.init_grid_plot(width, length, value_list_max)
wind.draw_policies(width, length, pi_rbt, cost_plot)
plt.show()

# --------- generate a random MDP based on wind sampling --------- 
# wind_min, wind_max, angles_min, angles_max = wind_bound_gen(length)    
# P, R = wind.bound_mdp_gen(angles_min, angles_max, wind_min, wind_max)   
# values, policy = dp.value_iteration(P, R, minimize=True, g=0.5)

# S = length * width
# _, A = R.shape 
# print(f'values shape {values.shape}')
# value_grid = values.reshape((width, length))
# print(f'value grid shape {value_grid.shape}')


# r_max = np.max(R, axis=1)
# r_grid = r_max.reshape((width, length))
# plt.figure()
# plt.imshow(r_grid, interpolation='nearest')
# # plt.imshow(value_grid.T, interpolation='nearest')
# plt.colorbar()
# plt.show() 

# plt.figure()
# # plt.imshow(r_grid.T, interpolation='nearest')
# plt.imshow(value_grid, interpolation='nearest')
# plt.colorbar()
# plt.show()


# value_list = value_grid.flatten()
# cost_plot, val_grids, _ = vs.init_grid_plot(width, length, value_list)

# original_policy = []
# for s in range(S):
#     pol, = np.argwhere(policy[s, s*A:(s+1)*A] == 1) 
#     print(f' state {s}, policy {pol[0]}')
#     original_policy.append(pol[0])
    
    
# wind.draw_policies(width, length, original_policy, cost_plot)
# plt.show()