# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 08:34:07 2021

@author: Sarah Li
"""
import numpy as np
import scipy.stats as ss


def action_angles():
    """ Get all action angles except for hte don't move action. 
    """
    return  np.linspace(0, np.pi * 2, 9)

def action_generator(action_mag = 1):
    """ Return the 2D action magnitudes for the 9 possible action directions. 
    """
    angles = action_angles()

    actions = [np.array([0,0])]
    # cartesian system
    for angle in angles[:-1]:
        actions.append(np.array([np.sin(angle), np.cos(angle)]))

    return actions

def policy_arrow_gen(policy_ind):
    lookup = {
        0: (0, 0), # no action
        1: (1, 0), # right,
        2: (1, 1), # upper_right
        3: (0, 1), # top, 
        4: (-1, 1), # top left
        5: (-1, 0), # left
        6: (-1, -1), # bottom left
        7: (0, -1), # bottom
        8: (1, -1) # bottom right
        }
    return lookup[policy_ind]

def draw_policies(s_width, s_length, policy, axis):
    length = 0.3

    color = 'xkcd:coral'
    # color = 'xkcd:pale yellow'
    # color = 'xkcd:lemon'
    for x_ind in range(s_length):
        for y_ind in range(s_width):   
            # print(f'grid {x_ind}, {y_ind}')
            dx, dy = policy_arrow_gen(policy[y_ind*s_length + x_ind])
            axis.arrow(x_ind + 0.5, y_ind+0.5, 
                       dx * length, dy * length, 
                       head_width=0.3, head_length=0.15, 
                       fc=color, ec=color)
    
            
def von_mises_angle_gen(mean, kappa, N=100):
    return ss.vonmises.rvs(kappa, size=N) + mean

def wind_gen(mean, var, N=100):
    return ss.norm.rvs(mean, var, size=N)


def distance(x_1, x_2, y_1, y_2, length, width): # 1 norm distance
    return abs(x_1 - y_1)/length + abs(x_2 - y_2)/width

def polar2cart(magnitude, angle):
    return magnitude *np.array([np.cos(angle), np.sin(angle)])

def velocity_magnitude(magnitude, angle): # 1 norm velocity
    vel = polar2cart(magnitude, angle)
    return abs(vel[0]) + abs(vel[1])

def neighbors(s_x, s_y, s_length, s_width):
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
    
    if s_x  == 0: # all lefts moved
        ns[4] = s_ind + s_length - 1
        ns[3] = s_ind + s_length - 1
        ns[5] = s_ind + s_length - 1
    elif s_x == s_length - 1: # all rights moved
        ns[0] = s_ind - s_length + 1
        ns[1] = s_ind - s_length + 1
        ns[7] = s_ind - s_length + 1
   
    if s_y == 0: # all bottoms moved
        ns[5] = s_ind + (s_length) * (s_width-1)
        ns[6] = s_ind + (s_length) * (s_width-1)
        ns[7] = s_ind + (s_length) * (s_width-1)
    elif s_y == s_width - 1: # all tops removed
        ns[1] = s_ind - (s_length) * (s_width-1)
        ns[2] = s_ind - (s_length) * (s_width-1)
        ns[3] = s_ind - (s_length) * (s_width-1)
        
    if s_x == 0 and s_y == 0: # bottom left
        ns[5] = s_length * s_width - 1
    elif s_x == 0 and s_y == s_width - 1: # top left
        ns[3] = s_length - 1
    elif s_x == s_length - 1 and s_y == 0: # bottom right
        ns[7] = s_length * (s_width - 1)
    elif s_x == s_length - 1 and s_y == s_width - 1: # top right
        ns[1] = 0
    
    # the zeroth neighbor is the base itself
    ns.insert(0, s_ind)
    
    # if 100 in ns:
    #     print(f'neighbours of {s_x}, {s_y} is wrong at ns:'
    #           f'{ns.index(100)}, s_ind = {s_ind}')
    return ns

    
def mdp_gen(s_length, s_width, action_mag, sample_num=100):
    S = s_length * s_width
    A = 9
    actions = action_generator(action_mag)
    P = np.zeros((S,S,A)) # destination, origin, action
    R = np.zeros((S,A))
    angle_mean = 0.122173 # 7 degrees
    angle_kappa = 1/400
    wind_mag_mean = 0.54 # m/s
    wind_variance = 0.11 # m/s 
    
    # angle_mean = 0 # 7 degrees
    # angle_kappa = 0
    wind_mag_mean = 0 # m/s
    wind_variance = 0 # m/s 
    
    target = (s_length-1, s_width-1)
    print(f'target is {target}')
    
    angle_bins = action_angles()
    
    for s_x in range(s_length):
        for s_y in range(s_width):
            s_ind = s_y * s_length + s_x
            dist = distance(s_x, s_y, target[0], target[1], s_length, s_width)
            
            for a_ind in range(A):
                N = 100
                wind_angles = von_mises_angle_gen(angle_mean, angle_kappa, N)
                wind_mags = wind_gen(wind_mag_mean, wind_variance, N)
                # print(f'wind_angles {wind_angles}')
                # print(f'wind_mags {wind_mags}')
                wind_vectors =  [polar2cart(wind_mags[i], wind_angles[i])
                    for i in range(N)]
                
                # for wind in wind_vectors:
                #     # print(f'wind_vectors {wind_vectors}')
                #     if np.linalg.norm(wind, 2) != 0:
                #         print(f'non zero wind vector {wind}')
                    
                net_vectors = [wind_vectors[i]+actions[a_ind] for i in range(N)]
                avg_magnitude = np.mean([
                    velocity_magnitude(net_vectors[i], net_vectors[i]) 
                    for i in range(N)])
                # rewards
                if avg_magnitude != 0:
                    R[s_ind, a_ind] = -dist / avg_magnitude
                    # print(f'distance is {np.mean(net_vectors)}')
                else:
                    R[s_ind, a_ind] = -99
                    
                # transitions
                net_angles = [np.angle(z) for z in net_vectors]
                large_wind = []
                for w_ind in range(len(wind_mags)):
                    if wind_mags[w_ind] > 1e-3: # wind causes movement
                        large_wind.append(w_ind)
                        
                net_moving = [net_angles[i] for i in large_wind] 
                
                freq, _ = np.histogram(net_moving, bins = angle_bins)
                f_ind = -1
                for neighbor in neighbors(s_x, s_y, s_length, s_width):
                    if f_ind == -1:
                        P[neighbor, s_ind, a_ind] = (
                            len(net_vectors) - len(large_wind)) / N
                    else:
                        P[neighbor, s_ind, a_ind] = freq[f_ind] / N
                    f_ind +=1

    return P, R
    
    

# actions = action_generator()
# print(actions)
# w = wind_mag_gen(10, 1)
# angles = von_mises_angle_gen(1, 10)        
            
# P, R = mdp_gen(10, 10, 0.5)   
# plt.imshow(R, interpolation='nearest')
# plt.show() 
            
            
            
            
            
            
    