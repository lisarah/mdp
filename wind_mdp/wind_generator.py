# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 08:34:07 2021

@author: Sarah Li
"""
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def action_generator(action_mag = 1):
    """ Return the 2D action magnitudes for the 9 possible action directions. 
    """
    angles = np.linspace(0, np.pi * 2, 9)

    actions = [np.array([0,0])]
    for angle in angles[:-1]:
        actions.append(np.array([np.sin(angle), np.cos(angle)]))
        
    return actions


def von_mises_angle_gen(mean, kappa, N=100):
    return ss.vonmises.rvs(kappa, size=N) + mean

def wind_gen(mean, var, N=100):
    return ss.norm.rvs(mean, var, size=N)

def distance(x_1, x_2, y_1, y_2, length, width): # 1 norm distance
    return abs(x_1 - y_1)/length + abs(x_2 - y_2)/width

def velocity_magnitude(magnitude, angle): # 1 norm velocity
    vel = magnitude *np.array([np.cos(angle), np.sin(angle)])
    return abs(vel[0]) + abs(vel[1])

def neighbors(s_x, s_y, s_length, s_width):
    """ Returns neighbors in counterclock sequence, starting from the 
        rightmost neighbour.
    """
    s_ind = s_y * s_length + s_x
    square_length = s_length * s_width
    ns = [s_ind + 1,  # right
          s_ind - s_length + 1, # upper right
          s_ind - s_length, # up
          s_ind - s_length - 1, # upper left
          s_ind - 1, # left
          s_ind + s_length - 1, # bottom left,
          s_ind + s_length, # bottom
          s_ind + s_length + 1 # bottom right
          ]
    for ind in range(len(ns)):
        if ns[ind] < 0: # wrap back:
            ns[ind] += square_length
        elif ns[ind] >= square_length:
            ns[ind] += -square_length
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
    
    target = (s_length-1, s_width-1)
    print(f'target is {target}')
    origin = int(s_length/3)
    angle_bins = np.linspace(0, np.pi * 2, 9)
    
    for s_x in range(s_length):
        for s_y in range(s_width):
            s_ind = s_y * s_length + s_x
            dist = distance(s_x, s_y, target[0], target[1], s_length, s_width)
            
            for a_ind in range(A):
                N = 100
                wind_angles = von_mises_angle_gen(angle_mean, angle_kappa, N)
                wind_vectors = wind_gen(wind_mag_mean, wind_variance, N)
               
                net_vectors = [
                    velocity_magnitude(wind_vectors[i] + actions[a_ind][0], 
                                       wind_angles[i] + actions[a_ind][1])
                    for i in range(N)]
                R[s_ind, a_ind] = -dist / np.mean(net_vectors)
                # print(f'average wind {np.mean(net_vectors)}')
                net_angles = [np.angle(z) for z in net_vectors]
                freq, _ = np.histogram(net_angles, bins = angle_bins)
                f_ind = 0
                for neighbor in neighbors(s_x, s_y, s_length, s_width):
                    P[neighbor, s_ind, a_ind] = freq[f_ind] / N

    return P, R
    
    

# actions = action_generator()
# print(actions)
# w = wind_mag_gen(10, 1)
# angles = von_mises_angle_gen(1, 10)        
            
# P, R = mdp_gen(10, 10, 0.5)   
# plt.imshow(R, interpolation='nearest')
# plt.show() 
            
            
            
            
            
            
    