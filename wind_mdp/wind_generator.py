# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 08:34:07 2021

@author: Sarah Li
"""
import numpy as np
import scipy.stats as ss

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

def wind_mag_gen(mean, var, N=100):
    return ss.norm.rvs(mean, var, size=N)


actions = action_generator()
print(actions)
w = wind_mag_gen(10, 1)
angles = von_mises_angle_gen(1, 10)