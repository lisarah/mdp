# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt



def value_iteration(P,c, minimize = True, g = 1.):
    plt.close('all')
    optimize = np.min if minimize is True else np.max
    opt_arg = np.argmin if minimize is True else np.argmax
    # print(f' optimize is {optimize}')
    # print(f' opt_arg is {opt_arg}')
    S, A, T = c.shape
    # T = T_over - 1
    pik = np.zeros((S, T));
    newpi = np.zeros((S,S*A, T));
    Vk = np.zeros((S, T));
    BO = 1*c
    # Vk[:, T] = optimize(BO[:,:,T_over], axis=1)
    for t in range(T):
        t_ind = T - t - 1 # T - 1 , T-2, T-3, ... 0
        if t_ind  <  T - 1:
            BO[:,:,t_ind] +=  g*np.reshape(Vk[:,t_ind+1].dot(P), (S,A))
        Vk[:,t_ind] = optimize(BO[:,:,t_ind], axis=1)
        pik[:,t_ind] = opt_arg(BO[:,:,t_ind], axis=1)
        
    for s in range(S):
        for t in range(T):
            newpi[s, int(s*A + pik[s,t]), t] = 1.

    return Vk, newpi