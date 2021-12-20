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
        actions.append(np.array([np.cos(angle), np.sin(angle)]))

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
    color = 'xkcd:coral'
    # color = 'xkcd:pale yellow'
    # color = 'xkcd:lemon'
    for x_ind in range(s_length):
        for y_ind in range(s_width):   
            # print(f'grid {x_ind}, {y_ind}')
            length =  0.3 
            dx, dy = policy_arrow_gen( policy[y_ind*s_length + x_ind]) #y_ind*s_length + x_ind
            axis.arrow(x_ind + 0.5, -y_ind+0.5, 
                       dx * length, -dy * length, 
                       head_width=0.3, head_length=0.15, 
                       fc=color, ec=color)
    
            
def von_mises_angle_gen(mean, kappa, N):
    # von mises returns a random variable vector between -pi and pi
    return ss.vonmises.rvs(kappa, size=N) + mean

def bounded_angle_gen(mean, bounds, N):
    return np.random.uniform(bounds[0], bounds[1], N)

def wind_gen(mean, var, N, bounds):
    if bounds is None:
        return ss.norm.rvs(mean, var, size=N)
    else:
        return np.random.uniform(bounds[0], bounds[1], N)


def distance(x_1, x_2, y_1, y_2, length, width): # 1 norm distance
    vec_1 = np.array([x_1, x_2])
    vec_2 = np.array([y_1, y_2])
    return np.linalg.norm(vec_1 - vec_2, 2)

def polar2cart(magnitude, angle):
    return magnitude *np.array([np.cos(angle), np.sin(angle)])

def arr2polar(vec):
    cp = vec[0] + 1j * vec[1]
    angle = np.angle(cp)
    if angle < 0:
        angle += 2 * np.pi
    return angle

def velocity_magnitude(vel): # 1 norm velocity
    # vel = polar2cart(magnitude, angle)
    return abs(vel[0]) + abs(vel[1])

def neighbors(s_x, s_y, s_length, s_width, wrap=True):
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
    if wrap:
        if s_x  == 0: # all lefts moved
            ns[3] = s_ind + 2 * s_length - 1 # upper left
            ns[4] = s_ind + s_length - 1 # left
            ns[5] = s_ind  - 1 # bottom left
        elif s_x == s_length - 1: # all rights moved
            ns[0] = s_ind - s_length + 1 # right
            ns[1] = s_ind + 1 # upper right
            ns[7] = s_ind - 2 * s_length + 1 # bottom right
       
        if s_y == 0: # all bottoms moved
            ns[5] = s_ind - 1 + (s_length) * (s_width-1) # bottom left
            ns[6] = s_ind + (s_length) * (s_width-1) # botoom
            ns[7] = s_ind + 1 + (s_length) * (s_width-1) # bottom right
        elif s_y == s_width - 1: # all tops removed
            ns[1] = s_ind - (s_length) * (s_width-1) + 1 # top right
            ns[2] = s_ind - (s_length) * (s_width-1) # top 
            ns[3] = s_ind - (s_length) * (s_width-1) - 1 # top left
            
        if s_x == 0 and s_y == 0: # bottom left
            ns[5] = s_length * s_width - 1 
        elif s_x == 0 and s_y == s_width - 1: # top left
            ns[3] = s_length - 1
        elif s_x == s_length - 1 and s_y == 0: # bottom right
            ns[7] = s_length * (s_width - 1)
        elif s_x == s_length - 1 and s_y == s_width - 1: # top right
            ns[1] = 0
    else: # not wrap
        if s_x == 0:
            ns[3] = None
            ns[4] = None
            ns[5] = None
        elif s_x == s_width - 1:
            ns[0] = None
            ns[1] = None
            ns[7] = None
        if s_y == 0:
            ns[5] = None
            ns[6] = None
            ns[7] = None
        elif s_y == s_length - 1:
            ns[1] = None
            ns[2] = None
            ns[3] = None
            
    # the zeroth neighbor is the base itself
    ns.insert(0, s_ind)
    
    # if 100 in ns:
    #     print(f'neighbours of {s_x}, {s_y} is wrong at ns:'
    #           f'{ns.index(100)}, s_ind = {s_ind}')
    return ns

def bound_mdp_gen(angle_min, angle_max, wind_min, wind_max):
    # generate P/R from these min/maxes    
    s_len, _ = angle_min.shape
    S = s_len * s_len
    A = 9
    action_mag = 2
    actions = action_generator(action_mag)
    P = np.zeros((S,S,A)) # destination, origin, action
    R = np.zeros((S,A))
    
    target = (s_len-1, s_len-1)

    print(f'target is {target}')
    
    angle_bins =  np.array([np.pi/8*(2*i-1) for i in range(9)])
    N = 100
    
    for s_x in range(s_len):
        for s_y in range(s_len):
            s_ind = s_y * s_len + s_x
            print(f'\r on element {s_x}, {s_y}    ', end='')
            dist = distance(s_x, s_y, target[0], target[1], s_len, s_len)
            ang_bound = (angle_min[s_x, s_y], angle_max[s_x, s_y])
            wind_angles = bounded_angle_gen(None, ang_bound, N)
            mag_bound = (wind_min[s_x, s_y], wind_max[s_x, s_y])
            wind_mags = wind_gen(None, None, N, mag_bound)
            wind_vecs =  [polar2cart(wind_mags[i], wind_angles[i])
                          for i in range(N)]
            for a_ind in range(A):  
                net_vectors = [wind_vecs[i]+actions[a_ind] for i in range(N)]
                # if s_x ==0 and s_y == 0 and a_ind == 1:
                #     print(f' {s_x}, {s_y}: net_vecs: {np.round(net_vectors, 2)}')
                avg_magnitude = np.mean([velocity_magnitude(net_vectors[i]) 
                    for i in range(N)])
                
                # rewards
                R[s_ind, a_ind] = - dist
                # if avg_magnitude != 0:
                #     R[s_ind, a_ind] = - dist/ avg_magnitude
                # elif s_x == target[0] and s_y == target[1]:
                #     # print(f'at target, action {a_ind} has positive reward')
                #     R[s_ind, a_ind] = 0                    
                # else:
                #     print(f' at {s_x}, {s_y}, action {a_ind} has no magnitude ')
                #     R[s_ind, a_ind] = - 999999
                    
                # transitions
                net_angles = [] # for angles with large enough wind
                for net_vec in net_vectors:
                    if velocity_magnitude(net_vec) > 1e-2: # wind causes movement
                        angle = arr2polar(net_vec)
                        if angle > max(angle_bins):
                            angle  = 2 * np.pi - angle
                        net_angles.append(angle)
                if s_x ==0 and s_y == 0 and a_ind == 1:
                    print(f' {s_x}, {s_y}: net_vecs: {np.round(net_angles, 2)}')
                    
                freq, _ = np.histogram(net_angles, bins = angle_bins)
                if s_x ==0 and s_y == 0 and a_ind == 1:
                    print(f' {s_x}, {s_y}: frequencies: {freq}')
                    print(f' {s_x}, {s_y}: neighbors: { neighbors(s_x, s_y, s_len, s_len,wrap=False)}')
                f_ind = -1
                for neighbor in neighbors(s_x, s_y, s_len, s_len,wrap=False):
                    # print(f'[{s_x}], [{s_y}] {neighbors(s_x, s_y, s_len, s_len)}')
                    if neighbor is not None:
                        if f_ind == -1:
                            P[neighbor, s_ind, a_ind] = (
                                len(net_vectors) - len(net_angles)) / N                        
                        else:
                            P[neighbor, s_ind, a_ind] = freq[f_ind] / N
                    else:
                        # if we hit a wall, go back to current state
                        P[s_ind, s_ind, a_ind] += freq[f_ind] / N 
                    f_ind +=1

    return P, R
    
def mdp_gen(s_length, s_width, action_mag, sample_num=100, mag_bound=None, 
            ang_bound=None):
    S = s_length * s_width
    A = 9
    actions = action_generator(action_mag)
    P = np.zeros((S,S,A)) # destination, origin, action
    R = np.zeros((S,A))
    angle_mean = 0.122173 # 7 degrees
    angle_kappa = 1/400
    wind_mag_mean = 0.54 # m/s
    wind_variance = 0.11 # m/s 

    # wind_mag_mean = 0 # m/s
    # wind_variance = 0 # m/s 
    
    target = (int(s_length/2), int(s_width/2))

    print(f'target is {target}')
    
    angle_bins = action_angles()
    N = sample_num
    
    for s_x in range(s_length):
        for s_y in range(s_width):
            s_ind = s_y * s_length + s_x
            dist = distance(s_x, s_y, target[0], target[1], s_length, s_width)
            
            for a_ind in range(A):                
                if ang_bound is None:
                    wind_angles = von_mises_angle_gen(angle_mean, angle_kappa,
                                                      N)
                else:
                    wind_angles = bounded_angle_gen(angle_mean, ang_bound, N)
                wind_mags = wind_gen(wind_mag_mean, wind_variance, N, mag_bound)

                wind_vectors =  [polar2cart(wind_mags[i], wind_angles[i])
                    for i in range(N)]
                    
                net_vectors = [wind_vectors[i]+actions[a_ind] for i in range(N)]
                avg_magnitude = np.mean([velocity_magnitude(net_vectors[i]) 
                    for i in range(N)])
                
                # rewards
                if avg_magnitude != 0:
                    R[s_ind, a_ind] = -dist / avg_magnitude
                elif s_x == target[0] and s_y == target[1]:
                    print(f'at target, action {a_ind} has positive reward')
                    R[s_ind, a_ind] = 0
                else:
                    R[s_ind, a_ind] = -99
                    
                # transitions
                net_angles = [] # for angles with large enough wind
                for net_vec in net_vectors:
                    if velocity_magnitude(net_vec) > 1e-2: # wind causes movement
                        net_angles.append(arr2polar(net_vec))

                freq, _ = np.histogram(net_angles, bins = angle_bins)
                f_ind = -1
                for neighbor in neighbors(s_x, s_y, s_length, s_width):
                    if f_ind == -1:
                        P[neighbor, s_ind, a_ind] = (
                            len(net_vectors) - len(net_angles)) / N
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
            
            
            
            
            
            
    