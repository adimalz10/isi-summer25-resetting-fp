import numpy as np
from numba import njit

@njit
def first_passage(dt, L, x0, D):
    """
    A first passage continuum diffusion is simulated here. The process terminates as soon as the particle reaches or crosses either boundary at 0 and L. 
    """

    x = x0
    c = 0

    if x0 == 0 or x0 == L:
        return 0

    while True:
        if x >= L or x <= 0:
            return c * dt # this is the first passage time
        
        x += np.random.normal(0, np.sqrt(2 * D * dt)) # there is an equal probability of moving to either side and the length of each step is drawn from a Gaussian distribution with mean 0 and standard deviation given by sqrt(2Ddt)

        
        c += 1 

@njit
def conditional_fp(dt, L, x0, D):
    """
    A conditional first passage continuum diffusion is simulated here. The process terminates as soon as the particle reaches or crosses either boundary at 0 and L. 
    """

    x = x0
    c = 0
    traj = [x0]

    if x0 == 0:
        return 0, 0
    elif x0 == L:
        return 1, 0

    while True:
        if x <= 0 or x>= L:
            break

        x += np.random.normal(0, np.sqrt(2 * D * dt))
        traj.append(x)

        c += 1
    
    if traj[-1] <= 0:
        return 0, c * dt
    elif traj[-1] >= L:
        return 1, c * dt

@njit  
def resetting_fp(dt, L, x0, r, D):
    """
    A first passage continuum diffusion is simulated here. The process terminates as soon as the particle reaches or crosses either boundary at 0 and L. 
    """

    x = x0
    c = 0

    if x0 == 0 or x0 == L:
        return 0

    while True:
        if x >= L or x <= 0:
            return c * dt # this is the first passage time
        
        rn = np.random.random()

        if rn < r * dt:
            x = x0
        else:
            x += np.random.normal(0, np.sqrt(2 * D * dt))# there is an equal probability of moving to either side and the length of each step is drawn from a Gaussian distribution with mean 0 and standard deviation given by sqrt(2Ddt) 

        c += 1

@njit
def resetting_fp_conditional(dt, L, x0, r, D):
    """
    A conditional first passage continuum diffusion is simulated here. The process terminates as soon as the particle reaches or crosses either boundary at 0 and L. 
    """

    x = x0
    c = 0
    traj = [x0]

    if x0 == 0:
        return 0, 0
    elif x0 == L:
        return 1, 0

    while True:
        if x <= 0 or x>= L:
            break

        if np.random.random() < r * dt:
            x = x0
        else:
            x += np.random.normal(0, np.sqrt(2 * D * dt))

        traj.append(x)

        c += 1
    
    if traj[-1] <= 0:
        return 0, c * dt
    elif traj[-1] >= L:
        return 1, c * dt
    
@njit
def fp_N(N, dt, L, x0, D):
    # underlying unconditional
    fpt_arr = np.zeros(N)
    for i in range(N):
        fpt_arr[i] = first_passage(dt, L, x0, D)

    if len(fpt_arr) == 0:
        return 0, 0
    
    return np.mean(fpt_arr), np.std(fpt_arr)

@njit
def fp_conditional_N(N, dt, L, x0, D):
    # underlying conditional
    fpt_0_arr = []
    fpt_L_arr = []
    for i in range(N):
        check, t = conditional_fp(dt, L, x0, D)
        if check == 0:
            fpt_0_arr.append(t)
        else:
            fpt_L_arr.append(t)

    epsplus = len(fpt_L_arr)/(len(fpt_0_arr) + len(fpt_L_arr))
    epsminus = len(fpt_0_arr)/(len(fpt_0_arr) + len(fpt_L_arr))

    if len(fpt_0_arr) == 0:
        return 0, 0, np.mean(np.array(fpt_L_arr)), np.std(np.array(fpt_L_arr)), epsminus, epsplus
    elif len(fpt_L_arr) == 0:
        return np.mean(np.array(fpt_0_arr)), np.std(np.array(fpt_0_arr)), 0, 0, epsminus, epsplus
    
    return np.mean(np.array(fpt_0_arr)), np.std(np.array(fpt_0_arr)), np.mean(np.array(fpt_L_arr)), np.std(np.array(fpt_L_arr)), epsminus, epsplus

@njit
def resetting_fp_N(N, dt, L, x0, r, D):
    # resetting unconditional
    fpt_arr = np.zeros(N)
    for i in range(N):
        fpt_arr[i] = resetting_fp(dt, L, x0, r, D)

    if len(fpt_arr) == 0:
        return 0, 0
    
    return np.mean(fpt_arr), np.std(fpt_arr)

@njit
def resetting_fp_conditional_N(N, dt, L, x0, r, D):
    # resetting conditional
    fpt_0_arr = []
    fpt_L_arr = []
    for i in range(N):
        check, t = resetting_fp_conditional(dt, L, x0, r, D)
        if check == 0:
            fpt_0_arr.append(t)
        else:
            fpt_L_arr.append(t)

    epsplus = len(fpt_L_arr)/(len(fpt_0_arr) + len(fpt_L_arr))
    epsminus = len(fpt_0_arr)/(len(fpt_0_arr) + len(fpt_L_arr))

    if len(fpt_0_arr) == 0:
        return 0, 0, np.mean(np.array(fpt_L_arr)), np.std(np.array(fpt_L_arr)), epsminus, epsplus
    elif len(fpt_L_arr) == 0:
        return np.mean(np.array(fpt_0_arr)), np.std(np.array(fpt_0_arr)), 0, 0, epsminus, epsplus
    
    return np.mean(np.array(fpt_0_arr)), np.std(np.array(fpt_0_arr)), np.mean(np.array(fpt_L_arr)), np.std(np.array(fpt_L_arr)), epsminus, epsplus