from collections import namedtuple
import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import random as rand


def adjacency_matrix(n): # defines the adjacency matrix for a nxn grid

    # initialise adj_matrix 
    adj_matrix = np.zeros((n*n,n*n))
    
    # defining neighbouring indices
    def below(i):
        return i + n
    
    def above(i):
        return i - n

    def left(i):
        return i - 1
    
    def right(i):
        return i + 1

    # special cases (corners and edges rows/columns)
    # 0 (top left corner)
    adj_matrix[0, below(0)] = 1 
    adj_matrix[0, right(0)] = 1 

    # n-1 (top right corner)
    adj_matrix[n-1,below(n-1)] =1 
    adj_matrix[n-1,left(n-1)] = 1 

    # n*(n-1) (bottom left corner)
    adj_matrix[n*(n-1), above(n*(n-1))] = 1 
    adj_matrix[n*(n-1), right(n*(n-1))] = 1 

    #n*n - 1 (bottom right corner)
    adj_matrix[n*n - 1, above(n*n - 1)] = 1 
    adj_matrix[n*n - 1, left(n*n - 1)] = 1 

    if n>2:
        # top-most row
        for i in range(1,n-1):
            adj_matrix[i,below(i)] = 1 
            adj_matrix[i,right(i)] = 1 
            adj_matrix[i,left(i)] = 1 

        # bottom-most row 
        for i in range(n*(n-1) + 1, n*n - 1):
            adj_matrix[i, above(i)] = 1
            adj_matrix[i,right(i)] = 1 
            adj_matrix[i,left(i)] = 1 

        # left-most column
        for i in range(0,(n-1)):
            adj_matrix[n*i, above(n*i)] = 1
            adj_matrix[n*i, below(n*i)] = 1
            adj_matrix[n*i, right(n*i)] = 1

        # right-most column
        for i in range(2, n):
            adj_matrix[n*i - 1, above(n*i -1)] = 1
            adj_matrix[n*i - 1, below(n*i -1)] = 1
            adj_matrix[n*i - 1, left(n*i -1)] = 1

        # middle points
        for i in range(1,n-1):
            for j in range(1,n-1):
                adj_matrix[n*i + j, above(n*i + j)] = 1
                adj_matrix[n*i + j, below(n*i + j)] = 1
                adj_matrix[n*i + j, left(n*i + j)] = 1
                adj_matrix[n*i + j, right(n*i + j)] = 1

    return adj_matrix

def grid_ODES(params, t, i0_index):

    # differential equations 
    def deriv(y, t, beta_w, beta_b, theta, N, adj_M, n, D):
        
        S = y[:n**2]
        E = y[n**2:2*(n**2)]
        I = y[2*(n**2):]

        # Implementing detection: 
        # flag: 0 if farm culled

        true_false_I = [I[i] < D*N[0,i] for i in range(n**2)]
        I_including_detection = np.multiply(true_false_I, I)

        if False in true_false_I:
            flag = 1

        # Adjacent infected farms
        I_adj = np.matmul(I_including_detection,adj_M)

        FOI_between = beta_b*(np.multiply(S,I_adj))

 
        dSdt = -beta_w*np.divide((np.multiply(S,I)),N) - beta_b*(np.multiply(S,I_adj))
        dEdt = -dSdt - theta*E
        dIdt = theta*E

        dydt1 = np.append(dSdt, dEdt)
        dydt = np.append(dydt1, dIdt)

        return dydt
    

    # Initial conditions: pick a random farm with 2 initially infected animals 
    n = params['n']
    total_n = n**2
    I0 = np.zeros((1,total_n))
    I0[0,i0_index] = params['I0']

    # Initial conditions for S, E compartments
    S0 = np.multiply(params['N'], np.ones((1,total_n)))
    S0[0,i0_index] = params['N'][0,i0_index] - I0[0,i0_index]
    E0 = np.zeros((1,total_n))
    

    y0_1 = np.append(S0, E0)
    y0 = np.append(y0_1, I0)
    S = []
    E = []
    I = []

    # Integrate the SIR equations over the time grid, t. 
    ret = odeint(deriv, y0, t, args=(params['beta_w'],params['beta_b'],params['theta'], params['N'], params['adj_matrix'], params['n'], params['D']))
    for i in range(len(ret)):
        S.append(ret[i][:n**2])
        E.append(ret[i][n**2:2*(n**2)])
        I.append(ret[i][2*(n**2):])

    # S = ret[:n**2]
    # E = ret[n**2:2*(n**2)]
    # I = ret[2*(n**2):]
    
    # S,E,I = ret.T

    return S, E, I

def new_from_old(old, len_t, n):
    new = np.zeros((n**2, len_t))
    for i in range(n**2):
        new[i] = [old[j][i] for j in range(len_t)]
    return new

def extract_timeseries(S, E, I, n, len_t):
    # initialise final S, E, I arrays
    S_new = new_from_old(S, len_t, n)
    E_new = new_from_old(E, len_t, n)
    I_new = new_from_old(I, len_t, n)

    return S_new, E_new, I_new

def grid_plotting(n, S, E, I, t, D, N, i0_index):
    S_ext, E_ext, I_ext = extract_timeseries(S, E, I, n, len(t))

    fig, axs = plt.subplots(n, n)
    for i in range(n):
        for j in range(n):
            number = n*i +j
            axs[i,j].plot(t,S_ext[number][ :])
            axs[i,j].plot(t,E_ext[number][ :])
            axs[i,j].plot(t,I_ext[number][ :])
            axs[i,j].plot(t,D*N[0,number]*np.ones(len(t)))
            if number == i0_index:
                axs[i,j].set_title('(initially infected) Farm %i' %number)
            else:
                axs[i,j].set_title('Farm %i' %number)
            
            

    # setting x and y labels for each of the plots 
    # for ax in axs.flat: 
    #     ax.set(xlabel = 'xlabel', ylabel = 'ylabel')

    plt.show()

    return 0



t = np.linspace(0, 200, 402)

t_short = np.linspace(0,20,21)

N10 = 1000
N20 = 500

gamma = 1/20

theta = 1/7

omega1 = 1/100
omega2 = 1/200

beta = 0.8*gamma

n = 3

detection = 0.1

N = 1000*np.ones((1, n**2))

params = dict([
    ('beta', beta),
    ('beta_w', 3*beta),
    # ('beta_b', 0.5*beta/(N10+N20)),
    ('beta_b', 0.000001),
    ('gamma', gamma),
    ('theta', theta),
    ('omega1', omega1),
    ('omega2', omega2),
    ('I10', 10),
    ('I20', 0),
    ('I0',2),
    ('N10', N10), 
    ('N20', N20),
    ('N', 1000*np.ones((1, n**2))),
    ('n',n),
    ('adj_matrix', adjacency_matrix(n)),
    ('D', detection)])

i0_index = rand.randint(0,n**2-1)
print(i0_index)


S,E,I = grid_ODES(params, t, i0_index)

grid_plotting(n, S, E, I, t, detection, N, i0_index)
