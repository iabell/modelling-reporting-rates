from collections import namedtuple
import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import random as rand



def two_farms(params, t):

    # initially farm 1 has 2 infected animals 
    I10 = params['I10']
    S10 = params['N10'] - I10
    E10 = 0
    R10 = 0


    I20 = params['I20']
    S20 = params['N20'] - I20
    E20 = 0
    R20 = 0  


    # differential equations 
    def deriv(y, t, beta, theta, gamma, omega1, omega2):
        N1 = sum(y[:3])
        N2 = sum(y[4:])

        S1, E1, I1, R1, S2, E2, I2, R2 = y
        dS1dt = -beta*S1*I1/N1 + omega2*(S2) - omega1*S1
        dE1dt = beta*S1*I1/N1 - theta*E1 + omega2*(E2) - omega1*E1
        dI1dt = theta*E1 -gamma*I1 + omega2*(I2) - omega1*I1
        dR1dt = gamma*I1 + omega2*(R2) - omega1* R1
        dS2dt = -beta*S2*I2/N2 + omega1*(S1) - omega2*S2
        dE2dt = beta*S2*I2/N2 - theta*E2 + omega1*(E1) - omega2*E2
        dI2dt = theta*E2 -gamma*I2 + omega1*(I1) - omega2*I2
        dR2dt = gamma*I2 + omega1*(R1) - omega2*R2

        print(S2 - S1)

        return dS1dt, dE1dt, dI1dt, dR1dt, dS2dt, dE2dt, dI2dt, dR2dt
    

    y0 = S10, E10, I10, R10, S20, E20, I20, R20



    # Integrate the SIR equations over the time grid, t. 
    ret = odeint(deriv, y0, t, args=(params['beta'],params['theta'],params['gamma'], params['omega1'],params['omega2']))
    S1, E1, I1, R1, S2, E2, I2, R2 = ret.T

    return np.array(S1), np.array(E1), np.array(I1), np.array(R1), np.array(S2), np.array(E2), np.array(I2), np.array(R2)


def two_farms_v2(params, t):

    # initially farm 1 has 2 infected animals 
    I10 = params['I10']
    S10 = params['N10'] - I10
    E10 = 0
    R10 = 0


    I20 = params['I20']
    S20 = params['N20'] - I20
    E20 = 0
    R20 = 0  


    # differential equations 
    def deriv(y, t, beta_w, beta_b, theta, gamma, N1, N2):


        S1, E1, I1, R1, S2, E2, I2, R2 = y
        dS1dt = -beta_w*S1*I1/N1 -beta_b*S1*I2
        dE1dt = beta_w*S1*I1/N1 + beta_b*S1*I2 - theta*E1 
        dI1dt = theta*E1 -gamma*I1 
        dR1dt = gamma*I1 
        dS2dt = -beta_w*S2*I2/N2 -beta_b*S2*I1
        dE2dt = beta_w*S2*I2/N2 + beta_b*S2*I1 - theta*E2 
        dI2dt = theta*E2 -gamma*I2 
        dR2dt = gamma*I2 

        print(S2 - S1)

        return dS1dt, dE1dt, dI1dt, dR1dt, dS2dt, dE2dt, dI2dt, dR2dt
    

    y0 = S10, E10, I10, R10, S20, E20, I20, R20



    # Integrate the SIR equations over the time grid, t. 
    ret = odeint(deriv, y0, t, args=(params['beta_w'],params['beta_b'],params['theta'],params['gamma'], params['N10'],params['N20']))
    S1, E1, I1, R1, S2, E2, I2, R2 = ret.T

    return np.array(S1), np.array(E1), np.array(I1), np.array(R1), np.array(S2), np.array(E2), np.array(I2), np.array(R2)

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


def grid_ODES(params, t):

    # differential equations 
    def deriv(y, t, beta_w, beta_b, theta, N, adj_M, n):
        
        S = y[:n**2]
        E = y[n**2:2*(n**2)]
        I = y[2*(n**2):]

        check = np.multiply(S,I)
        
        I_adj = np.matmul(I,adj_M)

        dSdt = -beta_w*(np.multiply(S,I))/N - beta_b*(np.multiply(S,I_adj))
        dEdt = -dSdt - theta*E
        dIdt = theta*E

        dydt1 = np.append(dSdt, dEdt)
        dydt = np.append(dydt1, dIdt)

        return dydt
    

    # Initial conditions: pick a random farm with 2 initially infected animals 
    n = params['n']
    total_n = n**2
    i = rand.randint(0,total_n-1)
    I0 = np.zeros((1,total_n))
    I0[0,i] = params['I0']

    # Initial conditions for S, E compartments
    S0 = params['N']*np.ones((1,total_n))
    S0[0,i] = params['N'] - I0[0,i]
    E0 = np.zeros((1,total_n))
    

    y0_1 = np.append(S0, E0)
    y0 = np.append(y0_1, I0)
    S = []
    E = []
    I = []

    # Integrate the SIR equations over the time grid, t. 
    ret = odeint(deriv, y0, t, args=(params['beta_w'],params['beta_b'],params['theta'], params['N'], params['adj_matrix'], params['n']))
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




t = np.linspace(0, 200, 402)

t_short = np.linspace(0,20,21)

N10 = 1000
N20 = 500

gamma = 1/20

theta = 1/7

omega1 = 1/100
omega2 = 1/200

beta = 0.4/7

n = 2


params = dict([
    ('beta', beta),
    ('beta_w', 2*beta),
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
    ('N', 1000),
    ('n',n),
    ('adj_matrix', adjacency_matrix(n))] )

# S1, E1, I1, R1, S2, E2, I2, R2 = two_farms(params, t)

# N1 = S1 + E1 + I1 + R1
# N2 = S2 + E2 + I2 + R2

# S1, E1, I1, R1, S2, E2, I2, R2 = two_farms_v2(params, t)


S,E,I = grid_ODES(params, t)




def grid_plotting(n, S, E, I, t):
    S_ext, E_ext, I_ext = extract_timeseries(S, E, I, n, len(t))

    fig, axs = plt.subplots(n, n)
    for i in range(n):
        for j in range(n):
            number = n*i +j
            axs[i,j].plot(t,S_ext[n*i + j][ :])
            axs[i,j].plot(t,E_ext[n*i + j][ :])
            axs[i,j].plot(t,I_ext[n*i + j][ :])
            axs[i,j].set_title('Farm %i' %number)

    # setting x and y labels for each of the plots 
    # for ax in axs.flat: 
    #     ax.set(xlabel = 'xlabel', ylabel = 'ylabel')

    plt.show()

    return 0

grid_plotting(n, S, E, I, t)

def plot(S1, E1, I1, R1, S2, E2, I2, R2):
    fig, axs = plt.subplots(2)
    # fig = plt.figure(facecolor = 'w')
    # ax = fig.add_subplot(111, facecolor = '#dddddd', axisbelow = True)
    axs[0].plot(t, S1, alpha = 0.5, lw =2, label = 'Susceptible_1')
    axs[0].plot(t, E1, alpha = 0.5, lw =2, label = 'Exposed_1')
    axs[0].plot(t, I1, alpha = 0.5, lw =2, label = 'Infected_1')
    axs[0].plot(t, R1, alpha = 0.5, lw =2, label = 'Recovered_1')
    # axs[0].plot(t, N1, alpha = 0.5, lw =2, label = 'N1')
    axs[1].plot(t, S2, alpha = 0.5, lw =2, label = 'Susceptible_2')
    axs[1].plot(t, E2, alpha = 0.5, lw =2, label = 'Exposed_2')
    axs[1].plot(t, I2, alpha = 0.5, lw =2, label = 'Infected_2')
    axs[1].plot(t, R2, alpha = 0.5, lw =2, label = 'Recovered_2')
    # axs[1].plot(t, N2, alpha = 0.5, lw =2, label = 'N2')
    # axs[1].set_ylabel('Animals')
    # # ax.set_ylim(0, N)
    # ax.yaxis.set_tick_params(length = 0)
    # ax.xaxis.set_tick_params(length = 0)
    plt.legend()

    plt.show()
    return 


# # ax.plot(t, H_2, 'g', alpha = 0.5, lw =2, label = 'Hospital')
# # ax.plot(t, M, 'c', alpha = 0.5, lw =2, label = 'Monitored')
# # ax.plot(t, H2, 'm', alpha = 0.5, lw =2, label = 'Hospital2')
# ax.plot(t, R, 'k', alpha = 0.5, lw =2, label = 'Removed')
# # ax.plot(t, beds_needed/1000, 'y', alpha = 0.5, lw =2, label = 'Beds')
# # ax.plot(t, R/1000, 'c', alpha = 0.5, lw =2, label = 'Removed')
# ax.set_ylabel('People')
# ax.set_ylim(0, N)
# ax.yaxis.set_tick_params(length = 0)
# ax.xaxis.set_tick_params(length = 0)
# # ax.title('Infections over time')
# plt.title('Epidemic')

# legend.get_frame().set_alpha(0.5)
# for spine in ('top', 'right', 'bottom', 'left'):
#     ax.spines[spine].set_visible(False)
# # plt.savefig("ODE_1_total.eps")
