# plotting beta distributions for figure in paper 
import numpy as np 
from scipy.stats import beta
from scipy.stats import uniform
import matplotlib.pyplot as plt
import seaborn as sns


def beta_distribution(samples):
    # everyone 0.5
    sample_list_0 = [0.5 for _ in range(samples)]
    
    # normalish centred at 0.5
    a = 30
    b = 30
    sample_list_1 = [beta.rvs(a,b) for _ in range(samples)]

    # smooshed normal 
    a = 1.5
    b = 1.5
    sample_list_2 = [beta.rvs(a,b) for _ in range(samples)]

    # one or none
    a = 0.25
    b = 0.25
    sample_list_3 = [beta.rvs(a,b) for _ in range(samples)]

    return sample_list_0, sample_list_1, sample_list_2, sample_list_3

def plot_beta_distributions():
    palette = sns.color_palette()
    fig, ax = plt.subplots(1, 1)

    loc_u = 0.5
    ax.axvline(x = 0.5, label = 'delta', color = palette[0])

    a = 30
    b = 30
    x = np.linspace(beta.ppf(0, a, b), beta.ppf(1, a, b), 100)
    ax.plot(x, beta.pdf(x, a, b), label='beta(30,30)', color = palette[1])

    a = 1.5
    b = 1.5
    x = np.linspace(beta.ppf(0, a, b), beta.ppf(1, a, b), 100)
    ax.plot(x, beta.pdf(x, a, b), label='beta(1.5,1.5)', color = palette[2])

    a = 0.25
    b = 0.25
    x = np.linspace(beta.ppf(0, a, b), beta.ppf(1, a, b), 100)
    ax.plot(x, beta.pdf(x, a, b), label='beta(0.25,0.25)', color = palette[3])


    ax.legend()

    plt.savefig('beta-distributions.pdf')

    return 