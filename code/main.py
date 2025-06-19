from collections import namedtuple
import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import random as rand
from scipy.stats import hypergeom
from tqdm import tqdm 
import csv
import pandas as pd
from IPython import display
import time
from pathlib import Path
# from moviepy.editor import ImageSequenceClip
import os
from matplotlib.patches import Circle
from shapely.geometry import Point
from shapely.geometry import Polygon
import math
import seaborn as sns
from datetime import datetime 


# import functions from other files 
import importlib
from abm_fn import *
from R0_calibration_code import *
from generate_property_grid import *
from save_data_fns import *
from plotting_code import *


def single_sim_plot(params, plotting, grid_size, property_radius, seed):
    np.random.seed(seed)
    total_culled, total_vaccinated, total_tested, length_of_outbreak, total_resources, time_of_first_report, total_reported, died_out, prob_of_report_dict = ABM(params, plotting, grid_size, property_radius)
    print('vaccinated properties = ' + str(total_vaccinated) + ', testing resources used = ' + str(total_tested) + ', culled properties = ' + str(total_culled) + ', length of outbreak = ' + str(length_of_outbreak) + ' days')
    print(total_culled, total_vaccinated)

    

    return 


# model parameters 
params = {
    'grid_size': 1000,
    'sims': 1000,
    'n': 40,
    # 'property_radius': np.ones(40),
    'r': 140,
    'r_ring_action': 200,
    'prob_report': 0.5,
    'prob_vaccinate':0.5,
    # average dairy property
    'size': 400,
    # size of properties
    'latent_period': 2,
    'infectious_period': 6,
    'pre-clinical_period': 3,
    # 'clinical_period': 100,
    'vax_modifier': 0,
    'beta_wind_modifier': 0.5,
    'beta_wind': 0.5*0.005,
    'beta_animal': 0.005, # beta parameters will be updated at start of simulation 
    'clinical_reporting_threshold': 0.05,
    'movement_frequency': 7, #days
    'movement_probability': 0.4,
    'movement_prop_animals': 0.2,
    'vaccination_delay': 5,
    'culling_delay': 5,
    'testing_delay': 5,
    'ring_culling': 1,
    'ring_testing': 0,
    'ring_vaccination': 0,
    # beta calibration parameters
    'R0': 2,
    'epsilon': 0.1,
    'beta_output_vals': 100
}
property_radius = params['n']*np.ones(params['n'])

prob_report_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

actions = ['ring_culling_', 'ring_testing_', 'ring_vaccination_perfect_', 'ring_vaccination_imperfect_']
vax_modifier = [0, 0, 0, 0.2]

strategy_names = ['ring_culling', 'ring_testing', 'ring_vaccination_perfect', 'ring_vaccination_imperfect']
objectives = ['total_culled', 'total_vaccinated', 'total_tested', 'length_of_outbreak', 'total_resources', 'first_report_date', 'number_reports']

plotting = 0

# turn plotting and data generation on and off
generate_data = 0
read_data = 1
calibrate_R0 = 0
plot_single_sim = 0
multiplots = 0
reporting_stats_plots = 0
generate_optimal_data = 0
plot_optimal_data = 1


# things for file locations
beta_calibration_file = 'results/beta_sample.csv'
now = datetime.now()
data_date = now.strftime('%d_%m_%Y') #change if data generated earlier 
data_date = '28_03_2025'
optimal_csv_date = '01_04_2025'

# plotting set up 
palette = sns.color_palette('Paired')
colour_chart = {
        'ring_culling': palette[9],
        'ring_testing': palette[8],
        'ring_vaccination_perfect': palette[3],
        'ring_vaccination_imperfect': palette[2],
        'multi': palette[0]
    }
colour_chart_list = [palette[9], palette[8], palette[3], palette[2], palette[0]]

# # nice names for parameters for axis labels
objectives_nice_names = ['Total properties depopulated', 'Total properties vaccinated', 'Total properties tested', 'Length of outbreak', 'Resources used', 'First report day', 'Total properties reported']
strategies_nice_names = ['Ring culling', 'Ring testing', 'Ring vaccination\n(perfect)', 'Ring vaccination\n(imperfect)']


#-----------------------------------------------------#
#plotting and data generation code starts here
#-----------------------------------------------------#
# BETA CALIBRATION
# only need to run once for each R0 value 
if calibrate_R0:
    beta_output = R0_calibration_ABC(params['R0'], params['epsilon'], params['beta_output_vals'], params, params['grid_size'], property_radius, params['sims'])

#-----------------------------------------------------#
# SINGLE SIM PLOTTING:
if plot_single_sim:
    plotting = 1
    seed = 1
    single_sim_plot(params, plotting, params['grid_size'], property_radius, seed)
    # triplet_plot('000.png', '015.png', '125.png')


#-----------------------------------------------------#
# GENERATE DATA:
if generate_data:
    one_at_a_time_sens(params, prob_report_range, 'results/27_03_2025/beta_sample.csv', objectives, strategy_names)

#-----------------------------------------------------#
# READ DATA 
if read_data:
    data_df_incl_died_out = read_all_data(objectives, strategy_names, prob_report_range, data_date)

    data_df = remove_died_out_sims(data_df_incl_died_out)

    # # properties for easier viewing of dataframe (if you want to print to have a look)
    pd.set_option('display.max_rows', None)      # Show all rows
    pd.set_option('display.max_columns', None)   # Show all columns
    pd.set_option('display.width', None)         # Disable width limit
    pd.set_option('display.max_colwidth', None)
    # print(data_df[data_df['sim_index']==1]) #print dataframe to see what's happening
    # print(data_df_no_die_out)

#-----------------------------------------------------#
# PLOTS FOR INDIVIDUAL STRATEGIES AND OBJECTIVES
if multiplots:
    multiplot_objectives = ['total_culled', 'length_of_outbreak', 'total_resources', 'number_reports']
    multiplot_objectives_nice_names = ['Total properties depopulated', 'Length of outbreak', 'Resources used', 'Number of properties reported']
    individual_strategies_multiplot(data_df, strategy_names, multiplot_objectives, strategies_nice_names, multiplot_objectives_nice_names, colour_chart, colour_chart_list)

#-----------------------------------------------------#
# REPORTING STATS
if reporting_stats_plots:
    plot_reporting_stats(data_df, strategy_names, ['first_report_date'], strategies_nice_names, ['First report day'], colour_chart, colour_chart_list)

#-----------------------------------------------------#
# CALCULATE AND SAVE DATAFRAME FOR OPTIMAL AND DIFFERENCE PLOTS 
if plot_optimal_data:

    # # calculating optimal strategies and difference between other strategies and culling 
    management_objectives = objectives[:5]
    management_objectives_nice_names = objectives_nice_names[:5]

    if generate_optimal_data:
        df_optimal = work_out_optimal_and_diff(data_df, params['sims'], prob_report_range, management_objectives)

        # save dataframe to csv because it takes forever to generate 
        date = now.strftime('%d_%m_%Y')
        optimal_dir = Path('results/' + date)
        optimal_dir.mkdir(parents = True, exist_ok = True)
        df_optimal.to_csv('results/' + date + '/df_optimal.csv', index = False)

    # # read df_optimal from csv if previously generated 
    df_optimal = pd.read_csv('results/' + optimal_csv_date + '/df_optimal.csv')

    strategy_names.append('multi')
    strategies_nice_names.append('Multiple')
    plotting_optimal_percentages_and_difference(df_optimal, prob_report_range, strategy_names, strategies_nice_names, management_objectives, management_objectives_nice_names, colour_chart, colour_chart_list)
#-----------------------------------------------------#






