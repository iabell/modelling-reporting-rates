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
from moviepy.editor import ImageSequenceClip
import os
from matplotlib.patches import Circle
from shapely.geometry import Point
from shapely.geometry import Polygon
import math
from datetime import datetime 
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from generate_property_grid import generate_properties
from class_definitions import Property 
from class_definitions import Animal 
from FOI_calculation_fns import calculate_FOI 
from plotting_code import plot_graph 
from plotting_code import make_video
from animal_movement_code import animal_movement 
from save_data_fns import save_data


def ABM(params, plotting, grid_size, property_radius):
    # outputs of interest - how many properties were culled and how many were vaccinated
    # i.e. how many reported, how many vaccinated and how widespread was the outbreak
    total_culled = 0
    total_vaccinated = 0

    # generate property grid
    property_coordinates, adjacency_matrix, neighbour_pairs, neighbourhoods, neighbourhoods_ring_action = generate_properties(grid_size, property_radius, params)
    
    # initialise properties 
    properties = []
    for i in range(params['n']):
        properties.append(Property(params))
        properties[i].init_animals(params)
        properties[i].coordinates = property_coordinates[i]
        properties[i].radius = property_radius[i]
        properties[i].area = np.pi*(property_radius[i]**2)
        properties[i].neighbourhood = neighbourhoods[i]
        properties[i].neighbourhood_ring_action = neighbourhoods_ring_action[i]
        properties[i].total_neighbours = len(properties[i].neighbourhood)

    # seed infection
    # property coordinates allocated at random, so we always the first property in each simulation 
    seed_property = 0
    # seed outbreak with 4 infected animals
    seed_animals = range(5)
    properties[seed_property].infection_status = 1
    for animal in seed_animals:
        properties[seed_property].animals[animal].infection_status = 'infectious'
    properties[seed_property].number_infected = 5
    properties[seed_property].number_infectious = 5
    properties[seed_property].prop_infectious = 5/params['size']
    properties[seed_property].cumulative_infections = 5
    infected_sum = 5

    # initialise list of cumulative infections from each property - calculated for FOI every loop
    cumulative_infection_proportions = list(np.zeros(params['n']))
    cumulative_infection_proportions[seed_property] = properties[seed_property].cumulative_infections/len(properties[seed_property].animals)

    # initialise FOI - calculated every loop
    FOI = list(np.zeros(params['n']))
    
    # things for single simulation plotting
    # plot_graph(properties, property_coordinates, time)
    # plotting_list = []
    if plotting:
        plt.figure()

    time = 0
    stop_movement = 0
    delay_clock = 0

    if plotting:
        plot_graph(properties, property_coordinates, time, params)
        # make_video()

    # if property doesn't report and infection dies out 
    reporting_tracker = [[-1, 0]]

    # keeping track of if infection dies out 
    died_out = 0

    # looking at probability of reporting infection over time 
    prob_of_report_dict = {
        'time': [],
        'infections': [],
        'prob': []
    }


    # start time loop
    while (infected_sum > 0 or delay_clock > 0) and time < 300: #time just as a failsafe
        
        time += 1
        # plotting_list.append([])
        # while loop checking
        
        # recalculated at the end of the time step
        infected_properties = 0

        # keeping track of properties to ring cull/vaccinate/test this timestep
        ring_action_status = list(np.zeros(params['n']))

        # calculate FOI for each property
        for i, premise in enumerate(properties):
            FOI[i] = calculate_FOI(properties, i, params)
                # FOI[i] = calculate_FOI(premise, params, cumulative_infection_proportions)

        # voluntary reporting and culling 
        for premise_index, premise in enumerate(properties): 
            
            #keeping track of reporting probability for the initial property i.e. index = 0
            if not premise_index: 
                # calculate probability of reporting 
                if premise.prop_clinical <= params['clinical_reporting_threshold']:
                    chance_of_reporting = 0
                else:
                    chance_of_reporting = params['prob_report']*premise.prop_clinical
                prob_of_report_dict['time'].append(time)
                prob_of_report_dict['infections'].append(premise.prop_clinical)
                prob_of_report_dict['prob'].append(chance_of_reporting)

            stop_movement, ring_action_flag, reporting_tracker = premise.reporting(params, stop_movement, time, premise_index, reporting_tracker)

                
            # keep track of properties to ring cull, ring vaccination and ring testing
            if ring_action_flag:
                # keep track of neighbours to cull/vaccinate/test this timestep
                for neighbour_index in premise.neighbourhood_ring_action: 
                    ring_action_status[neighbour_index] = 1

        # vaccination 
        for premise in properties: 
            # no voluntary vaccination 
            premise.vaccination(params)

        # testing
        for premise in properties:
            premise.testing(params)

        # ring culling
        if params['ring_culling']:
            for i, premise in enumerate(properties):
                if ring_action_status[i]:
                    premise.ring_cull(params)

        # ring vaccination
        if params['ring_vaccination']:
            for i, premise in enumerate(properties):
                if ring_action_status[i]:
                    premise.ring_vaccinate(params)

        # ring testing
        if params['ring_testing']:
            for i, premise in enumerate(properties):
                if ring_action_status[i]:
                     premise.ring_test(params)
            
        # infection_model
        for i, premise in enumerate(properties):
            premise.infection_model(params, FOI[i])
        
        # movement of animals
        if not stop_movement:
            animal_movement(properties, params, time)
        
        # while loop checking
        # update counts 
        # simulation ends when infected_sum = 0
        infected_sum = 0
        delay_clock = 0
        for i, premise in enumerate(properties):
            premise.update_counts()
            if not premise.culled_status:
                cumulative_infection_proportions[i] = premise.cumulative_infections/len(premise.animals)
                infected_sum += premise.number_infected

            # update delay clocks
            if premise.vaccination_delay_clock and not premise.vaccination_status:
                delay_clock += 1
            if premise.culling_delay_clock and not premise.culled_status:
                delay_clock += 1
            if premise.testing_delay_clock:
                delay_clock += 1 


        if plotting:
            plot_graph(properties, property_coordinates, time, params)
    
    if plotting:
        plot_graph(properties, property_coordinates, time, params)
        make_video()

    total_tested = 0
    # statistics from end of simulation
    for premise in properties:
        if premise.culled_status:
            total_culled += 1
        if premise.vaccination_status:
            total_vaccinated += 1
        total_tested += premise.number_times_tested

    # resources_count += total_culled + total_vaccinated

    length_of_outbreak = time
    total_resources = total_culled + total_vaccinated + total_tested

    # first element of first list in reporting_tracker is time of first report
    if len(reporting_tracker) > 1:
        time_of_first_report = reporting_tracker[1][0]
        properties_reported = [x[1] for x in reporting_tracker[1:]]
        total_reported = len(set(properties_reported))

    else:
        died_out = 1
        time_of_first_report = -1
        properties_reported = -1
        total_reported = 0
        
    return total_culled, total_vaccinated, total_tested, length_of_outbreak, total_resources, time_of_first_report, total_reported, died_out, prob_of_report_dict


def prob_report_plotting(params, beta_samples_filename, seed):
    # simulation set up 
    beta_samples = read_beta_sample_csv(beta_samples_filename)
    params['beta_animal'] = np.random.choice(beta_samples)
    # we want no transmission between properties and no reporting for this plot 
    params['beta_wind'] = 0
    params['movement_probability'] = 0
    params['prob_report'] = 0
    property_radius = params['n']*np.ones(params['n'])

    # assume ring culling (strategy doesn't actually matter for calculating probability of report because we're looking at the seeded property)
    set_params(params, 'ring_culling', 0)
    
    # set seed to reproduce plot
    np.random.seed(seed)

    # simulate infection for one property 
    total_culled, total_vaccinated, total_tested, length_of_outbreak, total_resources, time_of_first_report, number_reported, died_out, simulation_dict = ABM(params, 0, params['grid_size'], property_radius)
    
    prob_report_01 = [0.1*x for x in simulation_dict['infections']]
    prob_report_025 = [0.25*x for x in simulation_dict['infections']]
    prob_report_05 = [0.5*x for x in simulation_dict['infections']]
    prob_report_075 = [0.75*x for x in simulation_dict['infections']]
    prob_report_1 = [x for x in simulation_dict['infections']]

    daily_no_report = [1 - p for p in prob_report_01]
    prob_no_report = np.cumprod(daily_no_report)
    cdf_prob_report_01 = [1 - p for p in prob_no_report]
    
    daily_no_report = [1 - p for p in prob_report_025]
    prob_no_report = np.cumprod(daily_no_report)
    cdf_prob_report_025 = [1 - p for p in prob_no_report]

    daily_no_report = [1 - p for p in prob_report_05]
    prob_no_report = np.cumprod(daily_no_report)
    cdf_prob_report_05 = [1 - p for p in prob_no_report]

    daily_no_report = [1 - p for p in prob_report_075]
    prob_no_report = np.cumprod(daily_no_report)
    cdf_prob_report_075 = [1 - p for p in prob_no_report]

    daily_no_report = [1 - p for p in prob_report_1]
    prob_no_report = np.cumprod(daily_no_report)
    cdf_prob_report_1 = [1 - p for p in prob_no_report]

    # calculating cumulative probability function
    # daily_no_report = [1 - p for p in prob_report_dict_05['prob']]
    # prob_no_report = np.cumprod(daily_no_report)
    # cum_prob_reported = [1 - p for p in prob_no_report]

    # 1 figure version
    fig, axs = plt.subplots(1, 1, figsize = (12, 8))

    # 2 figure version
    # fig, axs = plt.subplots(2, 1, figsize = (8, 9.5))

    palette = sns.color_palette("muted")
    r01_colour = palette[0]
    r025_colour = palette[1]
    r05_colour = palette[2]
    r075_colour = palette[3]
    r1_colour = palette[4]


    axs.plot(simulation_dict['time'], cdf_prob_report_01, label = 'r = 0.1', color = r01_colour)
    axs.plot(simulation_dict['time'], cdf_prob_report_025, label = 'r = 0.25', color = r025_colour)
    axs.plot(simulation_dict['time'], cdf_prob_report_05, label = 'r = 0.5', color = r05_colour)
    axs.plot(simulation_dict['time'], cdf_prob_report_075, label = 'r = 0.75', color = r075_colour)
    axs.plot(simulation_dict['time'], cdf_prob_report_1, label = 'r = 1', color = r1_colour)
    axs.axvline(x = 6, linestyle = ':', color = 'black')
    axs.axvline(x = 22, linestyle = ':', color = 'black')
    axs.legend(reverse = True)
    plt.xlabel('Time (days)')
    axs.set_ylabel('Probability a property has reported infection by a given day\n(Cumulative distribution function)')
    


# 2 axes plot
    # axs[0].plot(simulation_dict['time'], prob_report_01, label = 'r = 0.1', color = r01_colour)
    # axs[0].plot(simulation_dict['time'], prob_report_025, label = 'r = 0.25', color = r025_colour)
    # axs[0].plot(simulation_dict['time'], prob_report_05, label = 'r = 0.5', color = r05_colour)
    # axs[0].plot(simulation_dict['time'], prob_report_075, label = 'r = 0.75', color = r075_colour)
    # axs[0].plot(simulation_dict['time'], prob_report_1, label = 'r = 1', color = r1_colour)
    # axs[0].legend(reverse = True)
    # axs[0].axvline(x = 6, linestyle = ':', color = 'black')
    # axs[0].axvline(x = 22, linestyle = ':', color = 'black')
    # axs[0].set_ylabel('Probability a property reports on a given day\n(Probability density function)')


    # axs[1].plot(simulation_dict['time'], cdf_prob_report_01, label = 'r = 0.1', color = r01_colour)
    # axs[1].plot(simulation_dict['time'], cdf_prob_report_025, label = 'r = 0.25', color = r025_colour)
    # axs[1].plot(simulation_dict['time'], cdf_prob_report_05, label = 'r = 0.5', color = r05_colour)
    # axs[1].plot(simulation_dict['time'], cdf_prob_report_075, label = 'r = 0.75', color = r075_colour)
    # axs[1].plot(simulation_dict['time'], cdf_prob_report_1, label = 'r = 1', color = r1_colour)
    # axs[1].axvline(x = 6, linestyle = ':', color = 'black')
    # axs[1].axvline(x = 22, linestyle = ':', color = 'black')
    # axs[1].legend(reverse = True)
    # plt.xlabel('Time (days)')
    # axs[1].set_ylabel('Probability a property has reported infection by a given day\n(Cumulative distribution function)')
    
    
    # filename and create folder to save things in 
    now = datetime.now()
    date = now.strftime('%d_%m_%Y/')
    filepath = 'plotting/' + date
    plotting_dir = Path(filepath)
    plotting_dir.mkdir(parents = True, exist_ok = True)
    filename = filepath + 'probability_of_reporting'
    filename_png = filename + '.png'
    filename_eps = filename + '.eps'

    plt.savefig(filename_png)
    plt.savefig(filename_eps)
    

    return 


def multiple_simulations(params, beta_samples):
    total_culled_sims = {'ring_culling': [],
                         'ring_testing': [],
                         'ring_vaccination_perfect': [],
                         'ring_vaccination_imperfect': []
                        }
    total_vaccinated_sims = {'ring_culling': [],
                         'ring_testing': [],
                         'ring_vaccination_perfect': [],
                         'ring_vaccination_imperfect': []
                            }
    total_tested_sims = {'ring_culling': [],
                         'ring_testing': [],
                         'ring_vaccination_perfect': [],
                         'ring_vaccination_imperfect': []
                        }
    length_of_outbreak_sims =  {'ring_culling': [],
                                'ring_testing': [],
                                'ring_vaccination_perfect': [],
                                'ring_vaccination_imperfect': []
                                }
    total_resources_sims =  {'ring_culling': [],
                                'ring_testing': [],
                                'ring_vaccination_perfect': [],
                                'ring_vaccination_imperfect': []
                                }
    number_reports_sims =  {'ring_culling': [],
                                'ring_testing': [],
                                'ring_vaccination_perfect': [],
                                'ring_vaccination_imperfect': []
                                }
    first_report_dates_sims =  {'ring_culling': [],
                                'ring_testing': [],
                                'ring_vaccination_perfect': [],
                                'ring_vaccination_imperfect': []
                                }
    property_radius = params['n']*np.ones(params['n'])
    plotting = 0

    strategy_names = ['ring_culling', 'ring_testing', 'ring_vaccination_perfect', 'ring_vaccination_imperfect']
    action_param_names = ['ring_culling', 'ring_testing', 'ring_vaccination', 'ring_vaccination']
    vax_modifier_strategies = [0, 0, 0, 0.2]

    for i in tqdm(range(params['sims']), desc = 'simulations'):
        np.random.seed(int(datetime.now().timestamp()))

        for a, a_name in enumerate(strategy_names):

            # draw  random beta from samples
            params['beta_animal'] = np.random.choice(beta_samples)
            params['beta_wind'] = params['beta_wind_modifier'] * params['beta_animal']

            # set params for strategy 
            set_params(params, action_param_names[a], vax_modifier_strategies[a])

            # run simulation
            total_culled, total_vaccinated, total_tested, length_of_outbreak, total_resources, time_of_first_report, number_reported, died_out, prob_report_dict = ABM(params, plotting, params['grid_size'], property_radius)
            # print('died out = ' + str(died_out) + ' properties')

            # only update if infection doesn't die out 
            # if not died_out:
            # update data lists
            total_culled_sims[a_name].append(total_culled)
            total_vaccinated_sims[a_name].append(total_vaccinated)
            total_tested_sims[a_name].append(total_tested)
            length_of_outbreak_sims[a_name].append(length_of_outbreak)
            total_resources_sims[a_name].append(total_resources)
            first_report_dates_sims[a_name].append(time_of_first_report)
            number_reports_sims[a_name].append(number_reported)
            # else:
            #     # else run another simulation 
            #     i = i-1
            #     break # get out of loop so other strategies not evaluated 

    return total_culled_sims, total_vaccinated_sims, total_tested_sims, length_of_outbreak_sims, total_resources_sims, first_report_dates_sims, number_reports_sims

def set_params(params, strat_name, vax_modifier):
    # turn all strategies off
    params['ring_culling'] = 0
    params['ring_testing'] = 0
    params['ring_vaccination'] = 0
    params['vax_modifier'] = 0  

    params[strat_name] = 1
    
    # perfect vs imperfect vaccine (doesn't matter for other strategies)
    params['vax_modifier'] = vax_modifier
    return params

def one_at_a_time_sens(params, prob_report_range, beta_filename, objective_names, strategy_names):
    total_culled_data = {}
    total_vaccinated_data = {}
    total_tested_data = {}
    length_of_outbreak_data = {}
    total_resources_data = {}
    first_report_date_data = {}
    number_reports_data = {}

    beta_samples = read_beta_sample_csv(beta_filename)

    for i in tqdm(range(len(prob_report_range)), desc = 'sensitivity param'):
        r = prob_report_range[i]

        # update probability of reporting parameter for sensitivity analysis
        params['prob_report'] = r 

        total_culled_sims, total_vaccinated_sims, total_tested_sims, length_of_outbreak_sims, total_resources_sims, first_report_date_sims, number_reports_sims = multiple_simulations(params, beta_samples)

        total_culled_data[str(r)] = total_culled_sims
        total_vaccinated_data[str(r)] = total_vaccinated_sims
        total_tested_data[str(r)] = total_tested_sims
        length_of_outbreak_data[str(r)] = length_of_outbreak_sims
        total_resources_data[str(r)] = total_resources_sims
        first_report_date_data[str(r)] = first_report_date_sims
        number_reports_data[str(r)] = number_reports_sims

    data = [total_culled_data, total_vaccinated_data, total_tested_data, length_of_outbreak_data, total_resources_data, first_report_date_data, number_reports_data]
    
    # objective names = ['total_culled', 'total_vaccinated', 'total_tested', 'length_of_outbreak', 'total_resources', 'first_report_date', 'number_reports']
    # strategy names = ['ring_culling', 'ring_testing', 'ring_vaccination_perfect', 'ring_vaccination_imperfect']
    
    save_data(data, prob_report_range, objective_names , strategy_names)
    
    return 

def read_beta_sample_csv(filename):
    beta_samples_df = pd.read_csv(filename)
    beta_samples = beta_samples_df['beta_samples'].tolist()
    return beta_samples