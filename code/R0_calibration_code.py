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

from generate_property_grid import generate_properties
from class_definitions import Property 
from class_definitions import Animal 
from FOI_calculation_fns import calculate_FOI 


def R0_calibration_single_sim(params, grid_size, property_radius):
    # outputs of interest - how many properties were culled and how many were vaccinated
    # i.e. how many reported, how many vaccinated and how widespread was the outbreak
  
    # generate property grid
    property_coordinates, adjacency_matrix, neighbour_pairs, neighbourhoods, neighbourhoods_ring_culling = generate_properties(grid_size, property_radius, params)
    
    # initialise properties 
    properties = []
    for i in range(params['n']):
        properties.append(Property(params))
        properties[i].init_animals(params)
        properties[i].coordinates = property_coordinates[i]
        properties[i].radius = property_radius[i]
        properties[i].area = np.pi*(property_radius[i]**2)
        properties[i].neighbourhood = neighbourhoods[i]
        properties[i].total_neighbours = len(properties[i].neighbourhood)

    # seed infection
    # property coordinates allocated at random, so we always the first property in each simulation 
    seed_property = 0
    seed_animal = 0
    properties[seed_property].infection_status = 1
    properties[seed_property].animals[seed_animal].status = 'infected'
    properties[seed_property].number_infected = 1
    properties[seed_property].number_infectious = 1
    properties[seed_property].prop_infectious = 1/params['size']
    properties[seed_property].cumulative_infections = 1
    infected_sum = 1

    # initialise list of cumulative infections from each property - calculated for FOI every loop
    cumulative_infection_proportions = list(np.zeros(params['n']))
    cumulative_infection_proportions[seed_property] = properties[seed_property].cumulative_infections/len(properties[seed_property].animals)

    # initialise FOI - calculated every loop
    FOI = list(np.zeros(params['n']))

    time = 0
    R0_cases = 0
    # start time loop
    while infected_sum > 0:
        time += 1

        # calculate FOI for each property
        for i, premise in enumerate(properties):
            if not premise.culled_status:
                FOI[i] = calculate_FOI(properties, i, params)
                # FOI[i] = calculate_FOI(premise, params, cumulative_infection_proportions)

        for i, premise in enumerate(properties):
            R0_cases += premise.infection_model_R0_calibration(params, FOI[i])

        
        # update counts 
        # simulation ends when infected_sum = 0
        infected_sum = 0
        for i, premise in enumerate(properties):
            premise.update_counts()
            
            # for FOI calculation 
            cumulative_infection_proportions[i] = premise.cumulative_infections/len(premise.animals)

            # when first infection finishes its course
            infected_sum += premise.number_infected
        
    return R0_cases

def R0_calibration_multi_sims(params, grid_size, property_radius, sims):
    R0_count = 0
    # for _ in tqdm(range(sims)):
    #     R0_single_sim = R0_calibration_single_sim(params, grid_size, property_radius)
    #     R0_count += R0_single_sim

    for _ in tqdm(range(sims)):
        R0_single_sim = R0_calibration_single_sim(params, grid_size, property_radius)
        R0_count += R0_single_sim

    R0 = R0_count / sims

    return R0

def R0_calibration_ABC(R0, epsilon, n_output_vals, params, grid_size, property_radius, sims):
    # rejection sampling algorithm 
    beta_output = []

    # start searching around beta ~ R0 /(infectious period * N)
    beta_mean = R0 / (params['infectious_period']* params['size'])
    beta_sd = 0.01

    while len(beta_output) < n_output_vals:
        beta_sample = np.random.normal(beta_mean, beta_sd)
        print(beta_sample)
        if beta_sample > 0:
            params['beta_animal'] = beta_sample
            params['beta_wind'] = params['beta_animal']*params['beta_wind_modifier']
            R0_sample = R0_calibration_multi_sims(params, grid_size, property_radius, sims)
            print("I've gotta check, R0 = " + str(R0_sample))
            if (R0_sample - R0)**2 < epsilon:
                # beta_output.append([beta_sample, R0])
                beta_output.append(beta_sample)
                print('I have ' + str(len(beta_output)) + ' samples, this one was R0 = ' +  str(R0_sample))

    # create dataframe (easier to save to csv)
    beta_dict = {'beta_samples': beta_output}
    beta_df = pd.DataFrame(beta_dict)

    # create folder in which to save beta vals (if not already created) 
    beta_dir = Path('results')
    beta_dir.mkdir(parents = True, exist_ok = True)

    # save beta values
    beta_df.to_csv('results/beta_sample.csv')

    return beta_output 