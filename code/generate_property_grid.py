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

def assign_coordinates(n, grid_size, property_radius):
    # initialise property coordinate array to be returned 
    property_coordinates = np.zeros((n, 2))

    # we consider a coordinate grid of grid_size x grid_size 
    low_coord = 0
    high_coord = grid_size

    # coordinate allocation loop 
    properties_allocated = 0
    while properties_allocated < n:
        # randomly generate coordinates
        x = np.random.randint(low_coord, high_coord)
        y = np.random.randint(low_coord, high_coord)
        
        # check for overlap between property areas 
        overlap = 0
        for i in range(properties_allocated):
            # distance between centres 
            distance = np.linalg.norm(np.array([x,y]) - np.array(property_coordinates[i]))
            
            # minimum allowed distance between centres 
            min_distance = property_radius[i]+property_radius[properties_allocated]

            # check for overlap between properties 
            if distance < min_distance:
                overlap = 1
        
        # If there's no overlap between the properties, assign coordinates
        if overlap == 0:
            property_coordinates[properties_allocated,0] = x
            property_coordinates[properties_allocated,1] = y
            properties_allocated += 1

    return property_coordinates

def assign_neighbours(property_coordinates, property_radius, params):
    # initialise adjacency matrix, neighbourhood list and neighbour pairs list for output 
    n = params['n']
    r = params['r']
    r_ring_action = params['r_ring_action']
    adjacency_matrix = np.zeros((n,n))
    neighbour_pairs = []
    neighbourhoods = []
    neighbourhoods_ring_action = []

    # loop over property list
    for p1 in range(n): 
        # create neighbourhood list for the property
        neighbourhoods.append([])
        neighbourhoods_ring_action.append([])

        # loop over property list
        for p2 in range(n):
            # don't consider property whose neighbourhood we're calculating
            if p1 != p2:
                # calculate distance between centres
                dist_centres = np.linalg.norm(np.array(property_coordinates[p1]) - np.array(property_coordinates[p2]))

                # calculate minimum distance between borders
                dist_boundaries = dist_centres - (property_radius[p1] + property_radius[p2])

                boundaryp2_centrep1 = dist_centres - property_radius[p2]

                # within neighbour range => neighbours 
                if boundaryp2_centrep1 <= r:
                    # update adjacency matrix 
                    # p2 is a neighbour of p1
                    adjacency_matrix[p1, p2] = 1
                    # adjacency_matrix[p2, p1] = 1

                    # coordinates of properties 
                    x = [property_coordinates[p1, 0], property_coordinates[p2, 0]]
                    y = [property_coordinates[p1, 1], property_coordinates[p2, 1]]

                    # append neighbour pair list with coordinates of each property (used for plotting)
                    neighbour_pairs.append([x,y])

                    # append neighbourhood list with index of neighbour and distance between boundaries 
                    neighbourhoods[p1].append([p2, dist_boundaries, dist_centres])

                if boundaryp2_centrep1 <= r_ring_action: 
                    neighbourhoods_ring_action[p1].append(p2)



    return adjacency_matrix, neighbour_pairs, neighbourhoods, neighbourhoods_ring_action

def plot_coordinates(property_coordinates, neighbour_pairs, property_radius, r):
    # initialise figure 
    fig, ax = plt.subplots(figsize = (6,6))
    
    # plot each property 
    for i in range(len(property_coordinates)):
            # property area - defined by property radius
            circle = Circle((property_coordinates[i,0], property_coordinates[i,1]), property_radius[i], alpha = 0.5, color = 'green')
            # circle_2 = Circle((property_coordinates[i,0], property_coordinates[i,1]), r, alpha = 0.3, color = 'blue')
            ax.scatter(property_coordinates[i,0], property_coordinates[i,1], color = 'green', s = 0.00001)
            ax.add_patch(circle)
            # ax.add_patch(circle_2)
    circle_2 = Circle((property_coordinates[6,0], property_coordinates[6,1]), r, alpha = 0.3, color = 'blue')
    circle_3 = Circle((property_coordinates[2,0], property_coordinates[2,1]), r, alpha = 0.3, color = 'purple')
    # ax.add_patch(circle_2)
    # ax.add_patch(circle_3)

    plt.axis('equal')
    plt.xlim(-100, 1100)
    plt.ylim(-100, 1100)
    plt.title('Property network 1')
    # fig.show()

    # plot edges between neighbouring properties
    # for node_coordinates in neighbour_pairs:
    #     ax.plot(node_coordinates[0], node_coordinates[1])

    return 

def generate_properties(grid_size, property_radius, params):
    # randomly place properties in grid_size x grid_size coordinate space
    property_coordinates = assign_coordinates(params['n'], grid_size, property_radius)

    # find neighbours given maximum neighbour distance
    adjacency_matrix, neighbour_pairs, neighbourhoods, neighbourhoods_ring_action = assign_neighbours(property_coordinates, property_radius, params)

    # plot network
    # plot_coordinates(property_coordinates, neighbour_pairs, property_radius, params['r'])

    return property_coordinates, adjacency_matrix, neighbour_pairs, neighbourhoods, neighbourhoods_ring_action