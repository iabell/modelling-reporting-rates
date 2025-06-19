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
import seaborn as sns
import cv2 as cv

def plot_graph(properties, property_coordinates, time, params):
    fig, ax = plt.subplots(figsize = (6,6))
    # nodes 

    # palette = sns.color_palette('Paired')
    # vax_colour = palette[3]
    # sus_colour = palette[2]
    # infection_colour = palette[1]
    # reported_colour = palette[4]
    # culled_colour = palette[5]

    palette = sns.color_palette('deep')
    vax_colour = palette[2]
    sus_colour = palette[1]
    infection_colour = palette[9]
    reported_colour = palette[4]
    culled_colour = palette[3]
    testing_colour = palette[0]
    ring_colour = palette[7]
    vax_and_culled_colour = palette[8]

    for index, premise in enumerate(properties):
        circle_ring = Circle((premise.coordinates[0], premise.coordinates[1]), params['r_ring_action'], alpha = 0, color = ring_colour)
        ax.add_patch(circle_ring)
        if premise.ring_action_status: 
            if params['ring_culling']:
                circle_ring = Circle((premise.coordinates[0], premise.coordinates[1]), params['r_ring_action'], alpha = 0.2, color = culled_colour)
                ax.add_patch(circle_ring)
            elif params['ring_vaccination']:
                circle_ring = Circle((premise.coordinates[0], premise.coordinates[1]), params['r_ring_action'], alpha = 0.2, color = vax_colour)
                ax.add_patch(circle_ring)
            elif params['ring_testing']:
                circle_ring = Circle((premise.coordinates[0], premise.coordinates[1]), params['r_ring_action'], alpha = 0.2, color = testing_colour)
                ax.add_patch(circle_ring)

    for index, premise in enumerate(properties):
       
        if params['ring_culling'] or params['ring_vaccination'] or params['ring_testing']:
            # plot ring culling neighbours using edges
            # for farm in premise.neighbourhood_ring_action:
            #     ax.plot([premise.coordinates[0], property_coordinates[farm, 0]], [premise.coordinates[1], property_coordinates[farm, 1]], alpha = 0.2, color = 'black')
            if params['ring_culling']:
                # plt.title('Strategy: ring culling \n Day ' + str(time))
                plt.title('Day ' + str(time), fontsize = 20)
            if params['ring_vaccination']:
                # plt.title('Strategy: ring vaccination \n Day ' + str(time))
                plt.title('Day ' + str(time), fontsize = 20)
            if params['ring_testing']:
                # plt.title('Strategy: ring testing \n Day ' + str(time))
                plt.title('Day ' + str(time), fontsize = 20)

            # plot ring actions neighbours using circles
            
            
        # else:
        #     # plot wind dispersal edges
        #     for farm in premise.neighbourhood:
        #         ax.plot([premise.coordinates[0], property_coordinates[farm[0], 0]], [premise.coordinates[1], property_coordinates[farm[0], 1]], alpha = 0.23, color = 'black')
        #     plt.title('Wind dispersal neighbourhood \n Time = ' + str(time))

        # if premise.vaccination_status:
        #     # vax_colour = '#2c7bbb'
        #     # vax_colour = '#edf8fb'
        #     ax.scatter(premise.coordinates[0], premise.coordinates[1], color = vax_colour, label = 'vaccinated')
        #     circle = Circle((premise.coordinates[0], premise.coordinates[1]), premise.radius, alpha = 1, color = vax_colour)
        #     ax.add_patch(circle)
        #     # circle_ring = Circle((premise.coordinates[0], premise.coordinates[1]), params['r_ring_action'], alpha = 0.3, color = vax_colour)
        #     # ax.add_patch(circle_ring)


        if premise.reported_status: 
            if not premise.culled_status: 
                # reported_colour = '#fdae61'
                # reported_colour = '#2ca25f'
                
                ax.scatter(premise.coordinates[0], premise.coordinates[1], color = reported_colour, label = 'reported')
                circle = Circle((premise.coordinates[0], premise.coordinates[1]), premise.radius, alpha = 1, color = reported_colour)
                ax.add_patch(circle)
                
            if premise.culled_status and premise.vaccination_status:
                # culled_colour = '#d7191c'
                # culled_colour = '#006d2c'
                
                ax.scatter(premise.coordinates[0], premise.coordinates[1], color = culled_colour, label = 'culled')
                circle = Circle((premise.coordinates[0], premise.coordinates[1]), premise.radius, alpha = 1, color = vax_and_culled_colour)
                ax.add_patch(circle)


            elif premise.culled_status:
                # culled_colour = '#d7191c'
                # culled_colour = '#006d2c'
                
                ax.scatter(premise.coordinates[0], premise.coordinates[1], color = culled_colour, label = 'culled')
                circle = Circle((premise.coordinates[0], premise.coordinates[1]), premise.radius, alpha = 1, color = culled_colour)
                ax.add_patch(circle)
                

        elif premise.testing_status: 
            ax.scatter(premise.coordinates[0], premise.coordinates[1], color = infection_colour, label = 'testing')
            circle = Circle((premise.coordinates[0], premise.coordinates[1]), premise.radius, alpha = 1, color = testing_colour)
            ax.add_patch(circle)
            # circle_ring = Circle((premise.coordinates[0], premise.coordinates[1]), params['r_ring_action'], alpha = 0.3, color = testing_colour)
            # ax.add_patch(circle_ring)

        elif premise.vaccination_status:
            # vax_colour = '#2c7bbb'
            # vax_colour = '#edf8fb'
            ax.scatter(premise.coordinates[0], premise.coordinates[1], color = vax_colour, label = 'vaccinated')
            circle = Circle((premise.coordinates[0], premise.coordinates[1]), premise.radius, alpha = 1, color = vax_colour)
            ax.add_patch(circle)
            # circle_ring = Circle((premise.coordinates[0], premise.coordinates[1]), params['r_ring_action'], alpha = 0.3, color = vax_colour)
            # ax.add_patch(circle_ring)
            
        elif premise.infection_status:
            # infection_colour = '#ffffbf'
            # infection_colour = '#66c2a4'
            ax.scatter(premise.coordinates[0], premise.coordinates[1], color = infection_colour, label = 'infected')
            circle = Circle((premise.coordinates[0], premise.coordinates[1]), premise.radius, alpha = 1, color = infection_colour)
            ax.add_patch(circle)
            # circle_ring = Circle((premise.coordinates[0], premise.coordinates[1]), params['r_ring_action'], alpha = 0.3, color = infection_colour)
            # ax.add_patch(circle_ring)
        

            
        # elif premise.vaccination_status:
        #     # vax_colour = '#2c7bbb'
        #     # vax_colour = '#edf8fb'
        #     ax.scatter(premise.coordinates[0], premise.coordinates[1], color = vax_colour, label = 'vaccinated')
        #     circle = Circle((premise.coordinates[0], premise.coordinates[1]), premise.radius, alpha = 1, color = vax_colour)
        #     ax.add_patch(circle)

        else:
            # sus_colour = '#abd9ed'
            # sus_colour = '#b2e2e2'
            ax.scatter(premise.coordinates[0], premise.coordinates[1], color = sus_colour, label = 'susceptible')
            circle = Circle((premise.coordinates[0], premise.coordinates[1]), premise.radius, alpha = 1, color = sus_colour)
            ax.add_patch(circle)
            # circle_ring = Circle((premise.coordinates[0], premise.coordinates[1]), params['r_ring_action'], alpha = 0.3, color = sus_colour)
            # ax.add_patch(circle_ring)

    


    
    plt.axis('equal')
    plt.xlim(-100, 1100)
    plt.ylim(-100, 1100)
    plt.xticks([])
    plt.yticks([])

    if time < 10:
        file_name = "00" + str(time) + ".svg"
    elif time < 100:
        file_name = "0" + str(time) + ".svg"
    else: 
        file_name = str(time) + ".svg"
    plt.savefig(file_name)
    return

def make_video():
    # current_dir = os.getcwd()
    # parent_dir = os.path.dirname(current_dir)

    image_files = [os.path.join(os.getcwd(), img) for img in sorted(os.listdir(os.getcwd())) if img.endswith(('png'))]

    # file_path = str(parent_dir) + "*.eps"
    fps = 2
    clip = ImageSequenceClip(image_files, fps = fps)
    output_file = 'plot_video.mp4'
    clip.write_videofile(output_file)
    # ffmpeg.input(file_path, pattern_type = 'glob', framerate = 1).output('plot.mp4').run()
    return 

def triplet_plot(file_1, file_2, file_3):
    plot_1 = cv.imread(file_1)
    plot_2 = cv.imread(file_2)
    plot_3 = cv.imread(file_3)

    plt.subplot(1, 3, 1)
    plt.imshow(plot_1)
    
    plt.subplot(1, 3, 2)
    plt.imshow(plot_2)

    plt.subplot(1, 3, 3)
    plt.imshow(plot_3)

    plt.show()

    plt.savefig('triplet_plot.png')

    return 