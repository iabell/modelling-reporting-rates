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
# import seaborn as sns

from abm_fn import ABM

def sensitivity_analysis(param_name, param_range, params, grid_size, sim, property_radius):
    total_culled_all_sims = []
    total_vaccinated_all_sims = []

    for val in param_range:
        params[param_name] = val
        total_culled_list, total_vaccinated_list = multiple_simulations(sim, params, grid_size, property_radius)
        total_culled_all_sims.append(total_culled_list)
        total_vaccinated_all_sims.append(total_vaccinated_all_sims)

    return total_culled_all_sims, total_vaccinated_all_sims

def multiple_simulations(sim, params, grid_size, property_radius):
    total_culled_list = []
    total_vaccinated_list = []
    while sim >0:
        total_culled, total_vaccinated = ABM(params, 0, grid_size, property_radius)
        total_culled_list.append(total_culled)
        total_vaccinated_list.append(total_vaccinated)
        sim -= 1

    return total_culled_list, total_vaccinated_list