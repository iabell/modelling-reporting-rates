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


class Property:
    def __init__(self, params):
        # property characteristics
        self.coordinates = []
        self.radius = 0
        self.area = 0
        self.neighbourhood = []
        self.neighbourhood_ring_action = []
        self.total_neighbours = 0
        self.animals = []
        self.size = params['size']

        # infection characteristics
        # not-status = 0
        # status = 1
        self.infection_status = 0
        self.vaccination_status = 0
        self.culled_status = 0
        self.reported_status = 0 
        self.testing_status = 0
        self.number_times_tested = 0
        self.ring_action_status = 0

        # vaccination, culling delay trackers
        self.vaccination_delay_clock = 0
        self.culling_delay_clock = 0
        self.testing_delay_clock = 0

        # self.prop_infected = 0
        self.prop_infectious = 0
        self.prop_clinical = 0
        self.number_infectious = 0
        self.cumulative_infections = 0
        self.number_infected = 0

        self.movement_start_day = np.random.randint(0, params['movement_frequency'])

        return 

    # initialise all animals on the property 
    def init_animals(self, params):
        for _ in range(self.size):
            self.animals.append(Animal(params))
        return 
    
    # update infection counts
    def update_counts(self):
        number_infected = 0
        number_infectious = 0
        number_clinical = 0
        if not self.culled_status:
            for cow in self.animals:
                # check how many animals are infected (infection status of property) and how many are infectious (FOI calculation)
                if cow.infection_status == 'exposed':
                    number_infected += 1
                elif cow.infection_status == 'infectious':
                    number_infected += 1
                    number_infectious += 1
                
                # check how many animals are showing clinical symptoms (reporting)
                if cow.clinical_status == 'clinical':
                    number_clinical += 1
                
            # record proportion of animals infectious and clinical for other calculations 
            self.prop_infectious = number_infectious / len(self.animals)
            self.number_infectious = number_infectious
            self.prop_clinical = number_clinical / len(self.animals)

            # if there's any infection, property is labelled infected
            self.number_infected = number_infected
            if number_infected > 0:
                self.infection_status = 1
        return 

    # does the property vaccinate on its own?
    # def self_vaccination(self, params, properties):
    #     # only susceptible properties can vaccinate
    #     if not self.culled_status and not self.infection_status and not self.vaccination_status: 
            
    #         # if property hasn't decided to vaccinate
    #         if not self.vaccination_delay_clock:
            
    #             # calculate proportion of culled neighbours 
    #             prop_culled_neighbours = 0

    #             # if premise has neighbours 
    #             if len(self.neighbourhood): 
    #                 # how many neighbours?
    #                 neighbours = [el[0] for el in self.neighbourhood]

    #                 # how many culled neighbours?
    #                 culled_neighbours = sum([properties[i].culled_status for i in neighbours])

    #                 # proportion of neighbours culled
    #                 prop_culled_neighbours = culled_neighbours/self.total_neighbours 

    #             # if there are culled neighbours
    #             if prop_culled_neighbours: 
    #                 # will property vaccinate?
    #                 vaccinate_rand = np.random.rand()

    #                 # if property vaccinates
    #                 if vaccinate_rand < params['prob_vaccinate']*prop_culled_neighbours:

    #                     #is there a vaccination delay?
    #                     if params['vaccination_delay'] == 0: #no vaccination delay
    #                         self.vaccination_status = 1
    #                     else: 
    #                         self.vaccination_delay_clock = 1
            
    #         # property has decided to vaccinate
    #         else: 
    #             # increment delay clock 
    #             if self.vaccination_delay_clock < params['vaccination_delay']:
    #                 self.vaccination_delay_clock += 1
                
    #             # vaccinate property 
    #             else: 
    #                 self.vaccination_status = 1
    #     return
    
    def vaccination(self, params):
        # property is getting vaccinate but has not yet been vaccinated
        if self.vaccination_delay_clock:
            if self.vaccination_delay_clock < params['vaccination_delay']:
                self.vaccination_delay_clock += 1 

            else:
                self.vaccination_delay_clock = 0
                self.vaccination_status = 1

        return 

    # does the property report infection?
    def reporting(self, params, movement_flag, t, index, reporting_tracker):
        ring_action_flag = 0
        # if property has already been culled 
        if self.culled_status:
            return movement_flag, ring_action_flag, reporting_tracker
        
        # if haven't already reported and are infected
        elif not self.reported_status and self.infection_status:
            


            # only report if proportion of clinical cases is above the clinical reporting threshold 
            if self.prop_clinical > params['clinical_reporting_threshold']:

                # does the property report?
                reporting_rand = np.random.rand()
                chance_of_reporting = params['prob_report']*self.prop_clinical

                # if property reports
                if reporting_rand < chance_of_reporting:
                    reporting_tracker.append([t, index])
                    # mark property as reported
                    self.reported_status = 1
                    self.ring_action_status = 1

                    # ring culling notification at property report
                    ring_action_flag = 1
                    
                    # if no culling delay
                    if not params['culling_delay']:
                        self.cull_property()    
                    
                    # if culling delay
                    else: 
                        # start delay clock calculation 
                        self.culling_delay_clock = 1
                        
                        # movement restrictions at first report
                        if not movement_flag:
                            movement_flag = 1

        # if property has already reported 
        elif self.reported_status: 
            # increment delay clock if not time to cull 
            if self.culling_delay_clock < params['culling_delay']:
                self.culling_delay_clock += 1
            
            # if time to cull property 
            else: 
                self.cull_property()

        return movement_flag, ring_action_flag, reporting_tracker
    
    def ring_cull(self, params):
        # mark property as reported
        self.reported_status = 1

        # if no culling delay, cull straight away
        if not params['culling_delay']:
            self.cull_property()

        # if there's a culling delay, start culling clock
        else:
            self.culling_delay_clock = 1
        
        return 
    
    def ring_vaccinate(self, params):
        if self.culled_status or self.reported_status:
            return

        # if no vaccination delay, vaccinate straight away 
        if params['vaccination_delay']:
            self.vaccination_delay_clock = 1

        # if no vaccination delay, vaccinate immediately 
        else: 
            self.vaccination_delay_clock = 0
            self.vaccination_status = 1

        return 

    def ring_test(self, params):
        if not self.reported_status:
            self.testing_status = 1

            # start testing delay clock
            if params['testing_delay']:
                self.testing_delay_clock += 1

            # kick off testing
            else: 
                self.testing(params)

        return 

    def testing(self, params):
        # if property has been marked as tested
        if self.testing_status:
            if not self.reported_status:
                # increment delay clock 
                if self.testing_delay_clock < params['testing_delay']:
                    self.testing_delay_clock += 1
                
                # else test immediately 
                else:
                    # end testing wait period 
                    self.testing_status = 0 
                    # increment number of times tested
                    self.number_times_tested += 1

                    # reset testing clock in case property is tested more than once
                    self.testing_delay_clock = 0


                    # if property infected
                    if self.infection_status:
                        # kick off reporting and culling therefore culling by ring culling function
                        # i.e. testing is do ring culling in property is infected and nothing otherwise 
                        self.ring_cull(params)
            else:
                # if property has reported while waiting to be tested, reset testing status
                self.testing_status = 0
        return 

    
    def cull_property(self):
        # property not longer infected 
        self.infection_status = 0
        self.cumulative_infections = 0

        # property now culled
        # set all delay clocks to 0
        self.culling_delay_clock = 0
        self.testing_delay_clock = 0
        self.vaccination_delay_clock = 0

        # property culled
        self.culled_status = 1

        # no animals on property anymore 
        self.animals = []
    
        return 



    # infection model for an individual property
    def infection_model(self, params, FOI):
        infected_cases = 0
        clinical_cases = 0
        infectious_cases = 0

        # infection model for each animals
        for cow in self.animals:
            cow.check_transition(params)
            cow.update_clock()
            animal_inf = cow.infection_event(params, FOI) 
            if animal_inf:
                self.cumulative_infections += 1
        return 
    
    def infection_model_R0_calibration(self, params, FOI):
        infected_cases = 0

        for cow in self.animals: 
            cow.check_transition(params)
            cow.update_clock()
            animal_inf = cow.infection_event_R0_calibration(params, FOI)
            if animal_inf:
                infected_cases += 1
        return infected_cases


class Animal: 
    def __init__(self, params):
        # infection_status: 'susceptible', 'exposed', 'infectious', 'recovered', 'culled'
        # clinical_status: 'susceptible', 'pre-clinical', 'clinical', 'recovered', 'culled'
        self.infection_status = 'susceptible'
        self.clinical_status = 'susceptible'
        self.infection_clock = 0
        self.clinical_clock = 0

        return 


    def update_clock(self, dt = 1):
        # infection clock
        if self.infection_status == 'exposed' or self.infection_status == 'infectious':
            self.infection_clock += dt

        # clinical clock
        if self.clinical_status == 'pre-clinical' or self.infection_status == 'clinical':
            self.clinical_clock += dt
        return 
    

    def check_transition(self, params):
        # exposed -> infectious
        if self.infection_status == 'exposed' and self.infection_clock > params['latent_period']:
            self.infection_status = 'infectious'
        # infectious -> recovered
        elif self.infection_status == 'infectious' and self.infection_clock > params['latent_period'] + params['infectious_period']:
            self.infection_status = 'recovered'
        
        # pre-clinical -> clinical
        if self.clinical_status == 'pre-clinical' and self.clinical_clock > params['pre-clinical_period']:
            self.clinical_status = 'clinical'

        return 


    def infection_event(self, params, FOI):
        if self.infection_status == 'susceptible':
            infection_rand = np.random.rand()
            infection_prob = 1 - np.exp(-FOI)

            # if infection occurs
            if infection_rand < infection_prob:
                if params['latent_period'] != 0:
                    self.infection_status = 'exposed'
                else: 
                    self.infection_status = 'infectious'

                if params['pre-clinical_period'] != 0:
                    self.clinical_status = 'pre-clinical'
                else: 
                    self.clinical_status = 'clinical'
                # if infection event happens - return 1
                return 1
        # otherwise if no infection event - return 0
        return 0
    
    def infection_event_R0_calibration(self, params, FOI):
        if self.infection_status == 'susceptible':
            infection_rand = np.random.rand()
            infection_prob = 1 - np.exp(-FOI)

            # if infection occurs
            if infection_rand < infection_prob:
                # if infection event happens - return 1
                return 1
        # otherwise if no infection event - return 0
        return 0