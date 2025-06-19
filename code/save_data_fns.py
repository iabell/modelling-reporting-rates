from collections import namedtuple
import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
from datetime import datetime 

def save_data(data, param_range, objectives, strategy_names):
    # data = dictionary of objective for each sim across all strategies

    # date for naming files 
    now = datetime.now()
    date = now.strftime('%d_%m_%Y')

    # create folder in which to save plots 
    plotting_dir = Path('results/' + date)
    plotting_dir.mkdir(parents = True, exist_ok = True)

    # new file for each objective
    for o, objective in enumerate(objectives):

        # file names for objective and all strategies
        filenames = ['results/' + date + '/' + s + '_' + objective + '.csv' for s in strategy_names]

        for i, strategy in enumerate(strategy_names):
            # loop over strategies to save data in separate csvs 
            filename = filenames[i]

            with open(filename, 'w', newline = '') as file:
                writer = csv.writer(file)
                for r in param_range:
                    # each row new param value
                    # each column in row, new simulation 
                    writer.writerow(data[o][str(r)][strategy])
    return

def read_all_data(objectives, strategy_names, param_range, date):
    output_to_df = []

    # for each objective
    for objective in objectives:

        # for each strategy 
        for strategy in strategy_names:

            # define filename strategy_objective.csv
            filename = 'results/' + date + '/' + strategy + '_' + objective + '.csv' 

            with open(filename, newline = '', mode = 'r') as f:
                reader = csv.reader(f, quoting = csv.QUOTE_NONNUMERIC)
                for param_index, row in enumerate(reader):
                    r = str(param_range[param_index])
                    for sim_index, sim_output in enumerate(row):
                        # put into dataframe
                        output_to_df.append([r, objective, strategy, sim_output, sim_index])

    output_df = pd.DataFrame(output_to_df)
    output_df.columns = ['r', 'objective', 'strategy', 'output', 'sim_index']
    return output_df

def remove_died_out_sims(df):
    num_removed = 0
    remove_vals = []
    remove_indices = []
    # filter for died out simulations, first report date = -1
    filtered_df = df[(df['objective']=='first_report_date') & (df['output']== -1)]

    # find r val and simulation index for died out sims
    for index, row in filtered_df.iterrows():
        remove_vals.append([row['r'], row['sim_index']])

    # count how many simulations died out
    num_removed = len(remove_vals)


    # remove simulations from original df
    for li in remove_vals:
        indices = list(df[(df['r'] == li[0]) & (df['sim_index'] == li[1])].index)
        remove_indices.append(indices)

    for el in remove_indices:
        for val in el:
            df.drop(val, inplace = True)

    print('I removed ' + str(num_removed) +' simulation instances.')

    return df

def work_out_optimal_and_diff(df, total_sims, param_range, objectives):
    optimal_data_frame_list = []
    # print(df['sim_index'].tolist())
    for s in tqdm(list(set(df['sim_index'].tolist())), desc = 'seeds'):
        # print(s)
        for r in param_range:
            for objective in objectives:
                # dataframe for desired output (objective) of all strategies given r value and simulation index 
                filtered_df = df[(df['r'] == str(r)) & (df['sim_index']== s) & (df['objective']== objective)]

                if filtered_df.empty: #simulation was removed
                    break

                # find row with minimum objective value 
                min_objective_strategy_row = filtered_df.nsmallest(1, 'output')

                # check if there are multiple optimal strategies
                min_output = filtered_df['output'].min()

                count_optimal_strats = (filtered_df['output'] == min_output).sum()
                
                if count_optimal_strats > 1:
                    optimal_strategy = 'multi'

                else: #if there's only one optimal strategy
                    # find strategy for minimum objective row 
                    optimal_strategy = min_objective_strategy_row.iloc[0]['strategy']

                # find the difference between the output of each strategy and the ring culling strategy 
                culling_output_df = filtered_df[filtered_df['strategy'] == 'ring_culling']
                culling_output = int(culling_output_df['output'])
                other_strat_df = filtered_df[filtered_df['strategy'] != 'ring_culling']
                
                # calculating the difference to ring culling
                for index, row in other_strat_df.iterrows():
                    strategy = row['strategy']
                    strategy_output = int(row['output'])
                    difference = int(strategy_output - culling_output)
                    # update output dataframe 
                    optimal_data_frame_list.append([r, s, optimal_strategy, objective, culling_output, strategy, difference])

    # output to dataframe 
    optimal_data_frame = pd.DataFrame(optimal_data_frame_list)
    optimal_data_frame.columns = ['param value', 'sim index', 'optimal strategy', 'objective', 'culling output', 'strategy', 'difference']
        
    return optimal_data_frame


def read_data_from_file(data_file, param_range, total_sims):

    output= []
    max_outputs = []
    output_to_df = []

    # read data from csv
    with open(data_file, newline = '') as f:
        reader = csv.reader(f, quoting = csv.QUOTE_NONNUMERIC)
        for row in reader:
            output.append(row)
            max_outputs.append(max(row))

    max_value = max(max_outputs)
    
    # put data in nice format for plotting
    for index, row in enumerate(output):
        for element in row:
            output_to_df.append(['r = ' + str(param_range[index]), element])

    # convert to dataframe 
    df = pd.DataFrame(output_to_df, columns = ['param value', 'output'])

    return df


def plotting_individual_strategies(df, strategies_in_df, outputs_in_df, strategy_names, output_names, colour_chart):
    # datetime for naming files 
    now = datetime.now()
    date = now.strftime('%d_%m_%Y_')

    for i, output in enumerate(outputs_in_df):
        df_filtered = df[df['objective'] == output]
        max_ylim = df_filtered['output'].max()
        for j, strat in enumerate(strategies_in_df): 
            filename = 'plotting/' + date + strat + '_'+ output + '.png'
            plt.figure()
            df_filtered_strat = df_filtered[df_filtered['strategy'] == strat]

            sns.boxplot(data = df_filtered_strat, x = 'r', y = 'output', hue = 'strategy', palette = colour_chart)
            handles, labels = plt.gca().get_legend_handles_labels()

            # # Manually change the legend labels
            new_labels = [strategy_names[j]]
            new_handles = [handles[0]]

            # # Update the legend
            plt.legend(handles = new_handles, labels = new_labels, bbox_to_anchor=(1, 0.5))
            
            # fix axis labels 
            # plt.legend(strategy_names[j])
            plt.xlabel('Probability of reporting (r)')
            plt.ylabel(output_names[i])
            plt.ylim(0, max_ylim+5)
            plt.tight_layout()
            
            plt.savefig(filename)

    return 

def individual_strategies_multiplot(df, strategies_in_df, outputs_in_df, strategy_names, output_names, colour_chart_dict, colour_chart_list):
    now = datetime.now()
    date = now.strftime('%d_%m_%Y')

    # create folder in which to save plots 
    plotting_dir = Path('plotting/' + date)
    plotting_dir.mkdir(parents = True, exist_ok = True)

    font_size = 18
    title_size = font_size + 2
    tick_size = font_size - 2
    plt.rcParams['font.size'] = font_size


    # 1 objective, 4 strategies
    for obj_index, obj in enumerate(outputs_in_df):
        # define filenames for png and eps files (png easier to immediately view)
        filename = 'plotting/' + date + '/' + outputs_in_df[obj_index] + '_all_strats'
        filename_png = filename + '.png'
        filename_eps = filename + '.eps'

        # filter dataframe for each objective
        df_filtered = df[df['objective'] == obj]

        # finding consistent ylim for subplots
        max_ylim = df_filtered['output'].max()

        # define figure 
        fig, axs = plt.subplots(2, 2, figsize = (12, 10))

        titles = ['(a)', '(b)', '(c)', '(d)']

        # loop over strategies 
        for strat_index, strat in enumerate(strategies_in_df):
            # filter dataframe for each strategy 
            df_filtered_strat = df_filtered[df_filtered['strategy'] == strat]

            # plot indices
            i_s = int(np.floor(strat_index/2))
            j_s = int(strat_index % 2)

            # draw plot
            sns.boxplot(data = df_filtered_strat, x = 'r', y = 'output', hue = 'strategy', palette = colour_chart_dict, ax = axs[i_s, j_s], legend = False)
            
            # remove x and y labels from subplots 
            axs[i_s,j_s].set_xlabel('')
            axs[i_s,j_s].set_ylabel('')
            
            # consistent y axis across subplots
            axs[i_s,j_s].set_ylim(-1, max_ylim+3)

            axs[i_s,j_s].set_title(titles[strat_index], fontsize = title_size)

            axs[i_s,j_s].tick_params(axis = 'x', labelsize = tick_size)
            axs[i_s,j_s].tick_params(axis = 'y', labelsize = tick_size)
        

        # axis labels for whole subfigure 
        fig.supxlabel('Farmer reporting rate (r)', fontsize = font_size)
        fig.supylabel(output_names[obj_index], fontsize = font_size)
        

        # manually create legend for entire subfigure 
        legend_labels = [mpatches.Patch(color = colour, label = strategy_names[i]) for i, colour in enumerate(colour_chart_list[:-1])]
        fig.legend(handles = legend_labels, loc = 'upper center', bbox_to_anchor = (1.1, 0.6), fontsize = tick_size)

        # making sure everything shows up on the plot 
        plt.subplots_adjust(right = 0.78)
        plt.tight_layout()

        # saving plot 
        plt.savefig(filename_eps, bbox_inches = 'tight')
        plt.savefig(filename_png, bbox_inches = 'tight')
    
    # 1 strategy, 4 objectives
    for strat_index, strat in enumerate(strategies_in_df):
        # define filenames for png and eps files (png easier to immediately view)
        filename = 'plotting/' + date + '/' + strategies_in_df[strat_index] + '_all_objs'
        filename_png = filename + '.png'
        filename_eps = filename + '.eps'

        # filter dataframe for each strategy
        df_filtered = df[df['strategy'] == strat]

        # finding consistent ylim for subplots
        max_ylim = df_filtered['output'].max()

        # define figure 
        fig, axs = plt.subplots(2, 2, figsize = (12, 8))

        # loop over strategies 
        for obj_index, obj in enumerate(outputs_in_df):
            # filter dataframe for each strategy 
            df_filtered_obj = df_filtered[df_filtered['objective'] == obj]

            # plot indices
            i_o = int(np.floor(obj_index/2))
            j_o = int(obj_index % 2)

            # draw plot
            sns.boxplot(data = df_filtered_obj, x = 'r', y = 'output', hue = 'strategy', palette = colour_chart_dict, ax = axs[i_o, j_o], legend = False)
            
            # remove x and y labels from subplots 
            axs[i_o, j_o].set_xlabel('')
            axs[i_o, j_o].set_ylabel(output_names[obj_index])
            
        
        # axis labels for whole subfigure 
        fig.supxlabel('Farmer reporting rate (r)')

        # manually create legend for entire subfigure 
        legend_labels = [mpatches.Patch(color = colour_chart_list[strat_index], label = strategy_names[strat_index])]
        fig.legend(handles = legend_labels, loc = 'upper center', bbox_to_anchor = (0.9, 0.5))

        # making sure everything shows up on the plot 
        # plt.subplots_adjust(right = 0.8)

        plt.tight_layout()

        # saving plot 
        plt.savefig(filename_eps)
        plt.savefig(filename_png)

    # for i, output in enumerate(outputs_in_df):
        
    #     max_ylim = df_filtered['output'].max()
    #     for j, strat in enumerate(strategies_in_df): 
    #         filename = 'plotting/' + date + strat + '_'+ output + '.png'
    #         plt.figure()
            

    #         sns.boxplot(data = df_filtered_strat, x = 'r', y = 'output', hue = 'strategy', palette = colour_chart)
    #         handles, labels = plt.gca().get_legend_handles_labels()

    #         # # Manually change the legend labels
    #         new_labels = [strategy_names[j]]
    #         new_handles = [handles[0]]

    #         # # Update the legend
    #         plt.legend(handles = new_handles, labels = new_labels, bbox_to_anchor=(1, 0.5))
            
    #         # fix axis labels 
    #         # plt.legend(strategy_names[j])
    #         plt.xlabel('Probability of reporting (r)')
    #         plt.ylabel(output_names[i])
    #         plt.ylim(0, max_ylim+5)
    #         plt.tight_layout()
            
    #         plt.savefig(filename)
    
    
    return 

def plot_reporting_stats(df, strategies, outputs, strategy_names, output_names, colour_chart_dict, colour_chart_list):
    now = datetime.now()
    date = now.strftime('%d_%m_%Y')

    # create folder in which to save plots 
    plotting_dir = Path('plotting/' + date)
    plotting_dir.mkdir(parents = True, exist_ok = True)

    for strat_index, strat in enumerate(strategies):
        # define filenames for png and eps files (png easier to immediately view)
        filename = 'plotting/' + date + '/' + strat + '_reporting'
        filename_png = filename + '.png'
        filename_eps = filename + '.eps'

        # filter dataframe for each strategy
        df_filtered = df[df['strategy'] == strat]

        # finding consistent ylim for subplots
        # max_ylim = df_filtered['output'].max()

        # define figure 
        fig, axs = plt.subplots(1, 1, figsize = (12, 8))

        # loop over strategies 
        for obj_index, obj in enumerate(outputs):
            # filter dataframe for each strategy 
            df_filtered_obj = df_filtered[df_filtered['objective'] == obj]

            # draw plot
            sns.boxplot(data = df_filtered_obj, x = 'r', y = 'output', hue = 'strategy', palette = colour_chart_dict, legend = False)
            
            # remove x and y labels from subplots 
            axs.set_xlabel('Probability of reporting infection (r)')
            axs.set_ylabel(output_names[obj_index])

            axs.set_ylim(0)

            # manually create legend 
            legend_labels = [mpatches.Patch(color = colour_chart_list[strat_index], label = strategy_names[strat_index])]
            fig.legend(handles = legend_labels, loc = 'upper right', bbox_to_anchor = (0.9,0.87))
            # fig.legend(handles = legend_labels, loc = 'upper center', bbox_to_anchor = (0.9, 0.5))

            # saving plot 
            plt.savefig(filename_eps)
            plt.savefig(filename_png)

        fig, axs = plt.subplots(1, 1, figsize = (12, 8))
    
    fig, axs = plt.subplots(2, 1, figsize = (12, 8))
    # specific plot for paper 
    # filter dataframe for each strategy 
    df_filtered = df[df['strategy'] == 'ring_culling']
    df_filtered_obj = df_filtered[df_filtered['objective'] == 'first_report_date']

    # draw plot
    # need to design new colour plot 
    palette = sns.color_palette("hls", 10)
    palette_dict = {'0.1': palette[0],
                    '0.2': palette[1],
                    '0.3': palette[2],
                    '0.4': palette[3],
                    '0.5': palette[4],
                    '0.6': palette[5],
                    '0.7': palette[6],
                    '0.8': palette[7],
                    '0.9': palette[8],
                    '1': palette[9]
    }

    sns.boxplot(data = df_filtered_obj, x = 'output', y = 'r', hue = 'r', palette = palette_dict, legend = False, ax = axs[0])
    
    df_filtered_obj_filtered = df_filtered_obj[df_filtered_obj['r'].isin(['0.1', '0.2', '0.5', '0.9', '1'])]

    sns.ecdfplot(data = df_filtered_obj_filtered, x = 'output', hue = 'r', ax = axs[1], palette = palette_dict)
    
    # set default font size
    font_size = 14
    title_size = font_size + 2
    tick_font_size = font_size -2
    plt.rcParams['font.size'] = font_size
    
    # remove x and y labels from subplots 
    axs[0].set_xlabel('First report day', fontsize = font_size)
    axs[1].set_xlabel('First report day', fontsize = font_size)

    axs[0].set_title('(a)', fontsize = title_size)
    axs[1].set_title('(b)', fontsize = title_size)

    axs[0].set_ylabel('Farmer reporting rate (r)', fontsize = font_size)
    axs[1].set_ylabel('Proportion of simulations', fontsize = font_size)
    axs[1].set_ylim((0, 1.05))

    axs[0].tick_params(axis = 'y', labelsize = tick_font_size)
    axs[1].tick_params(axis = 'y', labelsize = tick_font_size)
    axs[0].tick_params(axis = 'x', labelsize = tick_font_size)
    axs[1].tick_params(axis = 'x', labelsize = tick_font_size)

    # axs.set_xlim((0, 25))
    axs[0].invert_yaxis()
    
    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(labels = ['r = 1', 'r = 0.9', 'r = 0.5', 'r = 0.2', 'r = 0.1'])

    # manually create legend 
    # legend_labels = [mpatches.Patch(color = colour_chart_list[strat_index], label = strategy_names[strat_index])]
    # fig.legend(handles = legend_labels, loc = 'upper right', bbox_to_anchor = (0.9,0.87))
    # fig.legend(handles = legend_labels, loc = 'upper center', bbox_to_anchor = (0.9, 0.5))

    # plt.show()
    plt.tight_layout()

    # saving plot 
    plt.savefig('plotting/' + date + '/'+ 'ring-culling-first-report-multiplot.eps')
    plt.savefig('plotting/' + date + '/' + 'ring-culling-first-report-multiplot.png')


    fig, axs = plt.subplots(2, 1, figsize = (12, 8))
    





    return 




def plotting_optimal_percentages_and_difference(df, param_range, strategies, strategy_names, objectives, objective_names, colour_chart, colour_chart_list):
    # dataframe column headings: ['param value', 'sim index', 'optimal strategy', 'objective', 'culling output', 'strategy', 'difference']
    
    # date for naming files 
    now = datetime.now()
    date = now.strftime('%d_%m_%Y')

    # create folder in which to save plots 
    plotting_dir = Path('plotting/' + date)
    plotting_dir.mkdir(parents = True, exist_ok = True)

    for obj_index, obj in enumerate(objectives):
        #------------------------#
        # OPTIMAL PROPORTION PLOT
        #------------------------#  
        # define filenames for png and eps files (png easier to immediately view)
        filename_prop = 'plotting/' + date + '/' + obj + '_prop_optimal'
        filename_prop_png = filename_prop + '.png'
        filename_prop_eps = filename_prop + '.eps'
        filename_diff = 'plotting/' + date + '/' + obj + '_difference_plot'
        filename_diff_png = filename_diff + '.png'
        filename_diff_eps = filename_diff + '.eps'
        filename_diff_single = 'plotting/' + date + '/' + obj + '_difference_plot_single'
        filename_diff_single_png = filename_diff_single + '.png'
        filename_diff_single_eps = filename_diff_single + '.eps'
    
        # filtering dataframe to look at one objective at a time
        df_objective = df[df['objective'] == obj]

        # collecting data for plotting dataframe 
        strategy_props = []
        for i, r in enumerate(param_range):
            strategy_props.append([r])

            # find data for given r and just one strategy so we're not overcounting 
            df_objective_r = df_objective[df_objective['param value'] == r]

            # calculate proportions for each strategy 
            for strat in strategies:
                prop = (df_objective_r['optimal strategy'] == strat).sum() / len(df_objective_r)
                # print(prop)
                strategy_props[i].append(prop)
            
        # convert prop data to dataframe 
        prop_df = pd.DataFrame(strategy_props)
        prop_df.columns = ['param value', 'Ring culling', 'Ring testing', 'Ring vaccination\n(perfect)', 'Ring vaccination\n(imperfect)', 'Multiple']

        # save to csv to create table for paper 
        date = now.strftime('%d_%m_%Y')
        dir = Path('results/' + date)
        dir.mkdir(parents = True, exist_ok = True)     
        prop_df.to_csv('results/' + date + '/' + obj + '_prop_df.csv', index = False)
        # print(prop_df)
        
        # fix default font size 
        plt.rcParams['font.size'] = 10

        # draw plot
        prop_df.plot(kind = 'bar', stacked = True, color = colour_chart_list, linewidth = 0, x = 'param value')  

        # fix things on plot
        plt.xticks(rotation = 'horizontal')

        # move legend
        plt.legend(bbox_to_anchor = (1, 0.5), reverse = True)

        # axis labels
        plt.xlabel('Probability of reporting (r)')
        plt.ylabel('Proportion of simulations where strategy is optimal')
        plt.title(objective_names[obj_index])

        plt.tight_layout()
        # plt.show()
        plt.savefig(filename_prop_eps)
        plt.savefig(filename_prop_png)

        #------------------------#
        # DIFFERENCE PLOT
        #------------------------#  
        plt.figure(figsize = (12, 8), dpi = 100)
        font_size = 14
        title_size = font_size + 2
        tick_size = font_size -2
        plt.rcParams['font.size'] = 14


        # plt.subplot(4,1,1)
        sns.boxplot(data = df_objective, x = 'param value', y = 'difference', hue = 'strategy', palette = colour_chart, fill = True, gap = 0.5)

        # plot 0 horizontal line
        plt.axhline(y=0, linestyle = ':', linewidth=2, alpha=1)

        # Manually change legend 
        handles, labels = plt.gca().get_legend_handles_labels()
        

        # Update the legend
        plt.legend(handles = handles[:len(strategy_names)-1], labels = strategy_names[1:], bbox_to_anchor=(1, 0.561), fontsize = tick_size)
        print(strategy_names[1:])
        
        # fix axis labels 
        plt.xlabel('Probability of reporting (r)')
        plt.ylabel('Difference from ring culling strategy \n(' + objective_names[obj_index] + ')')

        plt.xticks(fontsize = tick_size)

        plt.tight_layout()
        plt.savefig(filename_diff_single_eps)
        plt.savefig(filename_diff_single_png)

        plt.figure(figsize = (20, 20), dpi = 100)
        plt.rcParams['font.size'] = 16

        # # multiplot
        # plt.subplot(4,1,1)
        # sns.boxplot(data = df_objective, x = 'param value', y = 'difference', hue = 'strategy', palette = colour_chart, fill = True, gap = 0.5)

        # # plot 0 horizontal line
        # plt.axhline(y=0, linestyle = ':', linewidth=2, alpha=1)

        # # Manually change legend 
        # handles, labels = plt.gca().get_legend_handles_labels()
        

        # # Update the legend
        # plt.legend(handles = handles[:len(strategy_names)-1], labels = strategy_names[1:], bbox_to_anchor=(1, 0.75))
        # print(strategy_names[1:])
        
        # # fix axis labels 
        # plt.xlabel('Probability of reporting (r)')
        # plt.ylabel('Difference from ring culling strategy \n(' + objective_names[obj_index] + ')')

        # # find ylim to align other plots 
        # y_limits = plt.gca().get_ylim()

        # # plotting individual strategies for clarity 
        # non_culling_strats = strategies[1:-1]
        # # use new_labels list for labels
        # for i, strat in enumerate(non_culling_strats):
        #     plt.subplot(4,1,i + 2)
        #     df_single_strat = df_objective[df_objective['strategy'] == strat]
        #     ax = sns.boxplot(data = df_single_strat, x = 'param value', y = 'difference', hue = 'strategy', palette = colour_chart, fill = True, gap = 0.5)

        #     # # set texture for plot
        #     # for bar in ax.patches:
        #     #     bar.set_hatch('x')

        #     # plot 0 horizontal line
        #     plt.axhline(y=0, linestyle = ':', linewidth=2, alpha=1)

        #     # change legend 
        #     handles, labels = plt.gca().get_legend_handles_labels()

        #     # Update the legend
        #     plt.legend(handles = handles[:len(strategy_names[i+1])], labels = [strategy_names[i+1]], bbox_to_anchor=(1, 0.5))

        #     # fix axis labels 
        #     plt.xlabel('Probability of reporting (r)')
        #     plt.ylabel('Difference from ring culling strategy \n(' + objective_names[obj_index] + ')')

        #     plt.ylim(y_limits)


        # plt.tight_layout()
        # # plt.show()
        # plt.savefig(filename_diff_eps)
        # plt.savefig(filename_diff_png)
    
    return 


