#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from exp_setup import *

from collect_results import read_in_results

import itertools



""" Form boxplots of the results. Bars grouped according to period. In each period group we have bars for the number of redundant copies """

inch_size = (4.5,3)

def get_overhead(x, ref):
    return 100 * (x-ref) / ref


class BoxDesign:
    def __init__(self, color):
        if color == "blue":
            self.boxprops = dict(color='black', facecolor='lightblue')
            self.flierprops = dict(marker='', markerfacecolor='lightblue')
            self.medianprops = dict(linewidth=1.5, color='darkblue')
            self.whiskerprops = dict(color='black')
        elif color == "red":
            self.boxprops = dict(facecolor='orange')
            self.flierprops = dict(marker='', markerfacecolor='orange')
            self.medianprops = dict(linewidth=1.5, color='firebrick')
            self.whiskerprops = dict(color='black')
        elif color == "green":
            self.boxprops = dict(color='black', facecolor='lightgreen')
            self.flierprops = dict(marker='', markerfacecolor='lightgreen')
            self.medianprops = dict(linewidth=1.5, color='darkgreen')
            self.whiskerprops = dict(color='black')
        else:
            raise ValueError("Unknown color scheme")


class PlotParameters:
    def __init__(self, copies_length):
        self.box_width = 1
        self.strategy_separation=0.4
        self.period_separation=4

        self.period_width = copies_length*self.box_width * 2 + self.strategy_separation + self.period_separation

        self.strategy_width = copies_length * self.box_width

    def compute_loc(self, i_period):
        return 0.5*self.period_separation + i_period*self.period_width



def make_six_boxes(fig, i_period, esrp, cr, parameters,\
                   esrp_design = BoxDesign("blue"), cr_design = BoxDesign("red")):

    """ To create plots as suggested by WG """

    loc = parameters.compute_loc(i_period)

    artists = []

    a = plt.boxplot(esrp,
                    positions = [loc + (0.5+x) * parameters.box_width for x in range(len(copies_list))],
                    widths = parameters.box_width,
                    boxprops    = esrp_design.boxprops,
                    flierprops  = esrp_design.flierprops,
                    medianprops = esrp_design.medianprops,
                    whiskerprops= esrp_design.whiskerprops,
                    patch_artist = True)

    artists.append(a)

    if cr is not None and len(cr) > 0:
        a = plt.boxplot(cr,
                        positions = [loc + parameters.strategy_width + \
                                     parameters.strategy_separation + \
                                     (0.5+x) * parameters.box_width for x in range(len(copies_list))],
                        widths = parameters.box_width,
                        boxprops    = cr_design.boxprops,
                        flierprops  = cr_design.flierprops,
                        medianprops = cr_design.medianprops,
                        whiskerprops= cr_design.whiskerprops,
                        patch_artist = True)
        artists.append(a)

    return artists


parameters = PlotParameters(len(copies_list))

situation_list = ["undisturbed","failures"]

def make_matrix_plots(matrix_results, matrix):

    strategies = strategy_dict.keys()

    reference_time, reference_std = \
            get_runtime_from_results_list(matrix_results["reference"], last=9)
    for situation in situation_list:

        fig = plt.figure()

        period_list_adj = period_list

        reference_band_limits = [0,len(period_list_adj)*parameters.period_width]
        ref_stdv_adjusted = 100 * reference_std / reference_time
        plt.fill_between(x=reference_band_limits,\
                         y1=[-ref_stdv_adjusted, -ref_stdv_adjusted],\
                         y2=[ref_stdv_adjusted, ref_stdv_adjusted],\
                         color="gray", alpha=0.25)

        for j,period in enumerate(period_list_adj):
            # Each period has a bar group


            boxes = []

            for i,strategy in enumerate(strategies):

                boxes.append([])

                if strategy == "checkpoint" and period == 0:
                    continue

                for k,copies in enumerate(copies_list):
                    # Each number of copies has two bars (undisturbed and failures)

                    # Each number of copies has an iteration and a position,
                    # but these will be averaged. We can't show every detail

                    if( situation == "undisturbed" ):
                        # Average values for undisturbed case
                        undisturbed_list = \
                            [x.runtime for x in matrix_results[strategy].failure_free[period][copies]]
                        undisturbed_overhead = \
                            [get_overhead(x,reference_time) for x in undisturbed_list]

                        boxes[i].append(undisturbed_overhead)

                    else:

                        if period == 0:
                            progress = progress_list[0]
                        else:
                            progress = progress_list[1]

                        # Average values with failures
                        failure_list_of_lists = \
                            [result_list
                            for result_list in [matrix_results[strategy].failures[period][broken][progress]
                            for broken
                            in break_transposition[copies]]]

                        failure_runtimes = [result.runtime for result in
                            itertools.chain.from_iterable(failure_list_of_lists)]

                        failure_overhead = \
                            [get_overhead(x, reference_time) for x in failure_runtimes]

                        boxes[i].append(failure_overhead)

                    # this completes the data for two boxes.


            artists = make_six_boxes(fig,  j, boxes[0], boxes[1], parameters)


        # Decoration and labeling

        plt.gca().set_xticks([(i+0.5)*parameters.period_width for i in range(4)])
        plt.gca().set_xticklabels(["$T = {}$".format(x) for x in period_list_adj])

        plt.xlim(0.25 * parameters.period_separation, \
                 parameters.period_width * len(period_list_adj) - 0.25 * parameters.period_separation)

        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.gca().grid(axis='y')

        plt.gca().legend([artists[0]["boxes"][0], artists[1]["boxes"][0]], \
                         [strategy_dict[x] for x in strategy_dict.keys()], \
                         loc = "upper right")

        plt.gca().set_ylabel("runtime overhead")
        plt.gca().set_xlabel("checkpointing interval")

        plt.tight_layout()
        plt.subplots_adjust(left=0.18, bottom=0.15)

        fig.set_size_inches(inch_size)
        plt.savefig("/tmp/{}_{}_boxplot.pdf".format(matrix, situation))
        plt.savefig("/tmp/{}_{}_boxplot.pgf".format(matrix, situation))


all_results = read_in_results(sys.argv[1])

for matrix in all_results.keys():

    # import pdb; pdb.set_trace()

    make_matrix_plots(all_results[matrix],matrix)
