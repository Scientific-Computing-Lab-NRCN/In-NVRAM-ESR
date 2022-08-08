#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from exp_setup import *

from collect_results import read_in_results

import itertools



""" Form boxplots of the results. Bars grouped according to period. In each period group we have bars for the number of redundant copies """

inch_size = (9,3)

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
        self.copies_separation=0.4
        self.period_separation=6

        self.setting_group_width = 2*self.box_width + self.copies_separation
        self.copy_group_width = copies_length * self.setting_group_width + self.period_separation - self.copies_separation
        self.unpadded_copy_group_width = copies_length * self.setting_group_width - self.copies_separation

    def compute_loc(self, i_copy, i_period):
        return (i_period * self.copy_group_width) + 0.5 * self.period_separation + i_copy * self.setting_group_width



def make_two_boxes(ax, i_copies, i_period, undisturbed_overhead, failure_overhead, parameters, undisturbed_design = BoxDesign("blue"), failure_design = BoxDesign("red")):

    loc = parameters.compute_loc(i_copies, i_period)

    ax.boxplot(undisturbed_overhead,
               positions = [loc + 0.5 * parameters.box_width],
               widths=parameters.box_width,
               boxprops    = undisturbed_design.boxprops,
               flierprops  = undisturbed_design.flierprops,
               medianprops = undisturbed_design.medianprops,
               whiskerprops= undisturbed_design.whiskerprops,
               patch_artist = True)

    ax.boxplot(failure_overhead,
               positions = [loc + 1.5 * parameters.box_width],
               widths=parameters.box_width,
               boxprops    = failure_design.boxprops,
               flierprops  = failure_design.flierprops,
               medianprops = failure_design.medianprops,
               whiskerprops= failure_design.whiskerprops,
               patch_artist = True)


parameters = PlotParameters(len(copies_list))

def make_matrix_plot(matrix_results, matrix):

    fig = plt.figure()
    gridspec = fig.add_gridspec(ncols=2,nrows=1,width_ratios=[0.57,0.43])
    axs = []

    strategies = strategy_dict.keys()

    reference_time, reference_std = \
            get_runtime_from_results_list(matrix_results["reference"])

    for i,strategy in enumerate(strategies):
        # Each strategy gets its own subplot

        axs.append(fig.add_subplot(gridspec[0,i]))

        period_list_adj = [x for x in period_list]
        if strategy == "inplace":
            period_list_adj = [0] + period_list_adj

        for j,period in enumerate(period_list_adj):
            # Each period has a bar group

            if period == 0:
                progress_list_adj = [progress_list[0]]
            else:
                progress_list_adj = progress_list


            for k,copies in enumerate(copies_list):
                # Each number of copies has two bars (undisturbed and failures)

                # Each number of copies has an iteration and a position,
                # but these will be averaged. We can't show every detail

                # Average values for undisturbed case
                undisturbed_list = \
                    [x.runtime for x in matrix_results[strategy].failure_free[period][copies]]
                undisturbed_overhead = \
                    [get_overhead(x,reference_time) for x in undisturbed_list]

                # Average values with failures
                failure_list_of_lists = \
                    [result_list
                    for result_list in [matrix_results[strategy].failures[period][broken][progress]
                    for progress,broken
                    in itertools.product(progress_list_adj,break_transposition[copies])]]

                failure_runtimes = [result.runtime for result in
                    itertools.chain.from_iterable(failure_list_of_lists)]

                failure_overhead = \
                    [get_overhead(x, reference_time) for x in failure_runtimes]

                # this completes the data for two boxes.


                make_two_boxes(axs[i], k, j, undisturbed_overhead, failure_overhead, parameters)


    # Decoration and labeling

    axs[0].set_xticks([(i+0.5)*parameters.copy_group_width for i in range(4)])
    axs[0].set_xticklabels(["$T = {}$".format(x) for x in [0] + period_list])

    axs[1].set_xticks([(i+0.5)*parameters.copy_group_width for i in range(3)])
    axs[1].set_xticklabels(["$T = {}$".format(x) for x in period_list])


    # Get the ylimits of both plots and select the maximum range
    ylims4 = [x for x in itertools.chain( axs[0].get_ylim(), axs[1].get_ylim() )]
    ylim = [min(ylims4), max(ylims4)]

    axs[0].set_ylim(ylim[0], ylim[1])
    axs[1].set_ylim(ylim[0], ylim[1])

    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter())

    axs[1].set_yticklabels([""]*len(axs[1].get_yticks()))

    axs[0].grid(axis='y')
    axs[1].grid(axis='y')

    axs[0].set_ylabel("Runtime overhead")
    axs[0].set_xlabel("Period")
    axs[1].set_xlabel("Period")

    axs[0].title.set_text("ESRP")
    axs[1].title.set_text("In-memory CR")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, left=0.1, bottom=0.15)

    fig.set_size_inches(inch_size)
    plt.savefig("/tmp/{}_boxplot.pdf".format(matrix))
    plt.savefig("/tmp/{}_boxplot.pgf".format(matrix))


all_results = read_in_results(sys.argv[1])

for matrix in all_results.keys():

    # import pdb; pdb.set_trace()

    make_matrix_plot(all_results[matrix],matrix)
