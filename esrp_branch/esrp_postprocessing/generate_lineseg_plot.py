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


class LineDesign:
    def __init__(self, color):
        self.kwargs={}

        self.kwargs["linewidth"]=1.0
        self.kwargs["linestyle"]="solid"
        self.kwargs["alpha"]=1.0
        self.kwargs["markersize"]=5
        if color == "blue":
            self.kwargs["color"]="blue"
            self.kwargs["marker"]="o"
        elif color == "red":
            self.kwargs["color"]="firebrick"
            self.kwargs["marker"]="^"
            self.kwargs["linestyle"]=":"
            self.kwargs["markersize"]=4
        elif color == "green":
            self.kwargs["color"]="green"
            self.kwargs["marker"]="s"
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



def make_lines(fig, i_period, esrp, cr, esr, parameters,
               copies_list,
               esrp_design = LineDesign("blue"),
               cr_design = LineDesign("red"),
               esr_design = LineDesign("green")
               ):

    artists = []

    loc = parameters.compute_loc(i_period)

    x = [loc + copies - 1 for copies in copies_list]

    artists.append( plt.semilogy(x, esrp, **esrp_design.kwargs) )
    artists.append( plt.semilogy(x, cr, **cr_design.kwargs) )
    artists.append( plt.semilogy(x, esr, **esr_design.kwargs) )

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

        # reference_band_limits = [0,len(period_list_adj)*parameters.period_width]
        # ref_stdv_adjusted = 100 * reference_std / reference_time
        # plt.fill_between(x=reference_band_limits,\
        #                  y1=[-ref_stdv_adjusted, -ref_stdv_adjusted],\
        #                  y2=[ref_stdv_adjusted, ref_stdv_adjusted],\
        #                  color="gray", alpha=0.25)


        esr_results = []
        for k,copies in enumerate(copies_list):
            if( situation == "undisturbed" ):
                esr_list = [x.runtime for x in matrix_results["inplace"].failure_free[0][copies]]
                esr_overhead = np.median( [get_overhead(x,reference_time) for x in esr_list] )
                esr_results.append( esr_overhead )
            else:
                progress = progress_list[0]
                failure_list_of_lists = \
                    [result_list
                    for result_list in [matrix_results["inplace"].failures[0][broken][progress]
                    for broken
                    in break_transposition[copies]]]

                failure_runtimes = [result.runtime for result in
                    itertools.chain.from_iterable(failure_list_of_lists)]

                failure_overhead = \
                    np.median([get_overhead(x, reference_time) for x in failure_runtimes])

                esr_results.append( failure_overhead )




        for j,period in enumerate(period_list_adj):
            # Each period has a bar group


            results = {}

            for i,strategy in enumerate(strategies):

                results[strategy] = []

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
                            np.median([get_overhead(x,reference_time) for x in undisturbed_list])

                        results[strategy].append(undisturbed_overhead)

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
                            np.median([get_overhead(x, reference_time) for x in failure_runtimes])

                        results[strategy].append(failure_overhead)

                    # this completes the data for two boxes.


            artists = make_lines(fig,  j, results["inplace"], results["checkpoint"], esr_results, parameters, copies_list)


        # Decoration and labeling

        plt.gca().set_xticks([(i+0.5)*parameters.period_width for i in range(4)])
        plt.gca().set_xticklabels(["$T = {}$".format(x) for x in period_list_adj])

        plt.xlim(0.25 * parameters.period_separation, \
                 parameters.period_width * len(period_list_adj) - 0.25 * parameters.period_separation)

        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.gca().grid(axis='y')

        labels = [strategy_dict[x] for x in strategy_dict.keys()]
        labels.insert(1, "ESR")
        if matrix == "Emilia_923" and situation == "failures":
            loc = "upper right"
        if matrix == "Emilia_923" and situation == "undisturbed":
            loc = "lower center"
        if matrix == "audikw_1" and situation == "failures":
            loc = "lower right"
        if matrix == "audikw_1" and situation == "undisturbed":
            loc = "lower left"
        plt.gca().legend([artists[0][0], artists[2][0], artists[1][0]], \
                         labels, \
                         loc = loc)

        plt.gca().set_ylabel("runtime overhead")
        plt.gca().set_xlabel("checkpointing interval")

        plt.tight_layout()
        plt.subplots_adjust(left=0.18, bottom=0.15)

        fig.set_size_inches(inch_size)
        plt.savefig("/tmp/{}_{}_lineseg.pdf".format(matrix, situation))
        plt.savefig("/tmp/{}_{}_lineseg.pgf".format(matrix, situation))


all_results = read_in_results(sys.argv[1])

for matrix in all_results.keys():

    make_matrix_plots(all_results[matrix],matrix)
