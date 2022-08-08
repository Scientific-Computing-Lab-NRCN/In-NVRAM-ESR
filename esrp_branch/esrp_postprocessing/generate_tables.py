#!/usr/bin/env python3

# Write a summary table for each matrix

import sys
import numpy as np

from exp_setup import *

import collect_results




class TableWriteSwitches:
    def reset(self):
        self.strategy = True
        self.period   = True
        self.progress = True

    def __init__(self):
        self.reset()

def percentual_overhead(reference, value):
    return 100 * (value - reference) / reference

def percentual_rel(reference, value):
    return 100 * value / reference


results = collect_results.read_in_results(sys.argv[1])


switches = TableWriteSwitches()


format_floats  = " & {:7.4f}" * len(copies_list)
format_strings = " & {:7}" * len(copies_list)


# strategy | period | relative overhead undisturbed | progress  | location | relative overhead failure | maybe failure overhead

for matrix in results.keys():

    fp = open("/tmp/{}_data.tex".format(matrix), "w")

    reference_time,_ = get_runtime_from_results_list(results[matrix]["reference"])

    fp.write("% {}. Reference time = {}s\n".format(matrix, reference_time))

    for strategy in strategy_dict.keys():

        switches.strategy = True

        period_list_adj = [x for x in period_list]
        if strategy == "inplace":
            period_list_adj = [0] + period_list_adj

        for period in period_list_adj:

            switches.period = True

            for progress in progress_list:

                if progress == "rightbeforecp" and period == 0:
                    continue

                # Exclude all that is not "worst case"
                if progress == "betweencp" and period != 0:
                    continue

                switches.progress = True

                for location in ["Start", "Center"]:

                    if location == "Center":
                        vertical_space = "[2pt]"
                    else:
                        vertical_space = ""

                    # Overhead failure

                    # results["audikw_1"]["inplace"].failures[0]["0,1,2"]["betweencp"]
                    failure_overhead_printout = \
                        [percentual_overhead(reference_time, get_runtime_from_results_list(L)[0]) \
                        for L in [results[matrix][strategy].failures[period][broken][progress] \
                        for broken in [break_transposition[copies][location_dict[location]] \
                        for copies in copies_list]]]

                    recovery_overhead_printout = \
                        [percentual_rel(reference_time, get_rec_time_from_results_list(L)[0]) \
                        for L in [results[matrix][strategy].failures[period][broken][progress] \
                        for broken in [break_transposition[copies][location_dict[location]] \
                        for copies in copies_list]]]


                    if switches.strategy:
                        strategy_printout = strategy_dict[strategy]
                        switches.strategy = False
                    else:
                        strategy_printout = ""

                    if switches.period:
                        period_printout = period if period != 0 else 1
                        overhead_undisturbed_printout = [percentual_overhead(reference_time, get_runtime_from_results_list(L)[0])
                            for L in [results[matrix][strategy].failure_free[period][copies]
                            for copies in copies_list]]
                        overhead_undisturbed_format = format_floats
                        switches.period = False
                    else:
                        overhead_undisturbed_printout = ["","",""]
                        overhead_undisturbed_format = format_strings
                        period_printout = ""

                    if switches.progress:
                        progress_printout = progress_dict[progress]
                        switches.progress = False
                    else:
                        progress_printout = ""

                    # Print the full line
                    final_format = "{:20} & {:4} &" + overhead_undisturbed_format + " & & {:8} &" + format_floats + " &" + format_floats + " \\\\{}\n"
                    fp.write(final_format \
                            .format(strategy_printout,\
                                    period_printout,\
                                    overhead_undisturbed_printout[0],\
                                    overhead_undisturbed_printout[1],\
                                    overhead_undisturbed_printout[2],\
                                    location,\
                                    failure_overhead_printout[0],\
                                    failure_overhead_printout[1],\
                                    failure_overhead_printout[2],\
                                    recovery_overhead_printout[0],\
                                    recovery_overhead_printout[1],\
                                    recovery_overhead_printout[2],\
                                    vertical_space,
                                    ))

    fp.close()
