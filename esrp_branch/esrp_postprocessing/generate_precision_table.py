#!/usr/bin/env python3

import sys
import numpy as np

from exp_setup import *
from collect_results import read_in_results

period_list_inplace = [0] + period_list

class TableWriteSwitches:
    def reset(self):
        self.strategy = True
        self.period   = True
        self.progress = True

    def __init__(self):
        self.reset()

def get_relative_deviation(comp, ref):
    return (ref - comp) / comp

break_location_dict = {
        "0": "Start ",
        "64":"Center",
        "0,1,2":"Start ",
        "64,65,66":"Center",
        "0,1,2,3,4,5,6,7":"Start ",
        "64,65,66,67,68,69,70,71":"Center",
}
break_size_dict = {
        "0":1,
        "64":1,
        "0,1,2":3,
        "64,65,66":3,
        "0,1,2,3,4,5,6,7":8,
        "64,65,66,67,68,69,70,71":8,
}

def process_matrix(matrix_name, matrix_results):
    # Get the reference drift

    print(matrix_name)

    ref_computed_residual = matrix_results["reference"][0].comp_residual
    ref_reported_residual = matrix_results["reference"][0].rep_residual

    reference_deviation = get_relative_deviation(ref_computed_residual,
                                                 ref_reported_residual)

    print("{:.2e}".format(reference_deviation))

    # Failure-free results should match the reference drift

    # With failures
    all_deviations = []
    for period in period_list_inplace:
        deviations = []
        for [copies, broken_nodes] in break_config:
            if period == 0:
                progress = "betweencp"
            else:
                progress = "rightbeforecp"

            computed_residual = matrix_results["inplace"].failures[period] \
                                              [broken_nodes][progress][0] \
                                              .comp_residual
            reported_residual = matrix_results["inplace"].failures[period] \
                                              [broken_nodes][progress][0] \
                                              .rep_residual

            deviation = get_relative_deviation(computed_residual, reported_residual)
            all_deviations.append(deviation)
            deviations.append(deviation)

        print("{} & {} & & {:.5e} & {:.5e} & {:.5e}\n"
              "{} & {} & & {:.5e} & {:.5e} & {:.5e}".format(period,"Start ",deviations[0],deviations[2],deviations[4],"  ", "Center",deviations[1],deviations[3],deviations[5]))

    deviations = np.array(deviations)

    print("{:.2e}".format(np.median(all_deviations)))
    print("")



all_results = read_in_results(sys.argv[1])

# import pdb; pdb.set_trace()
for matrix in all_results.keys():
    process_matrix(matrix, all_results[matrix])
