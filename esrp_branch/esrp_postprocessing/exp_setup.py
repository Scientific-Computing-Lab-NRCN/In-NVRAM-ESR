#!/usr/bin/env python3

import numpy as np

matrix_list = ["audikw_1", "Emilia_923"]
period_list = [20,50,100]

testruns = 5

nnodes = 128

break_config = [(1,"0"),(1,"64"), \
                (3,"0,1,2"),(3,"64,65,66"), \
                (8,"0,1,2,3,4,5,6,7"),(8,"64,65,66,67,68,69,70,71")]

progress_list = ["betweencp", "rightbeforecp"]

strategy_dict = {"inplace": "ESRP", "checkpoint": "IMCR"}
progress_dict = {"betweencp": "middle", "rightbeforecp": "end"}
location_dict = {"Start": 0, "Center": 1}

copies_list = list(set([x[0] for x in break_config]))
copies_list.sort()

# Produce a list of dictionaries. The position in the list marks the number of copies,
# and the keys are the string with the used nodes.
break_transposition = {x:[j[1] for j in break_config if j[0] == x] for x in copies_list}

def get_runtime_from_results_list(results_list, last=testruns):
    ref_list = [x.runtime for x in results_list[-last:]]
    median = np.median(ref_list)
    stdv = np.std (ref_list)
    return median, stdv

def get_rec_time_from_results_list(results_list):
    median = np.median([x.rec_time for x in results_list])
    stdv = np.std ([x.rec_time for x in results_list])
    return median, stdv
