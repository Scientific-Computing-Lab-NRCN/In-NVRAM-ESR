#!/usr/bin/env python3

import os.path

from exp_setup import *
from file_parser import *

import glob

class Strategy_Results:
    def __init__(self, failure_free, failures):
        self.failure_free = failure_free
        self.failures     = failures

def create_reference_filename(directory, matrix, repetition):
    filename = "{}_reference_rep_{}.txt".format(matrix, repetition)
    return os.path.join(directory, filename)

def create_reference_filename_root(directory, matrix):
    filename = "{}_reference_rep_".format(matrix)
    return os.path.join(directory, filename)

def create_filename(directory, matrix, repetition, \
                    strategy = None, \
                    period = None, \
                    copies = None, \
                    breaknodes = None, \
                    breakiter = None):
    filename = matrix
    if strategy is not None:
        filename += "_resilience_strategy_{}".format(strategy)
    if period is not None:
        filename += "_period_{}".format(period)
    if copies is not None:
        filename += "_copies_{}".format(copies)
    if breaknodes is not None:
        filename += "_breaknodes_{}".format(breaknodes)
    if breakiter is not None:
        filename += "_breakiter_{}".format(breakiter)
    filename += "_rep_{}.txt".format(repetition)

    return os.path.join(directory, filename)

def create_filename_root(directory, matrix, \
                         strategy = None, \
                         period = None, \
                         copies = None, \
                         breaknodes = None, \
                         breakiter = None):
    filename = matrix
    if strategy is not None:
        filename += "_resilience_strategy_{}".format(strategy)
    if period is not None:
        filename += "_period_{}".format(period)
    if copies is not None:
        filename += "_copies_{}".format(copies)
    if breaknodes is not None:
        filename += "_breaknodes_{}".format(breaknodes)
    if breakiter is not None:
        filename += "_breakiter_{}".format(breakiter)
    filename += "_rep_"

    return os.path.join(directory, filename)


def extract_valid_results(root_filename):
    files = glob.glob(root_filename+"*")
    nfiles = len(files)

    results = []

    for i in range(nfiles):
        rep = i+1
        try:
            results.append(fp.read_file(root_filename+"{}.txt".format(rep)))
        except:
            print("Skipping file {}".format(files[i]))
            pass

    return results


nof_copies_list = list(set([x[0] for x in break_config]))
nof_copies_list.sort()

period_list_inplace = [0] + period_list


fp = File_Parser()
rep_list = range(1,testruns+1)


def read_in_results(prefix_directory):

    results = {}

    for matrix in matrix_list:

        matrix_dict = {}

        directory = os.path.join(prefix_directory, "{}".format(nnodes), matrix)

        # Reference results
        root = create_reference_filename_root(directory, matrix)
        matrix_dict["reference"] = extract_valid_results(root)

        # CR, failure-free

        cr_failure_free_dic = {}
        for period in period_list:
            copies_dic = {}
            for copies in nof_copies_list:

                root = create_filename_root(directory, matrix,\
                                            strategy="checkpoint",\
                                            period=period, \
                                            copies=copies)

                copies_dic[copies] = extract_valid_results(root)

            cr_failure_free_dic[period] = copies_dic

        # CR, failures

        cr_failures_dic = {}

        for period in period_list:
            breaknode_dic = {}
            for [copies,broken_nodes] in break_config:
                progress_dic = {}
                for progress in progress_list:

                    root = create_filename_root(directory, matrix, \
                                                strategy = "checkpoint", \
                                                period = period, \
                                                copies = copies, \
                                                breaknodes = broken_nodes, \
                                                breakiter = progress)

                    progress_dic[progress] = extract_valid_results(root)

                breaknode_dic[broken_nodes] = progress_dic

            cr_failures_dic[period] = breaknode_dic

        matrix_dict["checkpoint"] = Strategy_Results(cr_failure_free_dic, cr_failures_dic)


        # Inplace, failure free

        inplace_failure_free_dic = {}
        for period in period_list_inplace:
            copies_dic = {}
            for copies in nof_copies_list:

                root = create_filename_root(directory,matrix,\
                                            strategy="inplace",\
                                            period=period, \
                                            copies=copies)

                copies_dic[copies] = extract_valid_results(root)

            inplace_failure_free_dic[period] = copies_dic

        # Inplace, failures
        inplace_failures_dic = {}
        for period in period_list_inplace:
            breaknode_dic = {}
            for [copies,broken_nodes] in break_config:
                progress_dic = {}
                for progress in progress_list:
                    if progress == "rightbeforecp" and period == 0:
                        continue    # This situation dpesn't have meaning. No file for it. We skip it

                    root = create_filename_root(directory, matrix, \
                                                strategy = "inplace", \
                                                period = period, \
                                                copies = copies, \
                                                breaknodes = broken_nodes, \
                                                breakiter = progress)

                    progress_dic[progress] = extract_valid_results(root)

                breaknode_dic[broken_nodes] = progress_dic

            inplace_failures_dic[period] = breaknode_dic

        matrix_dict["inplace"] = Strategy_Results(inplace_failure_free_dic, inplace_failures_dic)


        results.update({matrix: matrix_dict})

    return results
