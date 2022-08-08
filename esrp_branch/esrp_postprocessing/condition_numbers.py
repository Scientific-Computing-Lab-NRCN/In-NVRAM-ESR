#!/usr/bin/env python3

import os

import numpy as np
import scipy.io

from exp_setup import *

n=128
nhalf = int(n/2)
m=8

def get_filename(matrix):
    return os.path.join("/home/pachajoamc78cs/matrices", "{}.mtx".format(matrix))

def load_matrix(filename):
    return scipy.sparse.csr_matrix(scipy.io.mmread(filename))

def get_blocks(spm):
    msize = spm.shape[0]

    # Deduce block size and all that
    base_size = int(np.floor(msize / n))
    modulo = msize % n

    # Array with the block sizes
    bsizes=[base_size] * n
    for i in range(modulo):
        bsizes[i] = bsizes[i]+1

    indices = []
    indices.append((0,sum([bsizes[i] for i in range(m)])))
    indices.append((sum([bsizes[i] for i in range(nhalf)]), sum([bsizes[i] for i in range(nhalf+m)])))

    block1 = sp[indices[0][0]:indices[0][1],indices[0][0]:indices[0][1]]
    block2 = sp[indices[1][0]:indices[1][1],indices[1][0]:indices[1][1]]

    return block1, block2


def approx_condition(spm):
    greatest=scipy.sparse.linalg.eigsh(spm, k=1, which="LM", return_eigenvectors=False)
    print(greatest)
    smallest=scipy.sparse.linalg.eigsh(spm, k=1, which="SM", return_eigenvectors=False)
    print(smallest)
    return greatest/smallest


for matrix in matrix_list:

    print ("Processing matrix {}".format(matrix))

    fname = get_filename(matrix)

    sp = load_matrix(fname)

    print("Matrix loaded")

    b1,b2 = get_blocks(sp)

    print("Blocks extracted")

    approx_condition(b1)
    approx_condition(b2)

    print("Condition numbers computed")


"""
Matrix loaded
Blocks extracted
[  4.27585539e+09]
[ 67.2767505]
[  2.81616397e+09]
[ 260914.55787571]
Condition numbers computed
Processing matrix Emilia_923
Matrix loaded
Blocks extracted
[  2.36571564e+12]
[ 0.9999998]
[  3.68660373e+13]
[ 57122455.1271157]
Condition numbers computed
"""
