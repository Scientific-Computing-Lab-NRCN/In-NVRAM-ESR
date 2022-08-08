#ifndef REPEAL_BINARY_IO
#define REPEAL_BINARY_IO

#include <mpi.h>
#include <stdio.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_vector.h>

#include "repeal_size_t_mpi.h"

// Types required for binary read-in
typedef unsigned int Int;
typedef double Scalar;

/** Partial read of the matrix by ranks in a communicator
 *
 *  The root rank of the communicator will read up the relevant entries of the matrix and
 *  subsequently transfer them to the other ranks. This function is intended to be used by a
 *  general driver that creates multiple communicators in the set of replacement nodes, such
 *  that we can control how many ranks try to access the file in the hard disk simultaneously.
 *
 *  @param[in]  root        Rank that will read in the matrix from disk.
 *  @param[in]  startend    Array with successive pairs start-end of indexes assigned to the nodes.
 *                          Must be of size 2 x comm size.
 *
 *  @param[out] matrix      The matrix for this rank.
 */
void partial_read_matrix_block_split_in_comm(MPI_Comm comm,
                                             int root,
                                             const int *const startend,
                                             const char *const filename,
                                             gsl_spmatrix **matrix);

/** Partially read a vector binary file */
void partial_read_vector_block_split_in_comm(MPI_Comm comm,
                                             int root,
                                             const int *const startend,
                                             const char *const filename,
                                             gsl_vector **vector);

/** Read a binary matrix and produce its distribution
 *
 *  The number of groups corresponds of the number of roots reading in from the matrix
 */
void read_binary_matrix(MPI_Comm comm,
                        const int nof_groups,
                        const int root,
                        const char *const filename,
                        int **rows_per_process,
                        int **row_displacements,
                        gsl_spmatrix **matrix);

#endif
