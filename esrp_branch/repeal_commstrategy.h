#ifndef REPEAL_STRATEGY
#define REPEAL_STRATEGY

#include <stdio.h>
#include <gsl/gsl_spmatrix.h>

#include "repeal_utils.h"

struct SPMV_comm_structure *get_spmv_alltoallv_info(MPI_Comm comm, int comm_rank, int comm_size, const gsl_spmatrix* matrix, const int *elements_per_process, const int *displacements);
struct SPMV_comm_structure_minimal *get_spmv_minimal_info(MPI_Comm comm, int comm_rank, int comm_size, const gsl_spmatrix* M, const int *elements_per_process, const int *displacements, const int redundant_copies, const int *buddies_that_save_me);
void free_spmv_comm_structure(struct SPMV_comm_structure *comm_struct);
void free_spmv_comm_structure_minimal(struct SPMV_comm_structure_minimal *comm_struct);

//void split_matrix(struct repeal_matrix *mat, int comm_rank, const int *elements_per_process, const int *displacements, const int global_communication_required);

void get_buddies_for_rank(int rank, int comm_size, int redundant_copies, int *buddies);
void get_buddy_info(struct ESR_options *esr_opts, int comm_rank, int comm_size);

#endif
