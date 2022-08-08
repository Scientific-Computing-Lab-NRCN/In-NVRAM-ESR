#include "repeal_pc.h"

#include <stdio.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_spmatrix.h>
#include <mpi.h>

#include "repeal_mat_vec.h"
#include "repeal_options.h"

// functions for allocating the different preconditioners
// Preconditioner P = M^(-1)

// No preconditioner
gsl_spmatrix *get_identity_preconditioner(const gsl_spmatrix *A,
                                          const int *displacements,
                                          int comm_rank) {
  size_t local_size = A->size1;

  // allocate the part of the identity matrix held by the current rank
  gsl_spmatrix *P_triplet = gsl_spmatrix_alloc_nzmax(
      local_size, A->size2, local_size, GSL_SPMATRIX_TRIPLET);
  int offset = displacements[comm_rank];
  for (size_t i = 0; i < local_size; i++) {
    if (gsl_spmatrix_set(P_triplet, i, i + offset, 1.) !=
        0)  // this only works with a matrix in triplet format
      failure("Could not create preconditioner matrix.");
  }

  // convert to CCS storage format
  gsl_spmatrix *P = gsl_spmatrix_ccs(P_triplet);
  gsl_spmatrix_free(P_triplet);
  return P;
}

// Block Jacobi
gsl_spmatrix *get_blockjacobi_preconditioner(const gsl_spmatrix *A,
                                             const int *displacements,
                                             int max_block_size,
                                             int comm_rank) {
  size_t local_size = A->size1;

  int num_blocks, blocksize, first_row, last_row, num_rows, num_rem_rows,
      offset_i, offset_j;
  int signum;  // needed for LU decomposition (sign of the permutation)
  gsl_spmatrix *P_triplet, *P;
  gsl_matrix *block, *inverse_block;
  gsl_permutation *perm;

  // find out how many blocks are needed
  if (max_block_size ==
      -1)  // this means the block size should be the number of rows held by
           // each process (i.e. one block per process)
    max_block_size = local_size;
  num_blocks = local_size / max_block_size;
  if (local_size % max_block_size !=
      0)  // add one block to deal with the remaining rows
    num_blocks++;
  num_rows = local_size / num_blocks;
  num_rem_rows = local_size % num_blocks;

  // horizontal and vertical starting position of a block
  offset_j = displacements[comm_rank];
  offset_i = 0;

  // allocate the part of the preconditioner matrix held by the current rank
  P_triplet = gsl_spmatrix_alloc_nzmax(
      local_size, A->size2, max_block_size * max_block_size * num_blocks,
      GSL_SPMATRIX_TRIPLET);  // returns a matrix in triplet representation

  for (int block_id = 0; block_id < num_blocks; ++block_id) {
    // compute first and last row of the block and its number of rows
    first_row = block_id * num_rows + min(block_id, num_rem_rows);
    last_row = (block_id + 1) * num_rows + min(block_id + 1, num_rem_rows) - 1;
    blocksize = last_row - first_row + 1;

    // allocate memory for intermediate results
    block = gsl_matrix_alloc(blocksize, blocksize);
    inverse_block = gsl_matrix_alloc(blocksize, blocksize);
    perm = gsl_permutation_alloc(blocksize);

    // store the block in a separate matrix so it can be inverted
    for (size_t i = 0; i < blocksize; i++)
      for (size_t j = 0; j < blocksize; ++j)
        gsl_matrix_set(block, i, j,
                       gsl_spmatrix_get(A, i + offset_i, j + offset_j));

    // invert the block
    gsl_linalg_LU_decomp(block, perm, &signum);  // LU decomposition of block
    gsl_linalg_LU_invert(
        block, perm, inverse_block);  // inverse of block from LU decomposition

    // set appropriate preconditioner values
    for (size_t i = 0; i < blocksize; i++)
      for (size_t j = 0; j < blocksize; ++j)
        gsl_spmatrix_set(P_triplet, i + offset_i, j + offset_j,
                         gsl_matrix_get(inverse_block, i, j));

    // adjust offsets for next block
    offset_j += blocksize;
    offset_i += blocksize;

    // free memory
    gsl_matrix_free(block);
    gsl_matrix_free(inverse_block);
    gsl_permutation_free(perm);
  }

  // convert to CCS storage format
  P = gsl_spmatrix_ccs(P_triplet);

  // free memory
  gsl_spmatrix_free(P_triplet);
  return P;
}

// Jacobi
gsl_spmatrix *get_jacobi_preconditioner(const gsl_spmatrix *A,
                                        const int *displacements,
                                        int comm_rank) {
  size_t local_size = A->size1;

  // allocate the part of the preconditioner matrix held by the current rank
  gsl_spmatrix *P_triplet = gsl_spmatrix_alloc(
      local_size, A->size2);  // returns a matrix in triplet representation

  // set P to inverse diagonal of matrix A
  int offset = displacements[comm_rank];
  for (size_t i = 0; i < local_size; i++) {
    if (gsl_spmatrix_set(P_triplet, i, i + offset,
                         1. / gsl_spmatrix_get(A, i, i + offset)) != 0)
      failure("Could not create preconditioner matrix.");
  }

  // convert to CCS storage format
  gsl_spmatrix *P = gsl_spmatrix_ccs(P_triplet);

  gsl_spmatrix_free(P_triplet);
  return P;
}

// this function decides which preconditioner should be created
struct repeal_matrix *get_preconditioner(MPI_Comm comm, int comm_rank,
                                         int comm_size, const gsl_spmatrix *A,
                                         const struct PC_options *pc_opts,
                                         const enum COMM_strategy strategy,
                                         const struct SPMV_context *spmv_ctx) {
  gsl_spmatrix *P_data = NULL;
  switch (pc_opts->pc_type) {
    case PC_none:
      P_data =
          get_identity_preconditioner(A, spmv_ctx->displacements, comm_rank);
      break;
    case PC_BJ:
      P_data = get_blockjacobi_preconditioner(
          A, spmv_ctx->displacements, pc_opts->pc_bj_blocksize, comm_rank);
      break;
    case PC_Jacobi:
      P_data = get_jacobi_preconditioner(A, spmv_ctx->displacements, comm_rank);
      break;
    default:
      failure("Unknown PC type!");
  }

  int external_communication_needed;
  switch (pc_opts->pc_type) {
    case PC_none:
    case PC_BJ:
    case PC_Jacobi:
      external_communication_needed = 0;
      break;
    default:  // in case another preconditioner is added in the future
      external_communication_needed = 1;
  }

  struct ESR_options *esr_opts_default =
      get_esr_options();  // no resilience required for the preconditioner
                          // matrix
  struct repeal_matrix *P = repeal_matrix_create(
      comm, comm_rank, comm_size, P_data, strategy,
      external_communication_needed, spmv_ctx, esr_opts_default);
  free_esr_options(esr_opts_default);
  return P;
}
