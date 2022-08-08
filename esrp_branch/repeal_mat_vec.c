#include <math.h>

#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_spmatrix.h>

#include <mpi.h>

#include "repeal_utils.h"
#include "repeal_mat_vec.h"
#include "repeal_commstrategy.h"
#include "repeal_spmv.h"





double dot(MPI_Comm comm, const gsl_vector* v, const gsl_vector* w)
{
  double vw;
  if (gsl_blas_ddot(v, w, &vw) != 0)
    failure("Dot product computation failed.");
  MPI_Allreduce(MPI_IN_PLACE, &vw, 1, MPI_DOUBLE, MPI_SUM, comm);
  return vw;
}

double norm(MPI_Comm comm, const gsl_vector* v)
{
  return sqrt(dot(comm, v, v));
}

gsl_vector* random_vector(size_t size, double min_elem, double max_elem)
{
  gsl_vector* vector = gsl_vector_alloc(size);
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  double value;
  for (size_t i = 0; i < size; i++) {
    value = gsl_rng_uniform(rng) * (max_elem - min_elem) + min_elem;
    gsl_vector_set(vector, i, value);
  }
  gsl_rng_free(rng);
  return vector;
}


void matrix_min_max(MPI_Comm comm, const gsl_spmatrix* matrix, double* min_elem, double* max_elem)
{
  // Minimum and maximum elements of local submatrix
  gsl_spmatrix_minmax(matrix, min_elem, max_elem);
  if (gsl_spmatrix_nnz(matrix) < matrix->size1 * matrix->size2) {
    if (*min_elem > 0.)
      *min_elem = 0.;
    if (*max_elem < 0.)
      *max_elem = 0.;
  }

  // Minimum and maximum elements of whole matrix
  MPI_Allreduce(MPI_IN_PLACE, min_elem, 1,
                  MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(MPI_IN_PLACE, max_elem, 1,
                  MPI_DOUBLE, MPI_MAX, comm);
}


gsl_vector *create_random_vector_from_matrix(MPI_Comm comm, gsl_spmatrix *M)
{
	// Minimum and maximum matrix elements
    double M_min, M_max;
    matrix_min_max(comm, M, &M_min, &M_max);

	//create random vector within that range
	return random_vector(M->size1, M_min, M_max);
}


double compute_residual_norm(MPI_Comm comm, struct repeal_matrix *mat, const gsl_vector* x, const gsl_vector* b, struct SPMV_context *spmv_ctx)
{
	gsl_vector* vec = gsl_vector_alloc(mat->size1);

	spmv(comm, mat, x, vec, spmv_ctx, 0);

    if (gsl_blas_daxpy(-1., b, vec) != 0)
        failure("BLAS error");
	double abs_norm = norm(comm, vec);

	gsl_vector_free(vec);

	return abs_norm;
}


void split_matrix(struct repeal_matrix *mat, int comm_rank, const int *elements_per_process, const int *displacements, const int global_communication_required)
{
	int lower_col_index, upper_col_index, col_index, lower_value_index, upper_value_index, value_index;

	size_t local_size = mat->size1;
	size_t global_size = mat->size2;

	gsl_spmatrix *M = mat->M;

	//local part

	lower_col_index = displacements[comm_rank];
	upper_col_index = lower_col_index + elements_per_process[comm_rank];

	int nz_local = M->p[upper_col_index] - M->p[lower_col_index]; //number of nonzero values in the local part

	//gsl_spmatrix *M_intern_triplet = gsl_spmatrix_alloc(M->size1, M->size1);
	gsl_spmatrix *M_intern_triplet = gsl_spmatrix_alloc_nzmax(local_size, local_size, nz_local, GSL_SPMATRIX_TRIPLET);

	for (col_index = lower_col_index; col_index < upper_col_index; ++col_index)
	{
		//iterate over all values in that column
		lower_value_index = M->p[col_index];
		upper_value_index = M->p[col_index+1];
		for (value_index = lower_value_index; value_index < upper_value_index; ++value_index)
		{
			//copy value into new matrix
			//column indices are adjusted
			gsl_spmatrix_set(M_intern_triplet, M->i[value_index], col_index - displacements[comm_rank], M->data[value_index]);
		}
	}

	//convert to CCS storage
	mat->M_intern = gsl_spmatrix_ccs(M_intern_triplet);
	gsl_spmatrix_free(M_intern_triplet);


	//external part
	if (global_communication_required)
	{
		gsl_spmatrix *M_extern_triplet = gsl_spmatrix_alloc_nzmax(local_size, global_size, M->nz - nz_local, GSL_SPMATRIX_TRIPLET);

		//everything to the left of my range
		lower_col_index = 0;
		upper_col_index = displacements[comm_rank];
		for (col_index = lower_col_index; col_index < upper_col_index; ++col_index)
		{
			//iterate over all values in that column
			lower_value_index = M->p[col_index];
			upper_value_index = M->p[col_index+1];
			for (value_index = lower_value_index; value_index < upper_value_index; ++value_index)
			{
				//copy value into new matrix
				//no adjustment of column indices this time
				gsl_spmatrix_set(M_extern_triplet, M->i[value_index], col_index, M->data[value_index]);
			}
		}

		//everything to the right of my range
		lower_col_index = displacements[comm_rank] + elements_per_process[comm_rank];
		upper_col_index = global_size;
		for (col_index = lower_col_index; col_index < upper_col_index; ++col_index)
		{
			//iterate over all values in that column
			lower_value_index = M->p[col_index];
			upper_value_index = M->p[col_index+1];
			for (value_index = lower_value_index; value_index < upper_value_index; ++value_index)
			{
				//copy value into new matrix
				//no adjustment of column indices this time
				gsl_spmatrix_set(M_extern_triplet, M->i[value_index], col_index, M->data[value_index]);
			}
		}

		//convert to CCS
		mat->M_extern = gsl_spmatrix_ccs(M_extern_triplet);
		gsl_spmatrix_free(M_extern_triplet);
	}

}


struct repeal_matrix *repeal_matrix_create(MPI_Comm comm, int comm_rank, int comm_size, gsl_spmatrix *M, const enum COMM_strategy strategy, int external_communication_needed, const struct SPMV_context *spmv_ctx, const struct ESR_options *esr_opts)
{
	struct repeal_matrix *mat = malloc(sizeof(struct repeal_matrix));
	if (!mat) failure("Allocation of Matrix wrapper failed.");

	mat->M = M;
	mat->M_intern = NULL;
	mat->M_extern = NULL;
	mat->alltoallv_info = NULL;
	mat->minimal_info = NULL;
	mat->minimal_info_with_resilience = NULL;
	mat->external_communication_needed = external_communication_needed;
	mat->size1 = M->size1;
	mat->size2 = M->size2;

	switch(strategy)
	{
		case COMM_allgatherv:
			break;
		case COMM_alltoallv:
			mat->alltoallv_info = get_spmv_alltoallv_info(comm, comm_rank, comm_size, M, spmv_ctx->elements_per_process, spmv_ctx->displacements);
			break;
		case COMM_minimal:
			if (esr_opts->redundant_copies)
			{
				mat->minimal_info = get_spmv_minimal_info(comm, comm_rank, comm_size, M, spmv_ctx->elements_per_process, spmv_ctx->displacements, 0, NULL);
				mat->minimal_info_with_resilience = get_spmv_minimal_info(comm, comm_rank, comm_size, M, spmv_ctx->elements_per_process, spmv_ctx->displacements, esr_opts->redundant_copies, esr_opts->buddies_that_save_me);
			}
			else
			{
				mat->minimal_info = get_spmv_minimal_info(comm, comm_rank, comm_size, M, spmv_ctx->elements_per_process, spmv_ctx->displacements, esr_opts->redundant_copies, esr_opts->buddies_that_save_me);
			}
			break;
		case COMM_minimal_split:
			split_matrix(mat, comm_rank, spmv_ctx->elements_per_process, spmv_ctx->displacements, external_communication_needed);
			if (external_communication_needed)
			{
				if (esr_opts->redundant_copies)
				{
					printf("DEBUG: redundant copies is on for minimal info\n"); 
					mat->minimal_info = get_spmv_minimal_info(comm, comm_rank, comm_size, mat->M_extern, spmv_ctx->elements_per_process, spmv_ctx->displacements, 0, NULL);
					mat->minimal_info_with_resilience = get_spmv_minimal_info(comm, comm_rank, comm_size, mat->M_extern, spmv_ctx->elements_per_process, spmv_ctx->displacements, esr_opts->redundant_copies, esr_opts->buddies_that_save_me);
				}
				else
				{
					printf("DEBUG: redundant copies is off for minimal info\n"); 
					mat->minimal_info = get_spmv_minimal_info(comm, comm_rank, comm_size, mat->M_extern, spmv_ctx->elements_per_process, spmv_ctx->displacements, esr_opts->redundant_copies, esr_opts->buddies_that_save_me);
				}				
			}
			break;
		default:
			failure("Unknown communication strategy");
	}

	return mat;
}

void repeal_matrix_free(struct repeal_matrix *mat)
{
	gsl_spmatrix_free(mat->M);
	if (mat->M_intern) gsl_spmatrix_free(mat->M_intern);
	if (mat->M_extern) gsl_spmatrix_free(mat->M_extern);
	if (mat->alltoallv_info) free_spmv_comm_structure(mat->alltoallv_info);
	if (mat->minimal_info) free_spmv_comm_structure_minimal(mat->minimal_info);

	free(mat);
}
