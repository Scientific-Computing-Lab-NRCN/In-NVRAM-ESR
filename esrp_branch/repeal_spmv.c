#include <assert.h>
#include <stdlib.h>

#include <stdio.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_vector.h>
#include <mpi.h>

#include "repeal_utils.h"
#include "repeal_spmv.h"

/*
void get_data_distribution(MPI_Comm comm, int comm_size, int n_rows, struct SPMV_context *spmv_ctx)
{
	spmv_ctx->elements_per_process = malloc(comm_size * sizeof(int));
    spmv_ctx->displacements = malloc(comm_size * sizeof(int));

    //allgather n_rows so that each process knows how many rows the other processes own
    MPI_Allgather(&n_rows, 1, MPI_INT, *elements_per_process, 1, MPI_INT, comm);

    //displacements for allgatherv (= global index of the first row owned by a process)
    int sum = 0;
    for (int i = 0; i < comm_size; ++i)
	{
        (*displacements)[i] = sum;
        sum += (*elements_per_process)[i];
    }
}
*/


struct SPMV_context *get_spmv_context(MPI_Comm comm, int comm_size, enum COMM_strategy strategy, size_t local_size, size_t global_size)
{
	struct SPMV_context *spmv_ctx = malloc(sizeof(struct SPMV_context));
	if (!spmv_ctx) failure("Allocation of SPMV context failed.");

	spmv_ctx->strategy = strategy;

	spmv_ctx->sendbuf = gsl_vector_alloc(local_size); //that's only needed if strategy == split, right?
	spmv_ctx->buffer = gsl_vector_alloc(global_size);

	spmv_ctx->elements_per_process = malloc(comm_size * sizeof(int));
    spmv_ctx->displacements = malloc(comm_size * sizeof(int));
	if (!spmv_ctx->elements_per_process || !spmv_ctx->displacements) failure("Allocation of SPMV context failed.");

	int local_size_int = (int) local_size;
    //allgather number of rows (=localsize) so that each process knows how many rows the other processes own
    MPI_Allgather(&local_size_int, 1, MPI_INT, spmv_ctx->elements_per_process, 1, MPI_INT, comm);

    //displacements (= global index of the first row owned by a process)
    int sum = 0;
    for (int i = 0; i < comm_size; ++i)
	{
        (spmv_ctx->displacements)[i] = sum;
        sum += (spmv_ctx->elements_per_process)[i];
    }

	return spmv_ctx;
}


struct SPMV_context *get_spmv_context_from_existing_distribution(int comm_size, enum COMM_strategy strategy, size_t local_size, size_t global_size, int *elements_per_process, int *displacements)
{
	struct SPMV_context *spmv_ctx = malloc(sizeof(struct SPMV_context));
	if (!spmv_ctx) failure("Allocation of SPMV context failed.");

	spmv_ctx->strategy = strategy;

	spmv_ctx->sendbuf = gsl_vector_alloc(local_size); //that's only needed if strategy == split, right?
	spmv_ctx->buffer = gsl_vector_alloc(global_size);

	spmv_ctx->elements_per_process = malloc(comm_size * sizeof(int));
    spmv_ctx->displacements = malloc(comm_size * sizeof(int));
	if (!spmv_ctx->elements_per_process || !spmv_ctx->displacements) failure("Allocation of SPMV context failed.");

	//copy the given values
	for (int i = 0; i < comm_size; ++i)
		spmv_ctx->elements_per_process[i] = elements_per_process[i];
	for (int i = 0; i < comm_size; ++i)
		spmv_ctx->displacements[i] = displacements[i];

	return spmv_ctx;
}


void free_spmv_context(struct SPMV_context *spmv_ctx)
{
	gsl_vector_free(spmv_ctx->buffer);
	gsl_vector_free(spmv_ctx->sendbuf);
	free(spmv_ctx->elements_per_process);
	free(spmv_ctx->displacements);
	free(spmv_ctx);
}


void spmv_minimal_split(MPI_Comm comm, const gsl_spmatrix *M_intern, const gsl_spmatrix *M_extern, const gsl_vector* v,
    gsl_vector* Mv, gsl_vector* buffer, gsl_vector *sendbuf,
    const struct SPMV_comm_structure_minimal* M_info)
{
	//initiate communication
	MPI_Request request;
	if (M_extern)
	{
		gsl_blas_dcopy(v, sendbuf); //need to copy the data because using a sendbuffer for something else during the send is strictly speaking faulty code
		MPI_Ialltoallw(sendbuf->data, M_info->sendcounts, M_info->senddispl, M_info->sendtypes,
			buffer->data, M_info->recvcounts, M_info->recvdispl, M_info->recvtypes, comm, &request);
	}

	//compute local part of SPMV
	gsl_spblas_dgemv(CblasNoTrans, 1., M_intern, v, 0., Mv);

	//compute external part of SPMV and add to local result
	if (M_extern)
	{
		MPI_Wait(&request, MPI_STATUS_IGNORE);
		gsl_spblas_dgemv(CblasNoTrans, 1., M_extern, buffer, 1., Mv);
	}

}


void spmv_minimal(MPI_Comm comm, const gsl_spmatrix* M, const gsl_vector* v,
    gsl_vector* Mv, gsl_vector* buffer,
    const struct SPMV_comm_structure_minimal* M_info)
{
	//assert(v->size == M->size1);
    //assert(Mv->size == M->size1);
    //assert(buffer->size == M->size2);

	//gsl_vector_set_all(buffer, 0.);

	MPI_Alltoallw(v->data, M_info->sendcounts, M_info->senddispl, M_info->sendtypes, buffer->data, M_info->recvcounts, M_info->recvdispl, M_info->recvtypes, comm);

	//MPI_Request request;
	//MPI_Ialltoallw(v->data, M_info->sendcounts, M_info->senddispl, M_info->sendtypes, buffer->data, M_info->recvcounts, M_info->recvdispl, //M_info->recvtypes, comm, &request);
	//MPI_Wait(&request, MPI_STATUS_IGNORE);

	if (gsl_spblas_dgemv(CblasNoTrans, 1., M, buffer, 0., Mv) != 0)
      failure("Sparse matrix-vector product computation failed.");
}


void spmv_alltoallv(MPI_Comm comm, const gsl_spmatrix* M, const gsl_vector* v,
    gsl_vector* Mv, gsl_vector* buffer,
    const struct SPMV_comm_structure* M_alltoallv_info)
{
    assert(v->size == M->size1);
    assert(Mv->size == M->size1);
    assert(buffer->size == M->size2);

    gsl_vector_set_all(buffer, 0.);

    MPI_Alltoallv(v->data, M_alltoallv_info->sendcounts, M_alltoallv_info->senddispl, MPI_DOUBLE,
            buffer->data, M_alltoallv_info->recvcounts, M_alltoallv_info->recvdispl, MPI_DOUBLE,
            comm);

    if (gsl_spblas_dgemv(CblasNoTrans, 1., M, buffer, 0., Mv) != 0)
      failure("Sparse matrix-vector product computation failed.");

}


void spmv_allgatherv(MPI_Comm comm, const gsl_spmatrix* M, const gsl_vector* v,
     gsl_vector* Mv, gsl_vector* buffer,
     const int* elements_per_process, const int* displacements)
{
  assert(v->size == M->size1);
  assert(Mv->size == M->size1);
  assert(buffer->size == M->size2);

  //using MPI_Allgatherv because MPI_Allgather requires each process
  //to send the same number of elements
  MPI_Allgatherv(v->data, (int) v->size, MPI_DOUBLE,
                buffer->data, elements_per_process, displacements,
                MPI_DOUBLE, comm);

  if (gsl_spblas_dgemv(CblasNoTrans, 1., M, buffer, 0., Mv) != 0)
    failure("Sparse matrix-vector product computation failed.");
}


void spmv(MPI_Comm comm, const struct repeal_matrix *mat, const gsl_vector* v, gsl_vector* Mv, struct SPMV_context *spmv_ctx, int send_redundant_elements)
{
	//wrapper function: simply forwarding the arguments to the appropriate SPMV function


	switch(spmv_ctx->strategy)
	{
		case COMM_allgatherv:
			spmv_allgatherv(comm, mat->M, v, Mv, spmv_ctx->buffer, spmv_ctx->elements_per_process, spmv_ctx->displacements);
			break;
		case COMM_alltoallv:
			spmv_alltoallv(comm, mat->M, v, Mv, spmv_ctx->buffer, mat->alltoallv_info);
			break;
		case COMM_minimal:
			if (send_redundant_elements && mat->minimal_info_with_resilience) 
			{
				spmv_minimal(comm, mat->M, v, Mv, spmv_ctx->buffer, mat->minimal_info_with_resilience); //can only use this if an alternative info struct has been created for the matrix
			}
			else 
			{
				spmv_minimal(comm, mat->M, v, Mv, spmv_ctx->buffer, mat->minimal_info);
			}
			break;
		case COMM_minimal_split:
			if (send_redundant_elements && mat->minimal_info_with_resilience) 
			{
				spmv_minimal_split(comm, mat->M_intern, mat->M_extern, v, Mv, spmv_ctx->buffer, spmv_ctx->sendbuf, mat->minimal_info_with_resilience);
			}
			else
			{
				spmv_minimal_split(comm, mat->M_intern, mat->M_extern, v, Mv, spmv_ctx->buffer, spmv_ctx->sendbuf, mat->minimal_info);
			}
			break;
		//add any additional strategies here
		default:
			failure("Invalid communication strategy");
	}
	
}
