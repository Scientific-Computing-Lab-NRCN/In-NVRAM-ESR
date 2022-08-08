#include <stdlib.h>

#include <stdio.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_spmatrix.h>

#include <mpi.h>
#include <mpi_one_sided_extension/mpi_win_pmem.h>
#include <mpi_one_sided_extension/defines.h>

#include "repeal_reconstruction.h"
#include "repeal_reconstruction_utils.h"
#include "repeal_pcg.h"
#include "repeal_utils.h"
#include "repeal_options.h"
#include "repeal_mat_vec.h"
#include "repeal_spmv.h"
#include "repeal_save_state.h"
#include "repeal_recovery.h"
#include <omp.h>





void pcg_erase_local_data(struct PCG_solver_handle *pcg_handle, struct PCG_state_copy *pcg_state_copy)
{
	pcg_state_copy_set_zero(pcg_state_copy);
	pcg_solver_handle_set_zero(pcg_handle);
}




void pcg_break_and_recover(
	MPI_Comm *world_comm,
	PMEMobjpool* pool,
	MPI_Win win,
	bool win_ram_on,
	char * checkpoint_file_path,
	struct PCG_solver_handle *pcg_handle, //world communicator + dynamic solver data
	const struct repeal_matrix *A, const struct repeal_matrix *P, const gsl_vector *b, //static data
	struct PCG_state_copy *pcg_state_copy, //redundant copies etc.
	const struct ESR_options *esr_opts,
	const struct SPMV_context *spmv_ctx
)
{
	struct ESR_setup *esr_setup = create_esr_setup();
	//init_esr_setup(&esr_setup); // Nulls out the pointers

	/*** configure ***/

	int comm_rank;
	MPI_Comm_rank(*(world_comm), &comm_rank);

	printf("DEBUG: start break and recover --- rank %d\n", comm_rank); 

	// Configure reconstruction
	// Here, we decide roles and relationships between nodes related to the reconstruction
	configure_reconstruction(world_comm, esr_opts, esr_setup);

	/*
	//testoutput
	if (esr_setup->rec_role == ROLE_reconstructing)
	{
		printf("Rank %d: x before reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%.3f ", pcg_handle->x->data[i]);
		printf("]\n");

		printf("Rank %d: r before reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%.3f ", pcg_handle->r->data[i]);
		printf("]\n");

		printf("Rank %d: p before reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%.3f ", pcg_handle->p->data[i]);
		printf("]\n");

		printf("Rank %d: z before reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%.3f ", pcg_handle->z->data[i]);
		printf("]\n");

		printf("Rank %d: beta before reconstruction: %.3f \n", comm_rank, *(pcg_handle->beta));
		printf("Rank %d: rz before reconstruction: %.3f \n", comm_rank, *(pcg_handle->rz));
	}
	*/


	/*
	//testoutput for periodic ESR: check against the saved state, because that's what we're resetting to
	if (esr_setup->rec_role == ROLE_reconstructing)
	{
		printf("Rank %d: x before reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%.3f ", pcg_state_copy->x_local->data[i]);
		printf("]\n");

		printf("Rank %d: r before reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%.3f ", pcg_state_copy->r_local->data[i]);
		printf("]\n");

		printf("Rank %d: p before reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%.3f ", pcg_state_copy->p_local->data[i]);
		printf("]\n");

		printf("Rank %d: z before reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%.3f ", pcg_state_copy->z_local->data[i]);
		printf("]\n");

		printf("Rank %d: beta before reconstruction: %.3f \n", comm_rank, pcg_state_copy->beta);
		printf("Rank %d: rz before reconstruction: %.3f \n", comm_rank, pcg_state_copy->rz);
	}
	*/
	
	
	
	/*** break ***/

	//simulate loss of dynamic data
	// We can use it like this, with a branch and no wrappers,
	// until some more complicated scenario appears
	if (esr_setup->local_data_fate == FATE_erase)
	{
		pcg_erase_local_data(pcg_handle, pcg_state_copy);
	}

	retrieve_static_data(world_comm, esr_opts, esr_setup);

	/*** reconstruct ***/
    double start = omp_get_wtime();
	pcg_reconstruct(world_comm, pool, win, win_ram_on, checkpoint_file_path, pcg_handle, A, P, b, pcg_state_copy, esr_opts, esr_setup, spmv_ctx);
    double end = omp_get_wtime();
	double reconstruction_time = end-start;
	printf("Reconstruction Time = %f\n", reconstruction_time);
	/*
	//testoutput
	if (esr_setup->rec_role == ROLE_reconstructing)
	{
		printf("Rank %d: x after reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%.3f ", pcg_handle->x->data[i]);
		printf("]\n");

		printf("Rank %d: r after reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%.3f ", pcg_handle->r->data[i]);
		printf("]\n");

		printf("Rank %d: p after reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%.3f ", pcg_handle->p->data[i]);
		printf("]\n");

		printf("Rank %d: z after reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%.3f ", pcg_handle->z->data[i]);
		printf("]\n");

		printf("Rank %d: beta after reconstruction: %.3f \n", comm_rank, *(pcg_handle->beta));
		printf("Rank %d: rz after reconstruction: %.3f \n", comm_rank, *(pcg_handle->rz));
	}
	*/
	

	/*** redistribute ***/
	//pcg_redistribute_data();

	/*** re-setup ***/
	//pcg_setup_after_reconstruction();

	// Update communicator if required
	
	if(esr_opts->reconstruction_strategy == REC_on_survivors) {
		if(*world_comm != MPI_COMM_NULL)
			MPI_Comm_free(world_comm);
		if(esr_setup->continue_comm != MPI_COMM_NULL)
			MPI_Comm_dup(esr_setup->continue_comm, world_comm);
	}
	

	/*** Cleanup ***/
	free_esr_setup(esr_setup);
}




void pipelined_pcg_break_and_recover(
	MPI_Comm *world_comm,
	struct Pipelined_solver_handle *pipelined_handle, //world communicator + dynamic solver data
	const struct repeal_matrix *A, const struct repeal_matrix *P, const gsl_vector *b, //static data
	struct Pipelined_state_copy *pipelined_state_copy, //redundant copies
	const struct ESR_options *esr_opts,
	const struct SPMV_context *spmv_ctx
)
{

	/*** configure ***/

	struct ESR_setup *esr_setup = create_esr_setup();
	configure_reconstruction(world_comm, esr_opts, esr_setup);


	
	//testoutput
	int comm_rank;
	MPI_Comm_rank(*(world_comm), &comm_rank);

	/*
	//if (esr_setup->rec_role == ROLE_reconstructing)
	if (comm_rank == 1)
	{
		//printf("Rank %d: A:\n", comm_rank);
		//for (int i = 0; i < A->size1; ++i)
		//{
		//	for (int j = 0; j < A->size2; ++j)
		//		printf("%e ", gsl_spmatrix_get(A->M, i, j));
		//	printf("\n");
		//}
		
		
		printf("Rank %d: z before reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%f ", pipelined_handle->z->data[i]);
		printf("]\n");

		printf("Rank %d: q before reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%f ", pipelined_handle->q->data[i]);
		printf("]\n");

		printf("Rank %d: s before reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%f ", pipelined_handle->s->data[i]);
		printf("]\n");

		printf("Rank %d: p before reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%f ", pipelined_handle->p->data[i]);
		printf("]\n");
		

		//printf("Rank %d: scalars before reconstruction: alpha = %f, gamma = %f, gamma_prev = %f, delta = %f, rr = %f\n", comm_rank, *(pipelined_handle->alpha), *(pipelined_handle->gamma), *(pipelined_handle->gamma_prev), *(pipelined_handle->delta), *(pipelined_handle->rr));

	}
	*/

	



	/*** break ***/

	//simulate loss of dynamic data
	// We can use it like this, with a branch and no wrappers,
	// until some more complicated scenario appears
	if (esr_setup->local_data_fate == FATE_erase)
	{
		//TODO: pack this into pipelined_erase_local_data function

		pipelined_solver_handle_set_zero(pipelined_handle);
		pipelined_state_copy_set_zero(pipelined_state_copy);
	}

	//static data retrieval would go here, if it's needed



	/*** reconstruct ***/

	pipelined_reconstruct(world_comm, pipelined_handle, A, P, b, pipelined_state_copy, esr_opts, esr_setup, spmv_ctx);


	/*
	//if (esr_setup->rec_role == ROLE_reconstructing)
	if (comm_rank == 1)
	{
		
		
		printf("Rank %d: z after reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%f ", pipelined_handle->z->data[i]);
		printf("]\n");

		printf("Rank %d: q after reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%f ", pipelined_handle->q->data[i]);
		printf("]\n");

		printf("Rank %d: s after reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%f ", pipelined_handle->s->data[i]);
		printf("]\n");

		printf("Rank %d: p after reconstruction: [ ", comm_rank);
		for (int i = 0; i < A->size1; ++i)
			printf("%f ", pipelined_handle->p->data[i]);
		printf("]\n");
		

		//printf("Rank %d: scalars after reconstruction: alpha = %f, gamma = %f, gamma_prev = %f, delta = %f, rr = %f\n", comm_rank, *(pipelined_handle->alpha), *(pipelined_handle->gamma), *(pipelined_handle->gamma_prev), *(pipelined_handle->delta), *(pipelined_handle->rr));
		
	}
	*/

	
	


	//redistribution and re-setup would go here, but are currently not necessary for the pipelined solver




	/*** Cleanup ***/
	free_esr_setup(esr_setup);


}
