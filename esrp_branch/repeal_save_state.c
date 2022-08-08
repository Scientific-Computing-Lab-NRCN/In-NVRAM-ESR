#include <gsl/gsl_vector.h>
#include <gsl/gsl_spblas.h>
#include <stdio.h>
#include <mpi.h>
#include <mpi_one_sided_extension/mpi_win_pmem.h>
#include <mpi_one_sided_extension/defines.h>

#include <libpmemobj.h>
#include <omp.h>
#include "repeal_utils.h"
#include "repeal_save_state.h"


#include "persist_gsl_vector_mpi.h"


struct PCG_state_copy *create_pcg_state_copy(PMEMobjpool* pool, MPI_Win win, const struct ESR_options *esr_opts, size_t local_size, size_t global_size)
{
	struct PCG_state_copy *pcg_state_copy = malloc(sizeof(struct PCG_state_copy));
	if (!pcg_state_copy) failure("Allocation of PCG_state_copy struct failed");

	//in-memory checkpointing ---------------------------------------------------------------------------------------
	if (esr_opts->reconstruction_strategy == REC_checkpoint_in_memory)
	{
		//calculate total number of received elements
		int checkpoint_receive_buffer_size = 0;
		int next_buddy_to_save;
		for (int i = 0; i < esr_opts->redundant_copies; ++i)
		{
			next_buddy_to_save = esr_opts->buddies_that_I_save[i];
			checkpoint_receive_buffer_size += esr_opts->checkpoint_recvcounts[next_buddy_to_save];
		} 

		//allocate receive buffers for in-memory checkpointing
		pcg_state_copy->checkpoint_x = gsl_vector_alloc(checkpoint_receive_buffer_size);
		pcg_state_copy->checkpoint_r = gsl_vector_alloc(checkpoint_receive_buffer_size);
		pcg_state_copy->checkpoint_z = gsl_vector_alloc(checkpoint_receive_buffer_size);
		pcg_state_copy->checkpoint_p = gsl_vector_alloc(checkpoint_receive_buffer_size);

		//we will need to store local vectors in addition to the buffer copies
		pcg_state_copy->x_local = gsl_vector_alloc(local_size);
		pcg_state_copy->r_local = gsl_vector_alloc(local_size);
		pcg_state_copy->z_local = gsl_vector_alloc(local_size);
		pcg_state_copy->p_local = gsl_vector_alloc(local_size);

		//NULL out everything else
		pcg_state_copy->buffer_copy_1 = NULL;
		pcg_state_copy->buffer_copy_2 = NULL;
		pcg_state_copy->buffer_intermediate_copy = NULL;

 		// NVRAM buffers are not necessary.
    TOID_ASSIGN(pcg_state_copy->persistent_copy, OID_NULL);
	}

	//disk checkpointing ---------------------------------------------------------------------------------------
	else if (esr_opts->reconstruction_strategy == REC_checkpoint_on_disk)
	{
		//we will need to store local vectors for the reset to the checkpoint
		pcg_state_copy->x_local = gsl_vector_alloc(local_size);
		pcg_state_copy->r_local = gsl_vector_alloc(local_size);
		pcg_state_copy->z_local = gsl_vector_alloc(local_size);
		pcg_state_copy->p_local = gsl_vector_alloc(local_size);

		//everything else can be nulled out
		pcg_state_copy->checkpoint_x = NULL;
		pcg_state_copy->checkpoint_r = NULL;
		pcg_state_copy->checkpoint_z = NULL;
		pcg_state_copy->checkpoint_p = NULL;
		pcg_state_copy->buffer_copy_1 = NULL;
		pcg_state_copy->buffer_copy_2 = NULL;
		pcg_state_copy->buffer_intermediate_copy = NULL;
		
    TOID_ASSIGN(pcg_state_copy->persistent_copy, OID_NULL);
	}

    // NVRAM homogenous checkpointing ---------------------------------------------------------------------------------------
	else if (esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_homogenous)
	{
    TOID(struct PCG_persistent_state_copy) pcopy;
    pcopy = pcg_state_copy->persistent_copy = POBJ_ROOT(pool, struct PCG_persistent_state_copy);
    D_RW(pcopy)->checkpoint_x = pmem_vector_alloc(pool, local_size);
 		D_RW(pcopy)->checkpoint_r = pmem_vector_alloc(pool, local_size);
    D_RW(pcopy)->checkpoint_z = pmem_vector_alloc(pool, local_size);
    D_RW(pcopy)->checkpoint_p = pmem_vector_alloc(pool, local_size);

    // No need to assign values to homogenous NVESR copies.
    TOID_ASSIGN(D_RW(pcopy)->nvesr_buffer_copy_1, OID_NULL);
    TOID_ASSIGN(D_RW(pcopy)->nvesr_buffer_copy_2, OID_NULL);

    // Persist values.
    pmemobj_persist(pool, D_RW(pcopy), sizeof(struct PCG_persistent_state_copy));

		//we will need to store local vectors for the reset to the checkpoint
		pcg_state_copy->x_local = NULL; 
		pcg_state_copy->r_local = NULL; 
		pcg_state_copy->z_local = NULL; 
		pcg_state_copy->p_local = NULL; 

		//everything else can be nulled out
		pcg_state_copy->checkpoint_x = NULL;
		pcg_state_copy->checkpoint_r = NULL;
		pcg_state_copy->checkpoint_z = NULL;
		pcg_state_copy->checkpoint_p = NULL;
		pcg_state_copy->buffer_copy_1 = NULL;
		pcg_state_copy->buffer_copy_2 = NULL;
		pcg_state_copy->buffer_intermediate_copy = NULL;
	}
 // NVESR homogenous ---------------------------------------------------------------------------------------
 else if (esr_opts->reconstruction_strategy == REC_nvesr_homogenous) { 
    TOID(struct PCG_persistent_state_copy) pcopy;
    pcopy = pcg_state_copy->persistent_copy = POBJ_ROOT(pool, struct PCG_persistent_state_copy);
		D_RW(pcopy)->nvesr_buffer_copy_1 = pmem_vector_alloc(pool, local_size);
		D_RW(pcopy)->nvesr_buffer_copy_2 = pmem_vector_alloc(pool, local_size);

    // No need to assign values to homogenous nvram checkpoint copies.
    TOID_ASSIGN(D_RW(pcopy)->checkpoint_x, OID_NULL);
 		TOID_ASSIGN(D_RW(pcopy)->checkpoint_r, OID_NULL);
    TOID_ASSIGN(D_RW(pcopy)->checkpoint_z, OID_NULL);
    TOID_ASSIGN(D_RW(pcopy)->checkpoint_p, OID_NULL);
    // Persist values.
    pmemobj_persist(pool, D_RW(pcopy), sizeof(struct PCG_persistent_state_copy));
		// SPMV buffer copies over RAM are not needed for NVESR.
		pcg_state_copy->buffer_copy_1 = NULL;
		pcg_state_copy->buffer_copy_2 = NULL;

		// NVESR over pmdk does not support at the moment periodic. When introduced,
		// the x, r, p, and z vectors would be needed to be saved to nvram as well.
		// In addition, an additional, third, copy would be required as well.
		pcg_state_copy->buffer_intermediate_copy = NULL;
		pcg_state_copy->x_local = NULL;
		pcg_state_copy->r_local = NULL;
		pcg_state_copy->z_local = NULL;
		pcg_state_copy->p_local = NULL;

		// The checkpointing buffers are not required at the moment.
		pcg_state_copy->checkpoint_x = NULL;
		pcg_state_copy->checkpoint_r = NULL;
		pcg_state_copy->checkpoint_z = NULL;
		pcg_state_copy->checkpoint_p = NULL;     
	}

    // NVESR RDMA ---------------------------------------------------------------------------------------
    else if (esr_opts->reconstruction_strategy == REC_nvesr_RDMA) {  
    // TO DO YONI
   }
    // NVESR RDMA ---------------------------------------------------------------------------------------
    else if (esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_RDMA) {  
    //we will need to store local vectors for the reset to the checkpoint	    
		pcg_state_copy->x_local = gsl_vector_alloc(local_size);
		pcg_state_copy->r_local = gsl_vector_alloc(local_size);
		pcg_state_copy->z_local = gsl_vector_alloc(local_size);
		pcg_state_copy->p_local = gsl_vector_alloc(local_size);

		//everything else can be nulled out
		pcg_state_copy->checkpoint_x = NULL;
		pcg_state_copy->checkpoint_r = NULL;
		pcg_state_copy->checkpoint_z = NULL;
		pcg_state_copy->checkpoint_p = NULL;
		pcg_state_copy->buffer_copy_1 = NULL;
		pcg_state_copy->buffer_copy_2 = NULL;
		pcg_state_copy->buffer_intermediate_copy = NULL;
		
    TOID_ASSIGN(pcg_state_copy->persistent_copy, OID_NULL); //TO DO YONI
   }
	//ESR ---------------------------------------------------------------------------------------
	else
	{
		//SPMV buffer copies
		pcg_state_copy->buffer_copy_1 = gsl_vector_alloc(global_size);
		pcg_state_copy->buffer_copy_2 = gsl_vector_alloc(global_size);

		if (esr_opts->period)
		{
			//need a third buffer copy
			pcg_state_copy->buffer_intermediate_copy = gsl_vector_alloc(global_size);

			//we will neecreate_pcg_state_copyd to store local vectors in addition to the buffer copies
			pcg_state_copy->x_local = gsl_vector_alloc(local_size);
			pcg_state_copy->r_local = gsl_vector_alloc(local_size);
			pcg_state_copy->z_local = gsl_vector_alloc(local_size);
			pcg_state_copy->p_local = gsl_vector_alloc(local_size);
		}
		else
		{
			pcg_state_copy->buffer_intermediate_copy = NULL;
			pcg_state_copy->x_local = NULL;
			pcg_state_copy->r_local = NULL;
			pcg_state_copy->z_local = NULL;
			pcg_state_copy->p_local = NULL;
		}

		//don't need the checkpointing buffers
		pcg_state_copy->checkpoint_x = NULL;
		pcg_state_copy->checkpoint_r = NULL;
		pcg_state_copy->checkpoint_z = NULL;
		pcg_state_copy->checkpoint_p = NULL;

    // NVRAM buffers are not necessary.
    TOID_ASSIGN(pcg_state_copy->persistent_copy, OID_NULL);
	}
	return pcg_state_copy;
}

void pcg_state_copy_set_zero(struct PCG_state_copy *pcg_state_copy)
{
	if (pcg_state_copy->buffer_copy_1)
	{
		gsl_vector_set_zero(pcg_state_copy->buffer_copy_1);
		gsl_vector_set_zero(pcg_state_copy->buffer_copy_2);
	}
	if (pcg_state_copy->buffer_intermediate_copy)
	{
		gsl_vector_set_zero(pcg_state_copy->buffer_intermediate_copy);
	}

	if (pcg_state_copy->x_local)
	{
		gsl_vector_set_zero(pcg_state_copy->p_local);
		gsl_vector_set_zero(pcg_state_copy->x_local);
		gsl_vector_set_zero(pcg_state_copy->r_local);
		gsl_vector_set_zero(pcg_state_copy->z_local);
	}

	if (pcg_state_copy->checkpoint_x)
	{
		gsl_vector_set_zero(pcg_state_copy->checkpoint_p);
		gsl_vector_set_zero(pcg_state_copy->checkpoint_x);
		gsl_vector_set_zero(pcg_state_copy->checkpoint_r);
		gsl_vector_set_zero(pcg_state_copy->checkpoint_z);
	}

	pcg_state_copy->beta = 0.;
	pcg_state_copy->rz = 0;
	//TODO: what happens to iteration number?

}

void free_pcg_state_copy(struct PCG_state_copy *pcg_state_copy)
{
	//checkpointing
	if (pcg_state_copy->x_local)
	{
		gsl_vector_free(pcg_state_copy->x_local);
		gsl_vector_free(pcg_state_copy->r_local);
		gsl_vector_free(pcg_state_copy->z_local);
		gsl_vector_free(pcg_state_copy->p_local);

		if (pcg_state_copy->checkpoint_p)
		{
			gsl_vector_free(pcg_state_copy->checkpoint_x);
			gsl_vector_free(pcg_state_copy->checkpoint_r);
			gsl_vector_free(pcg_state_copy->checkpoint_z);
			gsl_vector_free(pcg_state_copy->checkpoint_p);
		}
	} else if (pcg_state_copy->buffer_copy_1) {
	//ESR
		gsl_vector_free(pcg_state_copy->buffer_copy_1);
		gsl_vector_free(pcg_state_copy->buffer_copy_2);

		// Periodic ESR
		if (pcg_state_copy->x_local)
		{
			gsl_vector_free(pcg_state_copy->buffer_intermediate_copy);
			gsl_vector_free(pcg_state_copy->x_local);
			gsl_vector_free(pcg_state_copy->r_local);
			gsl_vector_free(pcg_state_copy->z_local);
			gsl_vector_free(pcg_state_copy->p_local);
		}
	} else if (!TOID_IS_NULL(pcg_state_copy->persistent_copy)) {
		// NVESR
		if (!TOID_IS_NULL(D_RW(pcg_state_copy->persistent_copy)->nvesr_buffer_copy_1)) {
			pmem_vector_free(&D_RW(pcg_state_copy->persistent_copy)->nvesr_buffer_copy_1);
			pmem_vector_free(&D_RW(pcg_state_copy->persistent_copy)->nvesr_buffer_copy_2);
		}
		
    		// Checkpoint on NVRAM.
    		if (!TOID_IS_NULL(D_RW(pcg_state_copy->persistent_copy)->checkpoint_x)) {
    			pmem_vector_free(&D_RW(pcg_state_copy->persistent_copy)->checkpoint_x);
    			pmem_vector_free(&D_RW(pcg_state_copy->persistent_copy)->checkpoint_r);
    			pmem_vector_free(&D_RW(pcg_state_copy->persistent_copy)->checkpoint_z);
    			pmem_vector_free(&D_RW(pcg_state_copy->persistent_copy)->checkpoint_p);	
                }
  }

	free(pcg_state_copy);
}


struct PCG_solver_handle *create_pcg_solver_handle(gsl_vector *x, gsl_vector *r, gsl_vector *z, gsl_vector *p, gsl_vector *Ap, double *alpha, double *beta, double *pAp, double *rz, double *rz_prev, size_t *iteration)
{
	struct PCG_solver_handle *pcg_handle = malloc(sizeof(struct PCG_solver_handle));
	if (!pcg_handle) failure("Allocation of PCG_solver_handle struct failed");

	pcg_handle->x = x;
	pcg_handle->r = r;
	pcg_handle->z = z;
	pcg_handle->p = p;
	pcg_handle->Ap = Ap;
	pcg_handle->alpha = alpha;
	pcg_handle->beta = beta;
	pcg_handle->pAp = pAp;
	pcg_handle->rz = rz;
	pcg_handle->rz_prev = rz_prev;
	pcg_handle->iteration = iteration;

	return pcg_handle;
}

//to simulate data loss
void pcg_solver_handle_set_zero(struct PCG_solver_handle *pcg_handle)
{
	gsl_vector_set_zero(pcg_handle->x);
	gsl_vector_set_zero(pcg_handle->r);
	gsl_vector_set_zero(pcg_handle->z);
	gsl_vector_set_zero(pcg_handle->p);
	gsl_vector_set_zero(pcg_handle->Ap);
	*(pcg_handle->alpha) = 0.;
	*(pcg_handle->beta) = 0.;
	*(pcg_handle->pAp) = 0.;
	*(pcg_handle->rz) = 0.;
	*(pcg_handle->rz_prev) = 0.;
}

void free_pcg_solver_handle(struct PCG_solver_handle *pcg_handle)
{
	free(pcg_handle);
}


//creates the sendcounts etc. needed for checkpointing
void pcg_init_checkpointing_communication(int comm_rank, int comm_size, struct SPMV_context *spmv_ctx, struct ESR_options *esr_opts)
{
	//printf("Called pcg_init_checkpointing_communication\n");

	//allocate memory
	esr_opts->checkpoint_sendcounts = calloc(comm_size, sizeof(int));
	esr_opts->checkpoint_senddispls = calloc(comm_size, sizeof(int));
	esr_opts->checkpoint_recvcounts = calloc(comm_size, sizeof(int));
	esr_opts->checkpoint_recvdispls = calloc(comm_size, sizeof(int));

	if ( !(esr_opts->checkpoint_sendcounts) || !(esr_opts->checkpoint_senddispls) || !(esr_opts->checkpoint_recvcounts) || !(esr_opts->checkpoint_recvdispls) )
		failure("Memory allocation for checkpointing metadata failed");


	//fill with values

	//sendcounts: local vector size for all my buddies, 0 for all others
	int local_size = spmv_ctx->elements_per_process[comm_rank];
	int next_buddy;
	for (int i = 0; i < esr_opts->redundant_copies; ++i)
	{
		next_buddy = esr_opts->buddies_that_save_me[i];
		esr_opts->checkpoint_sendcounts[next_buddy] = local_size;
	}

	//senddisplacements are always zero

	//receivecounts: their local size for all ranks that have me as buddy, 0 for all others
	int current_offset = 0;
	for (int i = 0; i < esr_opts->redundant_copies; ++i)
	{
		//receivecount
		next_buddy = esr_opts->buddies_that_I_save[i];
		//printf("Rank %d: buddies_that_I_save[%d] = %d\n", comm_rank, i, next_buddy);
		esr_opts->checkpoint_recvcounts[next_buddy] = spmv_ctx->elements_per_process[next_buddy];

		//receivedisplacement
		esr_opts->checkpoint_recvdispls[next_buddy] = current_offset;

		//calculate displacement for the next round
		current_offset += spmv_ctx->elements_per_process[next_buddy];
	}


}

void create_vector_checkpoint_in_memory(MPI_Comm comm, gsl_vector *vec, gsl_vector *recvbuf, const struct ESR_options *esr_opts, MPI_Request *request)
{
	MPI_Ialltoallv(vec->data, esr_opts->checkpoint_sendcounts, esr_opts->checkpoint_senddispls, MPI_DOUBLE, recvbuf->data, esr_opts->checkpoint_recvcounts, esr_opts->checkpoint_recvdispls, MPI_DOUBLE, comm, request);
}

void create_vector_checkpoint_in_file(MPI_Comm comm, gsl_vector *vec, MPI_File fh, const int initial_offset, const struct SPMV_context *spmv_ctx, const struct ESR_options *esr_opts)
{
	//get the rank inside the communicator
	int comm_rank;
	MPI_Comm_rank(comm, &comm_rank);

	//calculate the offset where the data should be written
	//TODO: is this a byte offset?
	int offset = initial_offset + spmv_ctx->displacements[comm_rank];

	//start write to file
	//TODO: this should eventually be non-blocking, only my current OpenMPI installation doesn't support that
	MPI_File_write_at_all(fh, offset, vec->data, spmv_ctx->elements_per_process[comm_rank], MPI_DOUBLE, MPI_STATUS_IGNORE);

}



void pcg_save_local_state(const struct PCG_solver_handle *pcg_handle, struct PCG_state_copy *pcg_state_copy)
{
	gsl_blas_dcopy(pcg_handle->x, pcg_state_copy->x_local);
	gsl_blas_dcopy(pcg_handle->r, pcg_state_copy->r_local);
	gsl_blas_dcopy(pcg_handle->z, pcg_state_copy->z_local);
	gsl_blas_dcopy(pcg_handle->p, pcg_state_copy->p_local);

	pcg_state_copy->beta = *(pcg_handle->beta);
	pcg_state_copy->rz = *(pcg_handle->rz);
	pcg_state_copy->iteration = *(pcg_handle->iteration);
}

void pcg_save_current_state(PMEMobjpool* pool, MPI_Win_pmem win, bool win_ram_on, bool local_windows, bool one_window, char* checkpoint_file_path, MPI_Comm comm, struct PCG_solver_handle *pcg_handle, struct SPMV_context *spmv_ctx, struct PCG_state_copy *pcg_state_copy, const struct ESR_options *esr_opts)
{
	//assumption: we're only calling this function if the state actually needs to be saved in this iteration, i.e. no check of iteration number is necessary

	//in-memory checkpointing
	if (esr_opts->reconstruction_strategy == REC_checkpoint_in_memory)
	{
		//create copies of the local vectors and scalars
		pcg_save_local_state(pcg_handle, pcg_state_copy);

		//create checkpoints for x, r, z, p
		MPI_Request requests[4];
		create_vector_checkpoint_in_memory(comm, pcg_handle->x, pcg_state_copy->checkpoint_x, esr_opts, requests);
		create_vector_checkpoint_in_memory(comm, pcg_handle->r, pcg_state_copy->checkpoint_r, esr_opts, requests+1);
		create_vector_checkpoint_in_memory(comm, pcg_handle->z, pcg_state_copy->checkpoint_z, esr_opts, requests+2);
		create_vector_checkpoint_in_memory(comm, pcg_handle->p, pcg_state_copy->checkpoint_p, esr_opts, requests+3);
		MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
	}

	//disk checkpointing
	else if (esr_opts->reconstruction_strategy == REC_checkpoint_on_disk)
	{
		//get the rank inside the communicator
		int comm_rank;
		MPI_Comm_rank(comm, &comm_rank);

		//create copies of the local vectors and scalars
		pcg_save_local_state(pcg_handle, pcg_state_copy);

		//get global_size
		int global_size = spmv_ctx->buffer->size;
        bool mpi_file = true;
		if (mpi_file)
		{
			//open file
			MPI_File fh;		
			printf("DEBUG: opening file for checkpoint: %s\n",checkpoint_file_path);
			MPI_File_open(comm, checkpoint_file_path, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh); //mode: only need write access, file will be created it if does not exist
			//set the view so that offsets in the file will be computed as multiples of MPI_DOUBLE
			MPI_File_set_view(fh, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);      
			//create checkpoints for x, r, z, p
			//MPI_Request requests[4];
			create_vector_checkpoint_in_file(comm, pcg_handle->x, fh, 0, spmv_ctx, esr_opts);
			create_vector_checkpoint_in_file(comm, pcg_handle->r, fh, global_size,   spmv_ctx, esr_opts);
			create_vector_checkpoint_in_file(comm, pcg_handle->z, fh, 2*global_size, spmv_ctx, esr_opts);
			create_vector_checkpoint_in_file(comm, pcg_handle->p, fh, 3*global_size, spmv_ctx, esr_opts);
			//TODO: once File I/O is non-blocking, uncomment this
			//MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

			//rank 0: write the scalars to the file
			if (comm_rank == 0)
			{
				MPI_File_write_at(fh, 4*global_size, pcg_handle->beta, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
				MPI_File_write_at(fh, 4*global_size+1, pcg_handle->rz, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
			}		
			MPI_File_sync(fh); //TO DO YONI
			//close file
			MPI_File_close(&fh);
		}
		else
		{
	        int comm_size;
		    MPI_Comm_size(comm, &comm_size);
			int local_size = global_size/comm_size;
            FILE *fptr;
			char root_path[256];
			sprintf(root_path, "%s/%d.bin",checkpoint_file_path, comm_rank);
			fptr = fopen(root_path,"w");
			fwrite(pcg_handle->x, local_size*sizeof(double), 1, fptr); 
            fwrite(pcg_handle->r, local_size*sizeof(double), 1, fptr); 
			fwrite(pcg_handle->z, local_size*sizeof(double), 1, fptr); 
			fwrite(pcg_handle->p, local_size*sizeof(double), 1, fptr); 
            if (comm_rank == 0)
			{
			   fwrite(pcg_handle->beta, sizeof(double), 1, fptr); 	
			   fwrite(pcg_handle->rz, sizeof(double), 1, fptr); 
			}
			fflush(fptr);
			fclose(fptr); 
		}
	}
  else if (esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_homogenous) {
	pmem_vector_from_gsl_vector(pool, pcg_handle->x, &D_RW(pcg_state_copy->persistent_copy)->checkpoint_x);
	pmem_vector_from_gsl_vector(pool, pcg_handle->r, &D_RW(pcg_state_copy->persistent_copy)->checkpoint_r);
    pmem_vector_from_gsl_vector(pool, pcg_handle->z, &D_RW(pcg_state_copy->persistent_copy)->checkpoint_z);
	pmem_vector_from_gsl_vector(pool, pcg_handle->p, &D_RW(pcg_state_copy->persistent_copy)->checkpoint_p);

    D_RW(pcg_state_copy->persistent_copy)->beta = *(pcg_handle->beta);
    D_RW(pcg_state_copy->persistent_copy)->rz = *(pcg_handle->rz);
    pmemobj_persist(pool, D_RW(pcg_state_copy->persistent_copy), sizeof(struct PCG_persistent_state_copy));
  }
  else if (esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_RDMA) {
    // TO DO YONI
    //printf("DEBUG: checkpointing nvram rdma\n"); 
	int comm_rank;
	MPI_Comm_rank(comm, &comm_rank);
	int local_size = spmv_ctx->elements_per_process[comm_rank];
	//MPI_Aint target_disp = 0;
	long int target_disp = 0;
	if(one_window)			
	   target_disp = (4*local_size+2)*comm_rank;
	
	int num_of_ranks;
   	MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
	if (win_ram_on)
	{
		if (comm_rank != num_of_ranks-1)
		{
		int target = num_of_ranks-1;	
		MPI_Group group;
		MPI_Group comm_group;
		MPI_Comm_group(MPI_COMM_WORLD,&comm_group);
		int rank [1] = {num_of_ranks-1};
		MPI_Group_incl(comm_group,1,rank,&group);
		MPI_Win_start(group,0,win);
		//printf("DEBUG: target offset = %ld\n", target_disp);
		//double start_put_time = omp_get_wtime();	
        target_disp = put_gsl_vector_in_win(comm_rank, target, win, true, target_disp, pcg_handle->x, local_size);
		//double put_time = omp_get_wtime()-start_put_time;
		//printf("DEBUG: one put time=%f, rank=%d\n", put_time, comm_rank); 
		target_disp = put_gsl_vector_in_win(comm_rank, target, win, true, target_disp, pcg_handle->r, local_size);
		target_disp = put_gsl_vector_in_win(comm_rank, target, win, true, target_disp, pcg_handle->z, local_size);
		target_disp = put_gsl_vector_in_win(comm_rank, target, win, true, target_disp, pcg_handle->p, local_size);
		target_disp = put_scalar_in_win(comm_rank, target, win, true, target_disp, *pcg_handle->beta);	
		target_disp = put_scalar_in_win(comm_rank, target, win, true, target_disp, *pcg_handle->rz); 	
		MPI_Win_complete(win);	
		}

		if(comm_rank == num_of_ranks-1)
		{
		    MPI_Group group;
		    MPI_Group comm_group;
		    MPI_Comm_group(MPI_COMM_WORLD,&comm_group);
		    int * ranks  = (int*)malloc(sizeof(int)*(num_of_ranks-1));
			for(int i=0; i<num_of_ranks-1;i++)
			    ranks[i] = i;
		    MPI_Group_incl(comm_group,num_of_ranks-1,ranks,&group);	
		    MPI_Win_post(group,0,win);
            //Wait for epoch to end 
            MPI_Win_wait(win);
		}
	}

	else //pmem windows
	{
		int target = 0;	
		if(one_window && !local_windows)
			target = num_of_ranks-1;
        if(one_window)
		{		
		   if(comm_rank != num_of_ranks-1)
		   {
			MPI_Group group;
			MPI_Group comm_group;
			MPI_Comm_group(MPI_COMM_WORLD,&comm_group);
			int rank [1] = {num_of_ranks-1};
			MPI_Group_incl(comm_group,1,rank,&group);
			MPI_Win_start_pmem(group,0,win);
		   }		   
		}	
		
		else //local windows, and for now remote many windows...
		{

			MPI_Win_fence_pmem(MPI_MODE_NOPRECEDE, win);
		    //MPI_Win_lock_pmem(MPI_LOCK_EXCLUSIVE,target,MPI_MODE_NOCHECK,win);	

		}

        if(comm_rank != num_of_ranks-1)
		{
		target_disp = put_gsl_vector_in_win(comm_rank, target, win, false, target_disp, pcg_handle->x, local_size);
		target_disp = put_gsl_vector_in_win(comm_rank, target, win, false, target_disp, pcg_handle->r, local_size);
		target_disp = put_gsl_vector_in_win(comm_rank, target, win, false,  target_disp, pcg_handle->z, local_size);
		target_disp = put_gsl_vector_in_win(comm_rank, target, win, false, target_disp, pcg_handle->p, local_size);
		target_disp = put_scalar_in_win(comm_rank, target, win, false, target_disp, *pcg_handle->beta);
		target_disp = put_scalar_in_win(comm_rank, target, win, false, target_disp, *pcg_handle->rz); 
		}
		//double unlocking_time = omp_get_wtime();
		//printf("DEBUG: TRYING unlocking time=%f, rank=%d\n", unlocking_time, comm_rank); 
		
		//nlocking_time = omp_get_wtime();
		//printf("DEBUG: unlocking time=%f, rank=%d\n", unlocking_time, comm_rank); 
		//printf("DEBUG: working in locking time=%f, rank=%d\n", unlocking_time-locking_time, comm_rank); 
		//}		
	    if(one_window)
		{				   	
		   if(comm_rank != num_of_ranks-1)
		      MPI_Win_complete_pmem(win);	
		   if(comm_rank == num_of_ranks-1)
		   {
		      MPI_Group group;
		      MPI_Group comm_group;
		      MPI_Comm_group(MPI_COMM_WORLD,&comm_group);
		      int * ranks  = (int*)malloc(sizeof(int)*(num_of_ranks-1));
			  for(int i=0; i<num_of_ranks-1;i++)
			     ranks[i] = i;
		      MPI_Group_incl(comm_group,num_of_ranks-1,ranks,&group);	
		      MPI_Win_post_pmem(group,0,win);
              //Wait for epoch to end 
              MPI_Win_wait_pmem_persist(win);
		   }
		   
		}

		else
		{
		   MPI_Win_fence_pmem_persist(MPI_MODE_NOSUCCEED, win);
		   //MPI_Win_unlock_pmem(target,win);		   
		}
	}

  }
  else if (esr_opts->reconstruction_strategy == REC_nvesr_RDMA) {
    // TO DO YONI
  }
  // Homogenous NVESR
  else if (esr_opts->reconstruction_strategy == REC_nvesr_homogenous) {
    // NVESR is not periodic at the moment. Therefore, the ignore the period
    // settings for esrp, and store to memory only the required vectors for
    // exact state reconstruction over nvram.

    // We can simply shift the buffer copies over NVRAM, the rest are
    // reread or rewritten.	
	printf("DEBUG:----------------------------------------------------- 11\n"); 
    TOID(struct PCG_persistent_state_copy) pcopy = pcg_state_copy->persistent_copy;
	printf("DEBUG:----------------------------------------------------- 22\n"); 
    TOID(struct pmem_vector) help = D_RW(pcopy)->nvesr_buffer_copy_2;
	printf("DEBUG:----------------------------------------------------- 33\n"); 
    D_RW(pcopy)->nvesr_buffer_copy_2 = D_RW(pcopy)->nvesr_buffer_copy_1;
    D_RW(pcopy)->nvesr_buffer_copy_1 = help;
	printf("DEBUG:----------------------------------------------------- persist\n"); 
    pmemobj_persist(pool, D_RW(pcg_state_copy->persistent_copy), sizeof(struct PCG_persistent_state_copy));
	printf("DEBUG:----------------------------------------------------- after persist\n"); 
    pmem_vector_from_gsl_vector(pool, pcg_handle->p, &D_RW(pcopy)->nvesr_buffer_copy_1);
  }

	//periodic ESR
	else if (esr_opts->period)
	{
		if (*(pcg_handle->iteration) % esr_opts->period == 0)
		{
			//saving current buffer, but leaving the actual state alone because we need two subsequent iterations to reconstruct
			gsl_vector *help = pcg_state_copy->buffer_intermediate_copy;
			pcg_state_copy->buffer_intermediate_copy = spmv_ctx->buffer;
			spmv_ctx->buffer = help;
		}
		else //pcg_handle->iteration % esr_opts->period == 1
		{
			//shift everything, take care of the third copy created in the previous step
			gsl_vector *help = pcg_state_copy->buffer_copy_2;
			pcg_state_copy->buffer_copy_2 = pcg_state_copy->buffer_intermediate_copy;
			pcg_state_copy->buffer_intermediate_copy = pcg_state_copy->buffer_copy_1;
			pcg_state_copy->buffer_copy_1 = spmv_ctx->buffer;
			spmv_ctx->buffer = help;

			//create copies of the local vectors and scalars
			pcg_save_local_state(pcg_handle, pcg_state_copy);
		}
		//note: reconstruction always uses copy_1 and copy_2, no matter at which iteration the node failure occurs
	}

	//ESR without period
	else
	{
		//can simply shift the buffer copies
		gsl_vector *help = pcg_state_copy->buffer_copy_2;
		pcg_state_copy->buffer_copy_2 = pcg_state_copy->buffer_copy_1;
		pcg_state_copy->buffer_copy_1 = spmv_ctx->buffer;
		spmv_ctx->buffer = help;
	}

}


//reset the solver to a previously saved state
void pcg_solver_reset_to_saved_state(struct PCG_solver_handle *pcg_handle, struct PCG_state_copy *pcg_state_copy)
{
	//the parts of the solver that are not reset here should get overwritten during the next PCG iteration anyways
	gsl_blas_dcopy(pcg_state_copy->x_local, pcg_handle->x);
	gsl_blas_dcopy(pcg_state_copy->r_local, pcg_handle->r);
	gsl_blas_dcopy(pcg_state_copy->z_local, pcg_handle->z);
	gsl_blas_dcopy(pcg_state_copy->p_local, pcg_handle->p);
	*(pcg_handle->beta) = pcg_state_copy->beta;
	*(pcg_handle->rz) = pcg_state_copy->rz;

	//TODO: need to adjust iteration count on ALL ranks without having them enter the recovery phase again
	//*(pcg_handle->iteration) = pcg_state_copy->iteration;
}






// *************************************************
// PIPELINED Preconditioned Conjugate Gradient
// *************************************************


struct Pipelined_state_copy *create_pipelined_state_copy(const struct ESR_options *esr_opts, size_t local_size, size_t global_size)
{
	struct Pipelined_state_copy *pipelined_state_copy = malloc(sizeof(struct Pipelined_state_copy));
	if (!pipelined_state_copy) failure("Allocation of Pipelined_state_copy struct failed");

	//SPMV buffer copies
	pipelined_state_copy->buffer_copy_1 = gsl_vector_alloc(global_size);
	pipelined_state_copy->buffer_copy_2 = gsl_vector_alloc(global_size);

	if (esr_opts->period)
	{
		pipelined_state_copy->buffer_intermediate_copy = gsl_vector_alloc(global_size);

		pipelined_state_copy->m = gsl_vector_alloc(local_size);
		pipelined_state_copy->n = gsl_vector_alloc(local_size);
		pipelined_state_copy->r = gsl_vector_alloc(local_size);
		pipelined_state_copy->r_prev = gsl_vector_alloc(local_size);
		pipelined_state_copy->u = gsl_vector_alloc(local_size);
		pipelined_state_copy->u_prev = gsl_vector_alloc(local_size);
		pipelined_state_copy->w = gsl_vector_alloc(local_size);
		pipelined_state_copy->w_prev = gsl_vector_alloc(local_size);
		pipelined_state_copy->x = gsl_vector_alloc(local_size);
		pipelined_state_copy->x_prev = gsl_vector_alloc(local_size);
		pipelined_state_copy->z = gsl_vector_alloc(local_size);
		pipelined_state_copy->q = gsl_vector_alloc(local_size);
		pipelined_state_copy->s = gsl_vector_alloc(local_size);
 		pipelined_state_copy->p = gsl_vector_alloc(local_size);
	}
	else
	{
 		// set everything else to NULL
		pipelined_state_copy->buffer_intermediate_copy = NULL;

		pipelined_state_copy->m = NULL;
		pipelined_state_copy->n = NULL;
		pipelined_state_copy->r = NULL;
		pipelined_state_copy->r_prev = NULL;
		pipelined_state_copy->u = NULL;
		pipelined_state_copy->u_prev = NULL;
		pipelined_state_copy->w = NULL;
		pipelined_state_copy->w_prev = NULL;
		pipelined_state_copy->x = NULL;
		pipelined_state_copy->x_prev = NULL;
		pipelined_state_copy->z = NULL;
		pipelined_state_copy->q = NULL;
		pipelined_state_copy->s = NULL;
		pipelined_state_copy->p = NULL;
	}


	return pipelined_state_copy;
}

void pipelined_state_copy_set_zero(struct Pipelined_state_copy *pipelined_state_copy)
{
	gsl_vector_set_zero(pipelined_state_copy->buffer_copy_1);
	gsl_vector_set_zero(pipelined_state_copy->buffer_copy_2);

	//memory for periodic version
	if (pipelined_state_copy->buffer_intermediate_copy)
	{
		gsl_vector_set_zero(pipelined_state_copy->buffer_intermediate_copy);

		gsl_vector_set_zero(pipelined_state_copy->m);
		gsl_vector_set_zero(pipelined_state_copy->n);
		gsl_vector_set_zero(pipelined_state_copy->w);
		gsl_vector_set_zero(pipelined_state_copy->w_prev);
		gsl_vector_set_zero(pipelined_state_copy->u);
		gsl_vector_set_zero(pipelined_state_copy->u_prev);
		gsl_vector_set_zero(pipelined_state_copy->r);
		gsl_vector_set_zero(pipelined_state_copy->r_prev);
		gsl_vector_set_zero(pipelined_state_copy->x);
		gsl_vector_set_zero(pipelined_state_copy->x_prev);

		gsl_vector_set_zero(pipelined_state_copy->z);
		gsl_vector_set_zero(pipelined_state_copy->q);
		gsl_vector_set_zero(pipelined_state_copy->s);
		gsl_vector_set_zero(pipelined_state_copy->p);

		pipelined_state_copy->alpha = 0.;
		pipelined_state_copy->gamma_prev = 0.;
		pipelined_state_copy->gamma = 0.;
		pipelined_state_copy->delta = 0.;
		pipelined_state_copy->rr = 0.;
	}


}


void free_pipelined_state_copy(struct Pipelined_state_copy *pipelined_state_copy)
{
	gsl_vector_free(pipelined_state_copy->buffer_copy_1);
	gsl_vector_free(pipelined_state_copy->buffer_copy_2);

	//the memory that is only allocated for the periodic version
	if (pipelined_state_copy->buffer_intermediate_copy)
	{
		gsl_vector_free(pipelined_state_copy->buffer_intermediate_copy);
		gsl_vector_free(pipelined_state_copy->m);
		gsl_vector_free(pipelined_state_copy->n);
		gsl_vector_free(pipelined_state_copy->r);
		gsl_vector_free(pipelined_state_copy->r_prev);
		gsl_vector_free(pipelined_state_copy->u);
		gsl_vector_free(pipelined_state_copy->u_prev);
		gsl_vector_free(pipelined_state_copy->w);
		gsl_vector_free(pipelined_state_copy->w_prev);
		gsl_vector_free(pipelined_state_copy->x);
		gsl_vector_free(pipelined_state_copy->x_prev);
		gsl_vector_free(pipelined_state_copy->z);
		gsl_vector_free(pipelined_state_copy->q);
		gsl_vector_free(pipelined_state_copy->s);
		gsl_vector_free(pipelined_state_copy->p);
	}

	free(pipelined_state_copy);
}




struct Pipelined_solver_handle *create_pipelined_solver_handle(gsl_vector *m, gsl_vector *n, gsl_vector *r, gsl_vector *r_prev, gsl_vector *u, gsl_vector *u_prev, gsl_vector *w, gsl_vector *w_prev, gsl_vector *x, gsl_vector *x_prev, gsl_vector *z, gsl_vector *q, gsl_vector *s, gsl_vector *p, double *alpha, double *gamma, double *gamma_prev, double *delta, double *rr, size_t *iteration)
{
	struct Pipelined_solver_handle *pipelined_solver_handle = malloc(sizeof(struct Pipelined_solver_handle));
	if (!pipelined_solver_handle) failure("Allocation of Pipelined_solver_handle struct failed");

	//vectors
	pipelined_solver_handle->m = m;
	pipelined_solver_handle->n = n;
	pipelined_solver_handle->r = r;
	pipelined_solver_handle->r_prev = r_prev;
	pipelined_solver_handle->u = u;
	pipelined_solver_handle->u_prev = u_prev;
	pipelined_solver_handle->w = w;
	pipelined_solver_handle->w_prev = w_prev;
	pipelined_solver_handle->x = x;
	pipelined_solver_handle->x_prev = x_prev;

	pipelined_solver_handle->z = z;
	pipelined_solver_handle->q = q;
	pipelined_solver_handle->s = s;
	pipelined_solver_handle->p = p;

	//scalars
	pipelined_solver_handle->alpha = alpha;
	pipelined_solver_handle->gamma = gamma;
	pipelined_solver_handle->gamma_prev = gamma_prev;
	pipelined_solver_handle->delta = delta;
	pipelined_solver_handle->rr = rr;

	//iteration counter
	pipelined_solver_handle->iteration = iteration;

	return pipelined_solver_handle;
}


void pipelined_solver_handle_set_zero(struct Pipelined_solver_handle *pipelined_solver_handle)
{
	gsl_vector_set_zero(pipelined_solver_handle->m);
	gsl_vector_set_zero(pipelined_solver_handle->n);
	gsl_vector_set_zero(pipelined_solver_handle->w);
	gsl_vector_set_zero(pipelined_solver_handle->w_prev);
	gsl_vector_set_zero(pipelined_solver_handle->u);
	gsl_vector_set_zero(pipelined_solver_handle->u_prev);
	gsl_vector_set_zero(pipelined_solver_handle->r);
	gsl_vector_set_zero(pipelined_solver_handle->r_prev);
	gsl_vector_set_zero(pipelined_solver_handle->x);
	gsl_vector_set_zero(pipelined_solver_handle->x_prev);

	gsl_vector_set_zero(pipelined_solver_handle->z);
	gsl_vector_set_zero(pipelined_solver_handle->q);
	gsl_vector_set_zero(pipelined_solver_handle->s);
	gsl_vector_set_zero(pipelined_solver_handle->p);

	*(pipelined_solver_handle->alpha) = 0.;
	*(pipelined_solver_handle->gamma_prev) = 0.;
	*(pipelined_solver_handle->gamma) = 0.;
	*(pipelined_solver_handle->delta) = 0.;
	*(pipelined_solver_handle->rr) = 0.;
}


void free_pipelined_solver_handle(struct Pipelined_solver_handle *pipelined_solver_handle)
{
	free(pipelined_solver_handle);
}


void pipelined_save_local_state(struct Pipelined_solver_handle *pipelined_handle, struct Pipelined_state_copy *pipelined_state_copy)
{
	//copy local vectors
	gsl_blas_dcopy(pipelined_handle->m, pipelined_state_copy->m);
	gsl_blas_dcopy(pipelined_handle->n, pipelined_state_copy->n);
	gsl_blas_dcopy(pipelined_handle->r, pipelined_state_copy->r);
	gsl_blas_dcopy(pipelined_handle->r_prev, pipelined_state_copy->r_prev);
	gsl_blas_dcopy(pipelined_handle->u, pipelined_state_copy->u);
	gsl_blas_dcopy(pipelined_handle->u_prev, pipelined_state_copy->u_prev);
	gsl_blas_dcopy(pipelined_handle->w, pipelined_state_copy->w);
	gsl_blas_dcopy(pipelined_handle->w_prev, pipelined_state_copy->w_prev);
	gsl_blas_dcopy(pipelined_handle->x, pipelined_state_copy->x);
	gsl_blas_dcopy(pipelined_handle->x_prev, pipelined_state_copy->x_prev);
	gsl_blas_dcopy(pipelined_handle->z, pipelined_state_copy->z);
	gsl_blas_dcopy(pipelined_handle->q, pipelined_state_copy->q);
	gsl_blas_dcopy(pipelined_handle->s, pipelined_state_copy->s);
	gsl_blas_dcopy(pipelined_handle->p, pipelined_state_copy->p);

	//copy scalars
	pipelined_state_copy->alpha = *(pipelined_handle->alpha);
	pipelined_state_copy->gamma = *(pipelined_handle->gamma);
	pipelined_state_copy->gamma_prev = *(pipelined_handle->gamma_prev);
	pipelined_state_copy->delta = *(pipelined_handle->delta);
	pipelined_state_copy->rr = *(pipelined_handle->rr);
}



void pipelined_save_current_state(MPI_Comm comm, struct Pipelined_solver_handle *pipelined_handle, struct SPMV_context *spmv_ctx, struct Pipelined_state_copy *pipelined_state_copy, const struct ESR_options *esr_opts)
{
	//currently only inplace ESR available

	//periodic storage
	if (esr_opts->period)
	{
		if (*(pipelined_handle->iteration) % esr_opts->period == 0)
		{
			//saving current buffer, but leaving the actual state alone because we need two subsequent iterations to reconstruct
			gsl_vector *help = pipelined_state_copy->buffer_intermediate_copy;
			pipelined_state_copy->buffer_intermediate_copy = spmv_ctx->buffer;
			spmv_ctx->buffer = help;
		}
		else //pipelined_handle->iteration % esr_opts->period == 1
		{
			//shift everything, take care of the third copy created in the previous step
			gsl_vector *help = pipelined_state_copy->buffer_copy_2;
			pipelined_state_copy->buffer_copy_2 = pipelined_state_copy->buffer_intermediate_copy;
			pipelined_state_copy->buffer_intermediate_copy = pipelined_state_copy->buffer_copy_1;
			pipelined_state_copy->buffer_copy_1 = spmv_ctx->buffer;
			spmv_ctx->buffer = help;

			//create copies of the local vectors and scalars
			pipelined_save_local_state(pipelined_handle, pipelined_state_copy);
		}
		//note: reconstruction always uses copy_1 and copy_2, no matter at which iteration the node failure occurs
	}

	//saving state in every iteration
	else
	{
		//swap the pointers to the buffer copies around
		gsl_vector *help = pipelined_state_copy->buffer_copy_2;
		pipelined_state_copy->buffer_copy_2 = pipelined_state_copy->buffer_copy_1;
		pipelined_state_copy->buffer_copy_1 = spmv_ctx->buffer;
		spmv_ctx->buffer = help;
	}
}


//reset the solver to a previously saved state
void pipelined_solver_reset_to_saved_state(struct Pipelined_solver_handle *pipelined_handle, struct Pipelined_state_copy *pipelined_state_copy)
{


	//copy local vectors
	gsl_blas_dcopy(pipelined_state_copy->m, pipelined_handle->m);
	gsl_blas_dcopy(pipelined_state_copy->n, pipelined_handle->n);
	gsl_blas_dcopy(pipelined_state_copy->r, pipelined_handle->r);
	gsl_blas_dcopy(pipelined_state_copy->r_prev, pipelined_handle->r_prev);
	gsl_blas_dcopy(pipelined_state_copy->u, pipelined_handle->u);
	gsl_blas_dcopy(pipelined_state_copy->u_prev, pipelined_handle->u_prev);
	gsl_blas_dcopy(pipelined_state_copy->w, pipelined_handle->w);
	gsl_blas_dcopy(pipelined_state_copy->w_prev, pipelined_handle->w_prev);
	gsl_blas_dcopy(pipelined_state_copy->x, pipelined_handle->x);
	gsl_blas_dcopy(pipelined_state_copy->x_prev, pipelined_handle->x_prev);
	gsl_blas_dcopy(pipelined_state_copy->z, pipelined_handle->z);
	gsl_blas_dcopy(pipelined_state_copy->q, pipelined_handle->q);
	gsl_blas_dcopy(pipelined_state_copy->s, pipelined_handle->s);
	gsl_blas_dcopy(pipelined_state_copy->p, pipelined_handle->p);




	//copy scalars
	*(pipelined_handle->alpha) = pipelined_state_copy->alpha;
	*(pipelined_handle->gamma) = pipelined_state_copy->gamma;
	*(pipelined_handle->gamma_prev) = pipelined_state_copy->gamma_prev;
	*(pipelined_handle->delta) = pipelined_state_copy->delta;
	*(pipelined_handle->rr) = pipelined_state_copy->rr;







	//for now: not resetting the iteration count

}


