
#include <stdio.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_spblas.h>
#include <assert.h>

#include <stdlib.h>
#include <unistd.h>

#include <mpi.h>
#include <mpi_one_sided_extension/mpi_win_pmem.h>
#include <mpi_one_sided_extension/defines.h>

#include "repeal_utils.h"
#include "repeal_mat_vec.h"
#include "repeal_spmv.h"
#include "repeal_commstrategy.h"
#include "repeal_recovery.h"
#include "repeal_pcg.h"
#include "repeal_options.h"
#include "repeal_save_state.h"

#include "persist_gsl_vector_mpi.h"
#include <omp.h>

struct PCG_info *create_pcg_info()
{
	struct PCG_info *pcg_info = malloc(sizeof(struct PCG_info));
	if (!pcg_info) failure("Allocation of PCG info struct failed.");

	pcg_info->converged_atol = 0;
	pcg_info->converged_rtol = 0;
	pcg_info->converged_iterations = 0;
	pcg_info->reconstruction_time = 0.;

	pcg_info->abs_residual_norm = -1;
	pcg_info->iterations = -1;

	return pcg_info;
}


void free_pcg_info(struct PCG_info *pcg_info)
{
	free(pcg_info);
}


int send_redundant_elements(const struct ESR_options *esr_opts, size_t j)
{
	//check if redundancy is switched on
	if (!(esr_opts->redundant_copies))
		return 0;

	//checkpointing: never need to send redundant elements
	if (esr_opts->reconstruction_strategy == REC_checkpoint_in_memory
    || esr_opts->reconstruction_strategy == REC_checkpoint_on_disk
    || esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_homogenous
    || esr_opts->reconstruction_strategy == REC_nvesr_homogenous
    || esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_RDMA
    || esr_opts->reconstruction_strategy == REC_nvesr_RDMA)
		return 0;
	
	//ESR without period: sending redundant elements in every iteration
	//Periodic ESR: sending redundant elements if j%period equals 0 or 1
	if (esr_opts->period == 0 || ( j > 1 && j % esr_opts->period <= 1 ) )
		return 1;
	
	return 0;
}

int save_state_now(const struct ESR_options *esr_opts, size_t j)
{

	//checkpointing: saving state if j%period equals 0
	if (esr_opts->reconstruction_strategy == REC_checkpoint_in_memory 
      || esr_opts->reconstruction_strategy == REC_checkpoint_on_disk
      || esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_homogenous
      || esr_opts->reconstruction_strategy == REC_nvesr_homogenous
      || esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_RDMA
      || esr_opts->reconstruction_strategy == REC_nvesr_RDMA) {
    // checkpointing in every iteration (probably not a good idea)
    if (esr_opts->period == 0) {
      return 1;
    }
    // periodic checkpointing: only if we're in the right iteration
    if (j > 0 && j % esr_opts->period == 0) return 1;

    return 0;
  }

    //check if redundancy is switched on
	if (!(esr_opts->redundant_copies))
	   return 0;

 	//ESR without period: saving state in every iteration
	//Periodic ESR: saving state if j%period equals 0 or 1
	if (esr_opts->reconstruction_strategy == REC_inplace && (esr_opts->period == 0 || ( j > 1 && j % esr_opts->period <= 1 ) ) )
		return 1;
	
	return 0;
}

// *************************************************
// Preconditioned Conjugate Gradient
// *************************************************

gsl_vector*
pcg(MPI_Comm comm, int comm_rank, int comm_size,
	const struct repeal_matrix *A, const struct repeal_matrix *P,
	const gsl_vector* b, gsl_vector* x_0,
	const struct solver_options *solver_opts,
	const int redundant_copies,
	const struct ESR_options *esr_opts, const struct NVESR_options *nvesr_opts,
	struct SPMV_context *spmv_ctx, struct PCG_info *pcg_info
)
{
	printf("DEBUG: redundant_copies is ------ %d\n", redundant_copies);
 	PMEMobjpool *pool = NULL;
	const size_t local_size = A->size1;
	printf("DEBUG: local_size is ------ %d\n", local_size);
	const size_t global_size = A->size2;
	int broken_iteration = esr_opts->broken_iteration;
	//int period = esr_opts->period;
	//int send_redundant_elements, save_state_now;

	const double atol = solver_opts->atol;
	const double rtol = solver_opts->rtol;
	const int max_iter = solver_opts->max_iter;
	const int verbose = solver_opts->verbose;
	const int output_rank = solver_opts->output_rank;
	//vectors needed by the solver
  gsl_vector* x 	= x_0;
  gsl_vector* r 	= gsl_vector_alloc(local_size);
  gsl_vector* z 	= gsl_vector_alloc(local_size);
  gsl_vector* p 	= gsl_vector_alloc(local_size);
  gsl_vector* Ap 	= gsl_vector_alloc(local_size);

  // Open the persistent object pool for read and write.
  if (esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_homogenous
    ||esr_opts->reconstruction_strategy == REC_nvesr_homogenous) {
    int pathlen = strlen(nvesr_opts->pmem_pool_path) + 22;
    char *path = calloc(pathlen, sizeof(char));
    snprintf(path, pathlen - 1, "%s_%d", nvesr_opts->pmem_pool_path, comm_rank);
	printf("DEBUG: creating pmem_obj for rank %d on path: %s\n", comm_rank, path);
    pool = pmemobj_create(path, POBJ_LAYOUT_NAME(vector_store), 1024*1024*1024, 0666);
    if (!pool) {
      failure("pmemobj_open failed");
    }
  }

  char* checkpoint_path = "";
  char checkpoint_file_path [256] ="";
  if (esr_opts->reconstruction_strategy == REC_checkpoint_on_disk) {
	 checkpoint_path = nvesr_opts->pmem_pool_path;		
	 sprintf(checkpoint_file_path, "%s/%s", checkpoint_path, "checkpoint.tmp");
   }
  bool win_ram_on = false;
  bool local_windows=true;
  bool one_window=false;

  MPI_Win win_ram;
  MPI_Win_pmem win;
  int num_of_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
  double * win_data;
  if (esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_RDMA) {
   if (!one_window && !win_ram_on && !local_windows) //many remote windows
	{  	
		int win_size = (4 * local_size+2) * sizeof(double);
		if (!local_windows && comm_rank != num_of_ranks -1) 
			win_size =1;

		MPI_Win_pmem *wins = (MPI_Win_pmem*)malloc(sizeof(MPI_Win_pmem)*num_of_ranks);
		for (int i=0; i<num_of_ranks; i++)
		{   
			if(comm_rank != num_of_ranks-1 && i!=comm_rank)
				continue;

			MPI_Info info;
			char root_path[256];
			
			if (comm_rank == num_of_ranks -1)
				sprintf(root_path, "%s/%d", nvesr_opts->pmem_pool_path, i);
			if (comm_rank != num_of_ranks -1)
				sprintf(root_path, "%s/%d", "/home/yonif/A_PAZI/NVMESR/fake_windows", comm_rank);
					
			// Create directory for checkpoints.
			if (mkdir(root_path, 0775) != 0 && errno != EEXIST) {
				mpi_log_error("Unable to create checkpoint directory.");
				printf("DEBUG: Unable to create checkpoint directory \n");
				MPI_Abort(MPI_COMM_WORLD, 1);
				MPI_Finalize_pmem();
			}

			printf("DEBUG: setting root path=%s, by rank=%d \n", root_path, comm_rank);
			MPI_Win_pmem_set_root_path(root_path);
			MPI_Info_create(&info);
			MPI_Info_set(info, "pmem_is_pmem", "true");
			MPI_Info_set(info, "pmem_allocate_in_ram", "false");
			//MPI_Info_set(info, "pmem_mode", "checkpoint");
			MPI_Info_set(info, "pmem_mode", "expand");
			MPI_Info_set(info, "pmem_name", "CG");
			MPI_Group world_group;
			MPI_Comm_group(MPI_COMM_WORLD, &world_group);
			MPI_Group my_group;	

			if (comm_rank == num_of_ranks-1 && i==num_of_ranks-1 )
			{
				int ranks [1];
				ranks[0]= num_of_ranks-1;
				MPI_Group_incl(world_group, 1, ranks, &my_group);
			}
			else
			{
				int ranks [2];		   
				ranks[0]= num_of_ranks-1;
				ranks[1]= i;
				MPI_Group_incl(world_group, 2, ranks, &my_group);
			}   		

			MPI_Comm my_comm;
			MPI_Comm_create_group(MPI_COMM_WORLD, my_group, 0, &my_comm);
			printf("DEBUG: creating window pmem, rank=%d \n", comm_rank);
			MPI_Win_allocate_pmem(win_size, sizeof(double), info, my_comm, &win_data, &(wins[i]));
			printf("DEBUG: window pmem created by rank=%d \n", comm_rank);
			MPI_Info_free(&info);
			win= wins[i]; //for the rank 'num_of_rnks-1' we get on the last iteration the right window
		}
	}
	else if (!one_window && !win_ram_on && local_windows) //local windows
	{
		int win_size = (4 * local_size+2) * sizeof(double);
		char root_path[256];
		sprintf(root_path, "%s/%d", nvesr_opts->pmem_pool_path, comm_rank);

		// Create directory for checkpoints.
		if (mkdir(root_path, 0775) != 0 && errno != EEXIST) {
			mpi_log_error("Unable to create checkpoint directory.");
			printf("DEBUG: Unable to create checkpoint directory \n");
			MPI_Abort(MPI_COMM_WORLD, 1);
			MPI_Finalize_pmem();
		}
        MPI_Win_pmem *wins = (MPI_Win_pmem*)malloc(sizeof(MPI_Win_pmem)*num_of_ranks);
		for (int i=0; i<num_of_ranks; i++)
		{   
			if(comm_rank != i)
				continue;
		    printf("DEBUG: setting root path=%s, by rank=%d \n", root_path, comm_rank);
		    MPI_Win_pmem_set_root_path(root_path);
		    MPI_Info info;
		    MPI_Info_create(&info);
		    MPI_Info_set(info, "pmem_is_pmem", "true");
		    MPI_Info_set(info, "pmem_allocate_in_ram", "false");
		    //MPI_Info_set(info, "pmem_mode", "checkpoint");
		    MPI_Info_set(info, "pmem_mode", "expand");
		    MPI_Info_set(info, "pmem_name", "CG");
		    MPI_Group world_group;
		    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
		    MPI_Group my_group;	

		    int ranks [1];
		    ranks[0]= comm_rank;
		    MPI_Group_incl(world_group, 1, ranks, &my_group);

		    MPI_Comm my_comm;
		    MPI_Comm_create_group(MPI_COMM_WORLD, my_group, 0, &my_comm);
		    printf("DEBUG: creating window pmem, rank=%d \n", comm_rank);
		    MPI_Win_allocate_pmem(win_size, sizeof(double), info, my_comm, &win_data, &(win[i]));
		    printf("DEBUG: window pmem created by rank=%d \n", comm_rank);
		    MPI_Info_free(&info);
			win = wins[i];
	}
	else if (one_window && !local_windows && !win_ram_on) //one remote window
	{
		int win_size = (4 * local_size+2) * sizeof(double)*num_of_ranks;
		if (comm_rank != num_of_ranks -1) 
			win_size =1;

		char root_path[256];
		if (comm_rank == num_of_ranks -1)
			sprintf(root_path, "%s/%d", nvesr_opts->pmem_pool_path, num_of_ranks -1);
		if (comm_rank != num_of_ranks -1)
			sprintf(root_path, "%s/%d", "/home/yonif/A_PAZI/NVMESR/fake_windows", comm_rank);

		// Create directory for checkpoints.
		if (mkdir(root_path, 0775) != 0 && errno != EEXIST) {
			mpi_log_error("Unable to create checkpoint directory.");
			printf("DEBUG: Unable to create checkpoint directory \n");
			MPI_Abort(MPI_COMM_WORLD, 1);
			MPI_Finalize_pmem();
		}
		
		MPI_Info info;		

		printf("DEBUG: setting root path, rank=%d \n", comm_rank);
		MPI_Win_pmem_set_root_path(root_path);
		MPI_Info_create(&info);
		MPI_Info_set(info, "pmem_is_pmem", "true");
		MPI_Info_set(info, "pmem_allocate_in_ram", "false");
		//MPI_Info_set(info, "pmem_mode", "checkpoint");
		MPI_Info_set(info, "pmem_mode", "expand");
		MPI_Info_set(info, "pmem_name", "CG");

		printf("DEBUG: creating window pmem, rank=%d \n", comm_rank);
		MPI_Win_allocate_pmem(win_size, sizeof(double), info, MPI_COMM_WORLD, &win_data, &win);
		printf("DEBUG: window pmem created \n");
		MPI_Info_free(&info);		
   }
	else if (win_ram_on && !local_windows) //ram windows are implemented only for remote windows
	{
		int win_size = (4 * local_size+2) * sizeof(double)*num_of_ranks;
		if (comm_rank != num_of_ranks -1) 
			win_size =1;

			printf("DEBUG: creating RAM window rank=%d \n", comm_rank);
			MPI_Win_allocate(win_size, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_data, &win_ram);
			printf("DEBUG: RAM window created by rank=%d \n", comm_rank);
		
	}
}
 
  if (esr_opts->reconstruction_strategy == REC_nvesr_RDMA) {
	// we create pmem_window on rank 0.
	// we should then have to promise that rank 0 is running on the nvram node.
	if (comm_rank ==0){
    // TO DO YONI create win. Maybe for each process we can define a seperate window? 
	}
  }

	/*
	//utility vectors
	gsl_vector* copy_1 = NULL; //safety copy from the last iteration
	gsl_vector* copy_2 = NULL; //safety copy from two iterations previous
	gsl_vector* help = NULL; //utility pointer for swapping
	if (redundant_copies)
	{
		copy_1 = gsl_vector_alloc(global_size);
		copy_2 = gsl_vector_alloc(global_size);
	}
	*/

	//scalars
    double alpha, beta, pAp, rz, rz_prev;
	double save_state_total_time = 0.0;
    size_t j;

	//time measurement (for reconstruction)
	double times[2] = {0., 0.};
	//vector norms to compute the relative residual
	double norm_b = norm(comm, b);
	double norm_r = 0.;
    printf("DEBUG: spmv\n");
    // r_0 = b - A*x_0
	spmv(comm, A, x, r, spmv_ctx, 0);
	printf("DEBUG: after spmv\n");
    gsl_blas_dscal(-1., r);
    if (gsl_blas_daxpy(1., b, r) != 0)
        failure("BLAS operation failed");

    // z_0 = M^(-1)*r_0 = P*r_0
	spmv(comm, P, r, z, spmv_ctx, 0);

    // p_0 = z_0
    if (gsl_blas_dcopy(z, p) != 0)
        failure("BLAS copy failed");
    rz = dot(comm, r, z);

	struct PCG_state_copy *pcg_state_copy = NULL; //struct to keep the copies for resilience
	struct PCG_solver_handle *pcg_handle = NULL; //handle struct that will allow the reconstruction functions to access the solver state
	if (redundant_copies || esr_opts->reconstruction_strategy == REC_nvesr_homogenous
	|| esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_homogenous ||
	 esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_RDMA ||
	 esr_opts->reconstruction_strategy == REC_checkpoint_on_disk)
	{
		//TO DO YONI!! Why this is for? maybe should be in mpi_pmem method as well? is allocating unnessecary RAM?
		pcg_state_copy = create_pcg_state_copy(pool, win, esr_opts, local_size, global_size);
		pcg_handle = create_pcg_solver_handle(x, r, z, p, Ap, &alpha, &beta, &pAp, &rz, &rz_prev, &j);
	}
	


	/*** iterate ***/

    for (j = 0; j < max_iter; j++)
	{
		if(comm_rank==0){
		   printf("DEBUG:----------------------------------------------------- starting iter number %d \n", j); 		
		}
		/*** SPMV (+sending redundant elements if required) ***/

        // alpha_j = <r_j, z_j> / <A*p_j, p_j>
		//send_redundant_elements = redundant_copies && (period == 0 || ( j > 1 && j % period <= 1 ) );
		spmv(comm, A, p, Ap, spmv_ctx, send_redundant_elements(esr_opts, j)); //this is where p can be saved

		/*** storage ***/
        //printf("DEBUG:----------------------------------------------------- save current state \n"); 
		//save_state_now = redundant_copies && (period == 0 || ( j > 1 && j % period <= 1 ) );
		if (save_state_now(esr_opts, j)) 
		{			
			double start = omp_get_wtime();
			if (win_ram_on)
			   pcg_save_current_state(pool, win_ram, win_ram_on, local_windows, one_window, checkpoint_file_path, comm, pcg_handle, spmv_ctx, pcg_state_copy, esr_opts);
			else
			   pcg_save_current_state(pool, win, win_ram_on, local_windows, one_window, checkpoint_file_path, comm, pcg_handle, spmv_ctx, pcg_state_copy, esr_opts);
			double end = omp_get_wtime();
			save_state_total_time  = save_state_total_time + end-start;
			printf("DEBUG: accumulated save time %f , rank=%d\n", save_state_total_time, comm_rank); 

		}

		//printf("DEBUG:----------------------------------------------------- after save current state \n"); 
		/*** break and recover ***/

		if ((redundant_copies || esr_opts->reconstruction_strategy == REC_nvesr_homogenous
	|| esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_homogenous ||
	 esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_RDMA || esr_opts->reconstruction_strategy == REC_checkpoint_on_disk) && j == broken_iteration)
		{
			if (verbose && comm_rank == output_rank) printf("Simulating node failure, reconstructing\n");
			if (j < 2) failure("Could not yet create enough safety copies to reconstruct");
			times[0] = MPI_Wtime();
            printf("DEBUG: before break, beta =  %f, rank = %d\n, ", *pcg_handle->beta, comm_rank); 
			printf("DEBUG: before break, rz =  %f, rank = %d\n, ", *pcg_handle->rz, comm_rank);
			//pcg_break_and_reconstruct(comm, comm_rank, comm_size, A, P, b, x, r, p, z, &beta, &rz, pcg_state_copy->buffer_copy_1, pcg_state_copy->buffer_copy_2, esr_opts, spmv_ctx, innertol);
			if (win_ram_on)
			pcg_break_and_recover(&comm, pool, win, true, checkpoint_file_path ,pcg_handle, A, P, b, pcg_state_copy, esr_opts, spmv_ctx);
            else
			pcg_break_and_recover(&comm, pool, win, false, checkpoint_file_path ,pcg_handle, A, P, b, pcg_state_copy, esr_opts, spmv_ctx);
			printf("DEBUG: after reconstruct, beta =  %f, rank = %d\n, ", *pcg_handle->beta, comm_rank); 
			printf("DEBUG: after reconstruct, rz =  %f, rank = %d\n, ", *pcg_handle->rz, comm_rank); 
			times[1] = MPI_Wtime();
			pcg_info->reconstruction_time = times[1] - times[0];
			if (verbose && comm_rank == output_rank) printf("Reconstruction complete!\n");

			//recompute Ap because that's supposed to get lost during the node failure, too
			// alpha_j = <r_j, z_j> / <A*p_j, p_j>
			spmv(comm, A, p, Ap, spmv_ctx, 0);
		}


		/*** compute more stuff ***/

        pAp = dot(comm, p, Ap);

        assert(pAp != 0.);
        alpha = rz / pAp;

        // x_(j+1) = x_j + alpha_j*p_j
        if (gsl_blas_daxpy(alpha, p, x) != 0)
            failure("BLAS error");

        // r_(j+1) = r_j - alpha_j*A*p_j
        if (gsl_blas_daxpy(-alpha, Ap, r) != 0)
            failure("BLAS error");

        //compute current residual norm
		norm_r = norm(comm, r);

		//verbosity output
		if (verbose && comm_rank == output_rank) printf("%ld: Residual norm: %.15e\n", j, norm_r);

		// Stop if converged
        if (norm_r <= atol) //absolute residual
		{
			pcg_info->converged_atol = 1;
			break;
		}
		if (norm_r/norm_b <= rtol) //relative residual
		{
			pcg_info->converged_rtol = 1;
			break;
		}

        // z_(j+1) = M^(-1)*r_(j+1) = P*r_(j+1)
		spmv(comm, P, r, z, spmv_ctx, 0);

		

		//if it breaks before here, beta_(j-1) is just what's currently in beta (except in iteration 0)

        // beta_j = <r_(j+1), z_(j+1)> / <r_j, z_j>
        rz_prev = rz;
        rz = dot(comm, r, z);
        beta = rz / rz_prev;

        // p_(j+1) = z_(j+1) + beta_j*p_j
        gsl_blas_dscal(beta, p);
        if (gsl_blas_daxpy(1., z, p) != 0)
            failure("BLAS error");
    }

	pcg_info->abs_residual_norm = norm_r;
    if (j == max_iter)
	{
        if (verbose && comm_rank == output_rank) printf("Maximum number of iterations reached.\n");
		pcg_info->converged_iterations = 1;
		pcg_info->iterations = j;
    }
	else
	{ //if we hit a break statement, j is the number of iterations minus 1
		pcg_info->iterations = j+1;
	}

	printf("Save State Total Time = %f, rank=%d \n", save_state_total_time, comm_rank);
	gsl_vector_free(r);
    gsl_vector_free(z);
    gsl_vector_free(p);
    gsl_vector_free(Ap);

 	if (redundant_copies)
	{
		//free(copy_1);
		//free(copy_2);

		free_pcg_state_copy(pcg_state_copy);
		free_pcg_solver_handle(pcg_handle);

		//disk checkpointing: delete the file that was used for storing the checkpoints
		//local operation, should only be performed by a single process
		if (comm_rank == 0 && esr_opts->reconstruction_strategy == REC_checkpoint_on_disk) {
			MPI_File_delete(checkpoint_file_path, MPI_INFO_NULL);
    }

    if (esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_homogenous
      ||esr_opts->reconstruction_strategy == REC_nvesr_homogenous) {
      pmemobj_close(pool);
    }
	if (esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_RDMA
      ||esr_opts->reconstruction_strategy == REC_nvesr_RDMA) {
	  if (win_ram_on)
	     MPI_Win_free(&win_ram);
      MPI_Win_free_pmem(&win);
    }
	}
  return x;
}

//*************************************************
//PIPELINED Preconditioned Conjugate Gradient
//*************************************************


gsl_vector*
pipelined_pcg(MPI_Comm comm, int comm_rank, int comm_size,
	const struct repeal_matrix *A, const struct repeal_matrix *P,
    const gsl_vector* b, gsl_vector* x_0,
	const struct solver_options *solver_opts,
	const int with_residual_replacement, const int redundant_copies,
	const struct ESR_options *esr_opts, struct SPMV_context *spmv_ctx,
	struct PCG_info *pcg_info
)
{
	size_t localsize = A->size1;
	size_t globalsize = A->size2;

	const double atol = solver_opts->atol;
	const double rtol = solver_opts->rtol;
	const int max_iter = solver_opts->max_iter;
	const int verbose = solver_opts->verbose;

	//vectors needed for PPCG
    gsl_vector* x = x_0;
    gsl_vector* r = gsl_vector_alloc(localsize);
	gsl_vector* u = gsl_vector_alloc(localsize);
	gsl_vector* w = gsl_vector_alloc(localsize);
	gsl_vector* x_prev = gsl_vector_alloc(localsize);
    gsl_vector* r_prev = gsl_vector_alloc(localsize);
	gsl_vector* u_prev = gsl_vector_alloc(localsize);
	gsl_vector* w_prev = gsl_vector_alloc(localsize);
	gsl_vector* m = gsl_vector_alloc(localsize);
	gsl_vector* n = gsl_vector_alloc(localsize);
    gsl_vector* z = gsl_vector_calloc(localsize);
    gsl_vector* q = gsl_vector_calloc(localsize);
	gsl_vector* s = gsl_vector_calloc(localsize);
	gsl_vector* p = gsl_vector_calloc(localsize);

	//scalars needed for PPCG
	double reducearray[3] = {0., 0., 0.}; //array to hold the elements that will have to be sent for the global reductions
	double *gamma_ptr = reducearray; //gamma is the first element in reducearray
	double *delta_ptr = reducearray+1; //delta is the second element in reducearray
	double *rr_ptr = reducearray+2; //<r,r> is the third element in reducearray
	double alpha = 0., beta, norm_r, norm_b, gamma_prev;

	//iteration counter
	size_t j = 0;

	//request for the nonblocking global reduction
	MPI_Request request;

	//time measurements
	double times[2] = {0., 0.};


	struct Pipelined_state_copy *pipelined_state_copy = NULL; //struct to keep the copies for resilience
	struct Pipelined_solver_handle *pipelined_handle = NULL; //handle struct that will allow the reconstruction functions to access the solver state
	if (redundant_copies)
	{
		pipelined_state_copy = create_pipelined_state_copy(esr_opts, localsize, globalsize);
		pipelined_handle = create_pipelined_solver_handle(m, n, r, r_prev, u, u_prev, w, w_prev, x, x_prev, z, q, s, p, &alpha, gamma_ptr, &gamma_prev, delta_ptr, rr_ptr, &j);
	}



	// ******************
	// set initial values
	// ******************

	
	//norm of b to compute the relative residual
	norm_b = norm(comm, b);

    // r_0 = b - A*x_0
	spmv(comm, A, x, r, spmv_ctx, 0);
    gsl_blas_dscal(-1., r);
    if (gsl_blas_daxpy(1., b, r) != 0)
        failure("BLAS operation failed");

    // u_0 = M^(-1)*r_0 = P*r_0
	spmv(comm, P, r, u, spmv_ctx, 0);

    //w_0 = A*u_0
	spmv(comm, A, u, w, spmv_ctx, 0);




	// **********
	// iterations
	// **********

	int broken_iteration = esr_opts->broken_iteration;

    while(1)
	{
		//gamma_j = <r_j, u_j> (local part)
		gamma_prev = *gamma_ptr;
		gsl_blas_ddot(r, u, gamma_ptr);

		//delta = <w_j, u_j> (local part)
		gsl_blas_ddot(w, u, delta_ptr);

		//rr = <r_j, r_j> (local part)
		gsl_blas_ddot(r, r, rr_ptr);

		//start global reductions
		MPI_Iallreduce(MPI_IN_PLACE, reducearray, 3, MPI_DOUBLE, MPI_SUM, comm, &request);

		//m_j = M^(-1)*w_j = P*w_j
		spmv(comm, P, w, m, spmv_ctx, 0);

		//n_j = A*m_j
		spmv(comm, A, m, n, spmv_ctx, send_redundant_elements(esr_opts, j));

		//finish global reductions
		MPI_Wait(&request, MPI_STATUS_IGNORE);

		//store safety copies if resilience is required
		if (save_state_now(esr_opts, j)) //safety copies: we need the values of the vector m from the last two iterations
		{
			pipelined_save_current_state(comm, pipelined_handle, spmv_ctx, pipelined_state_copy, esr_opts);			
		}

		//break and recover!
		if (redundant_copies && j == broken_iteration)
		{
			if (verbose && comm_rank == 0) printf("Simulating node failure, reconstructing\n");
			if (j < 2) failure("Could not yet create enough safety copies to reconstruct");
			times[0] = MPI_Wtime();

			pipelined_pcg_break_and_recover(&comm, pipelined_handle, A, P, b, pipelined_state_copy, esr_opts, spmv_ctx);

			times[1] = MPI_Wtime();
			pcg_info->reconstruction_time = times[1] - times[0];
			if (verbose && comm_rank == 0) printf("Reconstruction complete!\n");

			//recompute n because that isn't included in the reconstruction phase
			spmv(comm, A, m, n, spmv_ctx, 0);
		}


		if (j == 0)
		{
			//beta_j = 0
			beta = 0.;

			//alpha_j = gamma_j / delta
			alpha = *gamma_ptr / *delta_ptr;
		}
		else
		{
			//convergence check
			norm_r = sqrt(*rr_ptr); //compute norm(r)
			if (verbose && comm_rank == 0) printf("%ld: Residual norm: %.15e\n", j-1, norm_r); //verbosity output
			if (norm_r <= atol) //absolute residual
			{
				pcg_info->converged_atol = 1;
				break;
			}
			if (norm_r/norm_b <= rtol) //relative residual
			{
				pcg_info->converged_rtol = 1;
				break;
			}
			if (j == max_iter) //reached maximum number of iterations
			{
				pcg_info->converged_iterations = 1;
				if (verbose && comm_rank == 0) printf("Maximum number of iterations reached.\n");
				break;
			}

			//beta_j = gamma_j / gamma_(j-1)
			beta = *gamma_ptr / gamma_prev;

			//alpha_j = gamma_j / (delta - beta_j * gamma_j / alpha_(j-1))
			alpha = *gamma_ptr / (*delta_ptr - beta * *gamma_ptr / alpha);
		}


		//z_j = n_j + beta_j*z_(j-1)
		gsl_blas_dscal(beta, z);
		gsl_blas_daxpy(1., n, z);

		//q_j = m_j + beta_j*q_(j-1)
		gsl_blas_dscal(beta, q);
		gsl_blas_daxpy(1., m, q);

		//s_j = w_j + beta_j*s_(j-1)
		gsl_blas_dscal(beta, s);
		gsl_blas_daxpy(1., w, s);

		//p_j = u_j + beta_j*p_(j-1)
		gsl_blas_dscal(beta, p);
		gsl_blas_daxpy(1., u, p);

		//x_(j+1) = x_j + alpha_j*p_j
		gsl_blas_dcopy(x, x_prev); //keep a copy of the current state of the vector around for reconstruction
		gsl_blas_daxpy(alpha, p, x);

		//keep copies of r, u, and w around for replacement
		gsl_blas_dcopy(r, r_prev);
		gsl_blas_dcopy(u, u_prev);
		gsl_blas_dcopy(w, w_prev);

		if ( with_residual_replacement && ((j+1) % with_residual_replacement == 0) ) //recompute the actual values for r, u, and w
		{
			//if (verbose && comm_rank == 0) printf("Performing residual replacement\n");

			//r_(j+1) = b - A*x_(j+1)
			spmv(comm, A, x, r, spmv_ctx, 0);
			gsl_blas_dscal(-1., r);
			gsl_blas_daxpy(1., b, r);

			//u_(j+1) = P*r_(j+1)
			spmv(comm, P, r, u, spmv_ctx, 0);

			//w_(j+1) = A*u_(j+1)
			spmv(comm, A, u, w, spmv_ctx, 0);
		}
		else //"ordinary" update of r, u, and w
		{
			//r_(j+1) = r_j - alpha_j*s_j
			gsl_blas_daxpy(-alpha, s, r);

			//u_(j+1) = u_j - alpha_j*q_j
			gsl_blas_daxpy(-alpha, q, u);

			//w_(j+1) = w_j - alpha_j*z_j
			gsl_blas_daxpy(-alpha, z, w);
		}

		++j; //increase loop counter
    }


	// *******
	// cleanup
	// *******

	pcg_info->abs_residual_norm = norm_r;
	pcg_info->iterations = j;


	gsl_vector_free(r);
    gsl_vector_free(u);
    gsl_vector_free(w);
	gsl_vector_free(x_prev);
	gsl_vector_free(r_prev);
    gsl_vector_free(u_prev);
    gsl_vector_free(w_prev);
    gsl_vector_free(m);
	gsl_vector_free(n);
    gsl_vector_free(z);
    gsl_vector_free(p);
    gsl_vector_free(q);
	gsl_vector_free(s);

    //gsl_vector_free(buffer);
	//gsl_vector_free(sendbuf);

	
 	if (redundant_copies)
	{
		free_pipelined_state_copy(pipelined_state_copy);
		free_pipelined_solver_handle(pipelined_handle);
	}
	


	

    return x;
}




// *************************************************
// Wrapper
// *************************************************
gsl_vector *solve_linear_system(
	MPI_Comm comm, int comm_rank, int comm_size,
	const struct repeal_matrix *A, const struct repeal_matrix *P,
    const gsl_vector* b, gsl_vector* x_0,
	const struct solver_options *solver_opts, const struct ESR_options *esr_opts, const struct NVESR_options *nvesr_opts, struct SPMV_context *spmv_ctx,
	struct PCG_info *pcg_info
)
{
	
	if (solver_opts->solvertype == SOLVER_pipelined)
	{
		
		return pipelined_pcg(comm, comm_rank, comm_size,
			A, P, b, x_0,
			solver_opts,
			solver_opts->with_residual_replacement, esr_opts->redundant_copies,
			esr_opts, spmv_ctx,
			pcg_info);
			
	}
	else //SOLVER_pcg
	{
	
		return pcg(comm, comm_rank, comm_size,
			A, P, b, x_0,
			solver_opts,
			esr_opts->redundant_copies,
			esr_opts, nvesr_opts, spmv_ctx,
			pcg_info);
	
	}
	
}
