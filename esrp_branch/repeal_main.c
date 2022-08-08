#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_vector.h>
#include <mpi.h>
#include <utmpx.h>

#include "repeal_utils.h"
#include "repeal_options.h"
#include "repeal_mat_vec.h"
#include "repeal_spmv.h"
#include "repeal_commstrategy.h"
#include "repeal_pc.h"
//#include "repeal_io.h"
#include "repeal_input_generation.h"
#include "repeal_pcg.h"
#include "repeal_save_state.h"





//*************************************************
//main
//*************************************************


int
main(int argc, char* argv[])
{
    // **************************************Initialize**************************************

	//Initialize MPI
 /*   int provided_thread_support;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread_support);
	if (provided_thread_support != MPI_THREAD_MULTIPLE) {
      log_error("MPI doesn't support MPI_THREAD_MULTIPLE");
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize_pmem();
      return 1;
   }
  */
    //if mpi pmem
	/*
    int provided_thread_support;
    MPI_Init_thread_pmem(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread_support);
    if (provided_thread_support != MPI_THREAD_MULTIPLE) {
      log_error("MPI doesn't support MPI_THREAD_MULTIPLE");
	 printf("DEBUG: MPI doesn't support MPI_THREAD_MULTIPLE");
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize_pmem();
      return 1;
   }
   */

    MPI_Init(&argc, &argv);
   
   //else - no mpi pmem
    //MPI_Init(&argc, &argv);

	//duplicate MPI_COMM_WORLD
	MPI_Comm world_comm;
	MPI_Comm_dup(MPI_COMM_WORLD, &world_comm);

    int comm_size, comm_rank;
    MPI_Comm_size(world_comm, &comm_size);
    MPI_Comm_rank(world_comm, &comm_rank);

    printf("DEBUG: rank=%d on cpu %d\n", comm_rank, sched_getcpu());
	// Initialize random number generator
    gsl_rng_env_setup();
    gsl_rng_default_seed = (unsigned long) comm_rank;


	//Initialize time measurement

	double times[6];
	times[0] = MPI_Wtime();


	// **************************************set options**************************************

	//create structs storing the default options
	struct PC_options *pc_opts = get_pc_options();
	struct solver_options *solver_opts = get_solver_options();
	struct ESR_options *esr_opts = get_esr_options();
        struct NVESR_options *nvesr_opts = get_nvesr_options();
	struct Input_options *input_opts = get_input_options();

	enum COMM_strategy comm_strategy;


	//override default options according to command line input
	set_from_options(argc, argv, comm_rank, comm_size, &comm_strategy, pc_opts, solver_opts, esr_opts, nvesr_opts, input_opts);


	//verbosity output
	if (solver_opts->verbose && comm_rank == solver_opts->output_rank)
	{
		printf("Using options:\nAbsolute tolerance: %e\nRelative tolerance: %e\nInner relative tolerance: %e\nMax iterations: %ld\nPreconditioner: %d\nCommunication: %d\nMatrix file: %s\nRedundant copies: %d\n", solver_opts->atol, solver_opts->rtol, esr_opts->innertol, solver_opts->max_iter, pc_opts->pc_type, comm_strategy, input_opts->matrix_file_name, esr_opts->redundant_copies);
		if (solver_opts->solvertype == SOLVER_pipelined) printf("Using the Pipelined PCG solver\n");
		if (solver_opts->with_residual_replacement) printf("Performing residual replacement every %d iterations\n", solver_opts->with_residual_replacement);
		if (esr_opts->period) printf("Using a period of %d for resilience\n", esr_opts->period);
		fflush(stdout);
	}


	// **************************************setup**************************************

    
	times[1] = MPI_Wtime();
	//get system matrix
	gsl_spmatrix* A_data = get_input_problem(world_comm, input_opts);

	times[2] = MPI_Wtime();
    
	//get "workspace" for the sparse matrix vector product
	struct SPMV_context *spmv_ctx = get_spmv_context(world_comm, comm_size, comm_strategy, A_data->size1, A_data->size2);

	//if checkpointing was chosen, we need to allocate the appropriate receive buffers etc.
	if (esr_opts->reconstruction_strategy == REC_checkpoint_in_memory)
	{
		pcg_init_checkpointing_communication(comm_rank, comm_size, spmv_ctx, esr_opts);
	}

	//create repeal_matrix from the gsl_spmatrix
	struct repeal_matrix *A = repeal_matrix_create(world_comm, comm_rank, comm_size, A_data, comm_strategy, 1, spmv_ctx, esr_opts);
    
    //create preconditioner matrix
	struct repeal_matrix *P = get_preconditioner(world_comm, comm_rank, comm_size, A_data, pc_opts, comm_strategy, spmv_ctx);
 
	//create the rest of the linear system
    gsl_vector* b = create_random_vector_from_matrix(world_comm, A_data); // Right-hand-side vector with random values
    gsl_vector* x = gsl_vector_calloc(A->size1); // Initial guess = 0
    
	times[3] = MPI_Wtime();



    // **************************************Solve linear system**************************************

	struct PCG_info *pcg_info = create_pcg_info();

	x = solve_linear_system(
		world_comm, comm_rank, comm_size,
		A, P, b, x,
		solver_opts, esr_opts, nvesr_opts, spmv_ctx,
		pcg_info
	);

	times[4] = MPI_Wtime();

	//compute the residual Ax - b
	double final_residual = compute_residual_norm(world_comm, A, x, b, spmv_ctx);

	//output results
	double norm_b = norm(world_comm, b);
	if (comm_rank == solver_opts->output_rank)
	{
		char *results = "\nFinal residual norm is: %e\n"
		"Final relative residual is: %e\n"
		"Reported residual norm: %e\n"
		"Reported relative residual: %e\n"
		"Iterations needed: %d\n"
		"Convergence reason: %s\n";
		char *convergence_reason;
		if (pcg_info->converged_atol)
			convergence_reason = "absolute tolerance";
		else if (pcg_info->converged_rtol)
			convergence_reason = "relative tolerance";
		else if (pcg_info->converged_iterations)
			convergence_reason = "reached maximum number of iterations";
		else //this should never happen
			convergence_reason = "something went terribly wrong and broke the program";
		printf(results, final_residual, final_residual/norm_b, pcg_info->abs_residual_norm, pcg_info->abs_residual_norm/norm_b, pcg_info->iterations, convergence_reason);
	}

	times[5] = MPI_Wtime();

	//print time measurements
	if (comm_rank == solver_opts->output_rank)
	{
		char *time_results = "\nTime measurements taken on rank %d:\n"
		"Total runtime: %.5f seconds\n"
		"Reading from file: %.5f seconds\n"
		"PCG preparation: %.5f seconds\n"
		"Solving the linear system: %.5f seconds, including reconstruction time: %.5f seconds\n";
		printf(time_results,
			comm_rank,
			times[5] - times[0],
			times[2] - times[1],
			times[3] - times[2],
			times[4] - times[3],
			pcg_info->reconstruction_time
		);
	}



	// **************************************cleanup**************************************

    //free memory
    //printf("DEBUG: done free memory in main\n");
	free_pc_options(pc_opts);
	free_solver_options(solver_opts);
	free_esr_options(esr_opts);
        free_nvesr_options(nvesr_opts);
	free_spmv_context(spmv_ctx);
	free_pcg_info(pcg_info);

	repeal_matrix_free(A);
	repeal_matrix_free(P);

    gsl_vector_free(b);
    gsl_vector_free(x);

    //printf("DEBUG: done free memory in main\n");
	MPI_Comm_free(&world_comm);
	//printf("DEBUG: done Comm_free in main\n");
   // TO DO YONI - Invalid communicator error here, why??
   //TO DO YONI is pmem win
  // MPI_Finalize(); 
   MPI_Finalize_pmem();

	//printf("DEBUG: done - free mpi in main\n");
    exit(EXIT_SUCCESS);
}
