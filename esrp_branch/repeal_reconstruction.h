#ifndef REPEAL_RECOVERY
#define REPEAL_RECOVERY

#include <libpmemobj.h>
#include <mpi.h>
#include <mpi_one_sided_extension/mpi_win_pmem.h>
#include <mpi_one_sided_extension/defines.h>
#include "repeal_utils.h"

void pcg_reconstruct(MPI_Comm *world_comm,
	PMEMobjpool* pool,
	MPI_Win win,
	bool win_ram_on,
	char * checkpoint_file_name,
	//ynamic solver data
	struct PCG_solver_handle *pcg_handle, 

	//static data
	const struct repeal_matrix *A, 
	const struct repeal_matrix *P, 
	const gsl_vector *b, 

	//redundant copies etc.
	struct PCG_state_copy *pcg_state_copy, 

	//user-defined options, utility structs, ...
	const struct ESR_options *esr_opts, 
	struct ESR_setup * esr_setup, 
	const struct SPMV_context *spmv_ctx
);


void pipelined_reconstruct(
	MPI_Comm *world_comm,

	//dynamic solver data
	struct Pipelined_solver_handle *pipelined_handle, 

	//static data
	const struct repeal_matrix *A, 
	const struct repeal_matrix *P, 
	const gsl_vector *b, 

	//redundant copies etc.
	struct Pipelined_state_copy *pipelined_state_copy,

	//user-defined options, utility structs, ...
	const struct ESR_options *esr_opts, 
	struct ESR_setup * esr_setup,
	const struct SPMV_context *spmv_ctx
);




#endif
