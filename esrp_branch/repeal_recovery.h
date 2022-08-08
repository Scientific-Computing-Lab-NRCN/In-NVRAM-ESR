#ifndef REPEAL_ESR
#define REPEAL_ESR

#include <libpmemobj.h>
#include <mpi_one_sided_extension/mpi_win_pmem.h>
#include <mpi_one_sided_extension/defines.h>
#include <mpi.h>
#include "repeal_utils.h"

void pcg_break_and_recover(
	MPI_Comm *world_comm,
	PMEMobjpool *pool,
	MPI_Win win,
	bool win_ram_on,
	char* checkpoint_file_path,
	struct PCG_solver_handle *pcg_handle, //dynamic solver data
	const struct repeal_matrix *A, const struct repeal_matrix *P, const gsl_vector *b, //static data
	struct PCG_state_copy *pcg_state_copy, //redundant copies etc.
	const struct ESR_options *esr_opts, const struct SPMV_context *spmv_ctx
);


void pipelined_pcg_break_and_recover(
	MPI_Comm *world_comm,
	struct Pipelined_solver_handle *pipelined_handle, //world communicator + dynamic solver data
	const struct repeal_matrix *A, const struct repeal_matrix *P, const gsl_vector *b, //static data
	struct Pipelined_state_copy *pipelined_state_copy, //redundant copies
	const struct ESR_options *esr_opts,
	const struct SPMV_context *spmv_ctx
);





#endif
