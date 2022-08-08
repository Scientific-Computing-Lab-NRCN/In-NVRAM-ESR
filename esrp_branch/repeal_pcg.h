#ifndef REPEAL_PCG
#define REPEAL_PCG

#include <mpi.h>
#include <mpi_one_sided_extension/mpi_win_pmem.h>
#include <mpi_one_sided_extension/defines.h>
#include <omp.h>


struct PCG_info *create_pcg_info();
void free_pcg_info(struct PCG_info *pcg_info);

gsl_vector *solve_linear_system(
	MPI_Comm comm, int comm_rank, int comm_size,
	const struct repeal_matrix *A, const struct repeal_matrix *P,
    const gsl_vector* b, gsl_vector* x_0,
	const struct solver_options *solver_opts, const struct ESR_options *esr_opts, const struct NVESR_options *nvesr_opts, struct SPMV_context *spmv_ctx,
	struct PCG_info *pcg_info
);


#endif
