#ifndef REPEAL_PC
#define REPEAL_PC

#include "repeal_utils.h"

struct repeal_matrix* get_preconditioner(MPI_Comm comm, int comm_rank, int comm_size, const gsl_spmatrix *A, const struct PC_options *pc_opts, const enum COMM_strategy strategy, const struct SPMV_context *spmv_ctx);


#endif
