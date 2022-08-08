#ifndef REPEAL_SPMV
#define REPEAL_SPMV

#include "repeal_utils.h"

struct SPMV_context *get_spmv_context(MPI_Comm comm, int comm_size, enum COMM_strategy strategy, size_t local_size, size_t global_size);
struct SPMV_context *get_spmv_context_from_existing_distribution(int comm_size, enum COMM_strategy strategy, size_t local_size, size_t global_size, int *elements_per_process, int *displacements);
void free_spmv_context(struct SPMV_context *spmv_ctx);

//wrapper function to choose the appropriate SPMV implementation
void spmv(MPI_Comm comm, const struct repeal_matrix *mat, const gsl_vector* v, gsl_vector* Mv, struct SPMV_context *spmv_ctx, int send_redundant_elements);


#endif
