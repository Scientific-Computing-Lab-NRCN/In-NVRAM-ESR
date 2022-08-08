#ifndef REPEAL_VEC
#define REPEAL_VEC

double dot(MPI_Comm comm, const gsl_vector* v, const gsl_vector* w);
double norm(MPI_Comm comm, const gsl_vector* v);
gsl_vector *random_vector(size_t size, double min_elem, double max_elem);

void matrix_min_max(MPI_Comm comm, const gsl_spmatrix* matrix, double* min_elem, double* max_elem);
gsl_vector *create_random_vector_from_matrix(MPI_Comm comm, gsl_spmatrix *M);

double compute_residual_norm(MPI_Comm comm, struct repeal_matrix *mat, const gsl_vector* x, const gsl_vector* b, struct SPMV_context *spmv_ctx);

struct repeal_matrix *repeal_matrix_create(MPI_Comm comm, int comm_rank, int comm_size, gsl_spmatrix *M, const enum COMM_strategy strategy, int external_communication_needed, const struct SPMV_context *spmv_ctx, const struct ESR_options *esr_opts);
void repeal_matrix_free(struct repeal_matrix *mat);

#endif
