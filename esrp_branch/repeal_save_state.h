#ifndef REPEAL_SAVE_STATE
#define REPEAL_SAVE_STATE

#include <libpmemobj.h>
#include <mpi.h>
#include <mpi_one_sided_extension/mpi_win_pmem.h>
#include <mpi_one_sided_extension/defines.h>

//PCG_state_copy
struct PCG_state_copy *create_pcg_state_copy(PMEMobjpool* pool, MPI_Win win, const struct ESR_options *esr_opts, size_t local_size, size_t global_size);
void pcg_state_copy_set_zero(struct PCG_state_copy *pcg_state_copy);
void free_pcg_state_copy(struct PCG_state_copy *pcg_state_copy);

//checkpointing-specific stuff
void pcg_init_checkpointing_communication(int comm_rank, int comm_size, struct SPMV_context *spmv_ctx, struct ESR_options *esr_opts);

//handle to solver
struct PCG_solver_handle *create_pcg_solver_handle(gsl_vector *x, gsl_vector *r, gsl_vector *z, gsl_vector *p, gsl_vector *Ap, double *alpha, double *beta, double *pAp, double *rz, double *rz_prev, size_t *iteration);
void pcg_solver_handle_set_zero(struct PCG_solver_handle *pcg_handle);
void free_pcg_solver_handle(struct PCG_solver_handle *pcg_handle);

//the actual saving
void pcg_save_current_state(PMEMobjpool* pool, MPI_Win_pmem win, bool win_ram_on, bool local_windows, bool one_window, char* checkpoint_path, MPI_Comm comm, struct PCG_solver_handle *pcg_handle, struct SPMV_context *spmv_ctx, struct PCG_state_copy *pcg_state_copy, const struct ESR_options *esr_opts);
void pcg_solver_reset_to_saved_state(struct PCG_solver_handle *pcg_handle, struct PCG_state_copy *pcg_state_copy);





struct Pipelined_state_copy *create_pipelined_state_copy(const struct ESR_options *esr_opts, size_t local_size, size_t global_size);
void pipelined_state_copy_set_zero(struct Pipelined_state_copy *pipelined_state_copy);
void free_pipelined_state_copy(struct Pipelined_state_copy *pipelined_state_copy);

struct Pipelined_solver_handle *create_pipelined_solver_handle(gsl_vector *m, gsl_vector *n, gsl_vector *r, gsl_vector *r_prev, gsl_vector *u, gsl_vector *u_prev, gsl_vector *w, gsl_vector *w_prev, gsl_vector *x, gsl_vector *x_prev, gsl_vector *z, gsl_vector *q, gsl_vector *s, gsl_vector *p, double *alpha, double *gamma, double *gamma_prev, double *delta, double *rr, size_t *iteration);
void pipelined_solver_handle_set_zero(struct Pipelined_solver_handle *pipelined_solver_handle);
void free_pipelined_solver_handle(struct Pipelined_solver_handle *pipelined_solver_handle);

void pipelined_save_current_state(MPI_Comm comm, struct Pipelined_solver_handle *pipelined_handle, struct SPMV_context *spmv_ctx, struct Pipelined_state_copy *pipelined_state_copy, const struct ESR_options *esr_opts);
void pipelined_solver_reset_to_saved_state(struct Pipelined_solver_handle *pipelined_handle, struct Pipelined_state_copy *pipelined_state_copy);



#endif
