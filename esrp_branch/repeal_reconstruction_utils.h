#ifndef REPEAL_ESR_UTILS
#define REPEAL_ESR_UTILS

#include <libpmemobj.h>

#include "pmem_vector.h"
#include "repeal_utils.h"

//void init_esr_setup(struct ESR_setup *esr_setup);
struct ESR_setup *create_esr_setup();
void free_esr_setup(struct ESR_setup *esr_setup);


void configure_reconstruction(MPI_Comm *world_comm, const struct ESR_options * esr_options, struct ESR_setup *esr_setup);

void retrieve_static_data(MPI_Comm *world_comm, const struct ESR_options * esr_opts, struct ESR_setup * esr_setup);


//to find out whether a node is a replacement node
int is_broken(int rank, const struct ESR_options *esr_opts);
void get_my_deceased_status(int comm_rank, int *i_am_broken, int *my_broken_index, const struct ESR_options *esr_opts);

struct reconstruction_data_distribution *get_data_distribution_after_failure(int comm_size, const struct ESR_options *esr_opts, const struct SPMV_context *spmv_ctx);
void free_data_distribution_after_failure(struct reconstruction_data_distribution *distribution);


//communicators
void setup_intercomm(MPI_Comm comm, int comm_size, const int i_am_broken, const struct ESR_options *esr_opts, MPI_Comm *inter_comm_ptr);
void setup_reconstruction_comm(MPI_Comm comm, const int i_am_broken, const struct ESR_options *esr_opts, MPI_Comm *reconstruction_comm_ptr);


//gather required data at replacement nodes

void retrieve_scalars(MPI_Comm inter_comm, const int i_am_broken, int num_scalars, double *scalars);

void retrieve_redundant_vector_copies(MPI_Comm comm, int comm_rank, int comm_size, const int i_am_broken, const int my_broken_index, const int local_size,
	const struct ESR_options *esr_opts, const struct SPMV_comm_structure_minimal *comm_info,
	gsl_vector *vec_local, gsl_vector *vec_local_prev, gsl_vector *copy_1, gsl_vector *copy_2,
	const int *displacements);

void retrieve_redundant_vector_copies_from_nvram_homogenous(
    PMEMobjpool *pool, const int i_am_broken, const int my_broken_index,
    gsl_vector *vec_local, gsl_vector *vec_local_prev,
    const TOID(struct pmem_vector) * copy_1, const TOID(struct pmem_vector) * copy_2,
    const struct SPMV_context *spmv_ctx);

void retrieve_scalars(MPI_Comm inter_comm, const int i_am_broken, int num_scalars, double *scalars);

void gather_global_vector_begin(MPI_Comm inter_comm, int comm_rank, const int i_am_broken,
	gsl_vector *vec_local, gsl_vector *vec_global,
	MPI_Request *request, const struct SPMV_context *spmv_ctx, const struct reconstruction_data_distribution *distribution
);

void gather_global_vectors_end(int num_requests, MPI_Request *requests);


//reconstruct lost data
void create_inner_system(
	MPI_Comm broken_comm, int my_broken_index, size_t local_size, size_t failed_size,
	const struct repeal_matrix *matrix,
	const struct ESR_options *esr_opts, const struct SPMV_context *spmv_ctx, const struct reconstruction_data_distribution *distribution, //structs describing the "current state"
	const struct ESR_options *esr_opts_sub, const struct SPMV_context *spmv_ctx_sub, const enum PC_type pc_type_sub, //structs describing the new subsystem that should be created
	struct repeal_matrix **matrix_sub, struct repeal_matrix **matrix_sub_prec //output: new system matrix + preconditioner
);

void solve_inner_system(
    MPI_Comm broken_comm, const int my_broken_index, const int num_broken_nodes,
    const struct repeal_matrix *M_sub, const struct repeal_matrix *P_sub,
    const gsl_vector *rhs, gsl_vector *vec,
    struct solver_options *solver_opts_sub,
    struct ESR_options
        *esr_opts_sub,  // create these outside because we use the same settings
                        // every time we solve an innner system
    struct NVESR_options *nvesr_opts_sub, struct SPMV_context *spmv_ctx_sub);

#endif
