#ifndef REPEAL_UTILS
#define REPEAL_UTILS

#include <mpi.h>
#include <stdio.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_vector.h>
#include <libpmemobj.h>

#include "pmem_vector.h"

//Macros

#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(x, y) (((x) > (y)) ? (x) : (y))

#define failure(message) do{perror(message); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);} while(0)


//Enums

enum COMM_strategy{COMM_allgatherv, COMM_alltoallv, COMM_minimal, COMM_minimal_split}; //communication stratey for SPMV

enum PC_type{PC_none, PC_Jacobi, PC_BJ}; //preconditioner type

enum Options {
  OPT_file,
  OPT_generate,
  OPT_bfile,
  OPT_rhs,
  OPT_pc,
  OPT_comm,
  OPT_atol,
  OPT_rtol,
  OPT_innertol,
  OPT_maxit,
  OPT_verbose,
  OPT_rec_verbose,
  OPT_copies,
  OPT_breakiter,
  OPT_breaknodes,
  OPT_blocksize,
  OPT_residual_replacement,
  OPT_solver_mode,
  OPT_period,
  OPT_esr_strategy,
  OPT_pmem_pool_path
};  // to make handling of command line input more legible

enum Solver{SOLVER_pcg, SOLVER_pipelined}; //to make extension to additional solver modes possible

enum Reconstruction_strategy {
  REC_inplace,
  REC_on_survivors,
  REC_checkpoint_in_memory,
  REC_checkpoint_on_disk,
  REC_checkpoint_on_nvram_homogenous,
  REC_nvesr_homogenous,
  REC_checkpoint_on_nvram_RDMA,
  REC_nvesr_RDMA
};
enum Redistribution_strategy{REDIST_none, REDIST_rowcol, REDIST_keeporder};

enum Local_data_fate{FATE_keep, FATE_erase};
enum Reconstruction_role{ROLE_reconstructing, ROLE_not_reconstructing};

enum Checkpoint_tags{CHECKPOINT_scalars, CHECKPOINT_x, CHECKPOINT_r, CHECKPOINT_p, CHECKPOINT_z}; //tags for MPI messages sent during checkpointing reconstruction

enum Input_matrix_source{INPUT_from_mtx_file, INPUT_from_bin_file, INPUT_generate}; //flag to define where the problem matrix is coming from
enum Input_matrix_generation_strategy{GEN_poisson_7, GEN_poisson_27}; //define which input problem should be generated



// Structures holding reconstructed data

struct Recovered_static_data
{
	gsl_vector * b;
	gsl_spmatrix * csr_A;
	gsl_spmatrix * csc_A;
};


struct Recovered_dynamic_data
{
	gsl_vector * x;
	gsl_vector * r;
	gsl_vector * z;
	gsl_vector * p;
};



//Structs holding the necessary data for SPMV for a matrix

//struct for alltoallv
struct SPMV_comm_structure{
    int *recvcounts;
    int *recvdispl;
    int *sendcounts;
    int *senddispl;
};

//for alltoallw with MPI_Type_indexed
struct SPMV_comm_structure_minimal{
	int *sendcounts;
	int *recvcounts;
	int *senddispl;
	int *recvdispl;
	MPI_Datatype *sendtypes;
	MPI_Datatype *recvtypes;
	MPI_Datatype *datatypes; //not needed for communication, this is just a handle to make it possible to deallocate the custom datatypes in the end
	int num_datatypes;
};

//wrapper
/*
struct SPMV_info{
	struct SPMV_comm_structure *alltoallv_info;
	struct SPMV_comm_structure_minimal *minimal_info;
};
*/


//Wrapper around all the information relating to a single matrix
struct repeal_matrix
{
		gsl_spmatrix *M;
		gsl_spmatrix *M_intern;
		gsl_spmatrix *M_extern;
		struct SPMV_comm_structure *alltoallv_info;
		struct SPMV_comm_structure_minimal *minimal_info;
		struct SPMV_comm_structure_minimal *minimal_info_with_resilience;
		int external_communication_needed;
		size_t size1;
		size_t size2;
};


//Working context for SPMV

struct SPMV_context
{
	enum COMM_strategy strategy;
	gsl_vector *buffer;
	gsl_vector *sendbuf;
	int *elements_per_process;
	int *displacements;
};



//Structs storing the user-defined options

//preconditioner options
struct PC_options
{
	enum PC_type pc_type;
	int pc_bj_blocksize;
};

//solver options
struct solver_options
{
	enum Solver solvertype;
 	double atol;
	double rtol;
	size_t max_iter;
	int verbose;
	int output_rank; //rank of the node that should print any information to standard output
	int with_residual_replacement;
};

//resilience and reconstruction
struct ESR_options
{
	int redundant_copies;
	int period; //not saving the state in every single iteration
	enum Reconstruction_strategy reconstruction_strategy; //with / without spares
	enum Redistribution_strategy redistribution_strategy; //what to do during the redistribution phase

	int verbose_reconstruction; //flag to control whether the time measurements during the reconstruction phase will be printed
	int reconstruction_output_rank; //the rank that should print to stdout during reconstruction (will usually be the same as the rank that prints to stdout anyways)

	int broken_iteration;
	int num_broken_nodes;
	int *broken_node_ranks;
	int *buddies_that_save_me; //ranks of the nodes that I send my redundant data to
	int *buddies_that_I_save; //ranks of the nodes that send their redundant data to me
	double innertol;

	char * binary_matrix_filename; // It seems like the options module stores this data, so we just point at it
	char * binary_rhs_filename;

	int *checkpoint_sendcounts;
	int *checkpoint_senddispls;
	int *checkpoint_recvcounts;
	int *checkpoint_recvdispls;
};

// Options for NVESR.
struct NVESR_options {
  // The path to the persistent memory object pool.
  char *pmem_pool_path;
};

struct ESR_setup
{
	enum Local_data_fate local_data_fate; // Whether the node erases its data to simulate a failure
	enum Reconstruction_role rec_role; // Whether the node will perform reconstruction
	int node_continues; // If the node will be present during the reconstruction and continue afterwards
	int *rec_node_ranks;
	struct Recovered_static_data  rec_static_data;
	struct Recovered_dynamic_data rec_dynamic_data;
	MPI_Comm continue_comm; // Comm for nodes that continue after the node failure
	MPI_Comm reconstruction_comm; // Comm for reconstructing nodes
};

//input problem
struct Input_options
{
	enum Input_matrix_source input_matrix_source;

	//reading from file
	char *matrix_file_name; //.mtx-file or binary file storing the matrix

	//generating a matrix
	enum Input_matrix_generation_strategy generation_strategy; //which stencil to use
	int grid_dimension; //the discretization grid will have the x/y/z dimensions grid_dimension * grid_dimension * <number of processors>
	int matrix_factor; //factor to multiply every matrix element with
};



//some metadata needed during reconstruction phase
struct reconstruction_data_distribution
{
	size_t failed_size;
	int *elements_per_broken_process;
	int *displacements_per_broken_process;
	int *elements_per_surviving_process;
	int *displacements_per_surviving_process;
	int *allgatherv_recvcounts;
	int *allgatherv_recvdispls;
};

//storage of data for resilience on persistent memory.

POBJ_LAYOUT_BEGIN(pmem_store);
POBJ_LAYOUT_ROOT(pmem_store, struct PCG_persistent_state_copy);
POBJ_LAYOUT_END(pmem_store);

struct PCG_persistent_state_copy {
	// for NVESR Homogenous
	TOID(struct pmem_vector) nvesr_buffer_copy_1;
	TOID(struct pmem_vector) nvesr_buffer_copy_2;

  // for homogenous checkpointing on NVRAM
	TOID(struct pmem_vector) checkpoint_x;
	TOID(struct pmem_vector) checkpoint_r;
  TOID(struct pmem_vector) checkpoint_z;
	TOID(struct pmem_vector) checkpoint_p;
  double beta;
	double rz;
};

//storage of data for resilience

struct PCG_state_copy
{
	//for ESR
	gsl_vector *buffer_copy_1;
	gsl_vector *buffer_copy_2;

	//for periodic ESR
	gsl_vector *buffer_intermediate_copy;

	//for periodic ESR and checkpointing
	gsl_vector *x_local;
	gsl_vector *r_local;
	gsl_vector *z_local;
	gsl_vector *p_local;
	double beta;
	double rz;
	size_t iteration;

	//for checkpointing
	gsl_vector *checkpoint_x;
	gsl_vector *checkpoint_r;
	gsl_vector *checkpoint_z;
	gsl_vector *checkpoint_p;

  // Persistent copy on NVRAM.
  TOID(struct PCG_persistent_state_copy) persistent_copy;
};

struct Pipelined_state_copy
{
	//for ESR
	gsl_vector *buffer_copy_1;
	gsl_vector *buffer_copy_2;

	//for periodic ESR
	gsl_vector *buffer_intermediate_copy;

	gsl_vector *m;
	gsl_vector *n;
	gsl_vector *r;
	gsl_vector *r_prev;
	gsl_vector *u;
	gsl_vector *u_prev;
	gsl_vector *w;
	gsl_vector *w_prev;
	gsl_vector *x;
	gsl_vector *x_prev;
	gsl_vector *z;
	gsl_vector *q;
	gsl_vector *s;
	gsl_vector *p;
	double alpha;
	double gamma;
	double gamma_prev;
	double delta;
	double rr;
};



//this is a utility struct so we can modify the solver state during the reconstruction without needing a function parameter for each of these vectors and scalars
//if anything else needs to be modified, just add a pointer to it inside this struct
struct PCG_solver_handle
{
	//pointers to vectors / scalars in the solver
	gsl_vector *x;
	gsl_vector *r;
	gsl_vector *z;
	gsl_vector *p;
	gsl_vector *Ap;
	double *alpha;
	double *beta;
	double *pAp;
	double *rz;
	double *rz_prev;
	size_t *iteration;
};


struct Pipelined_solver_handle
{
	gsl_vector *m;
	gsl_vector *n;
	gsl_vector *r;
	gsl_vector *r_prev;
	gsl_vector *u;
	gsl_vector *u_prev;
	gsl_vector *w;
	gsl_vector *w_prev;
	gsl_vector *x;
	gsl_vector *x_prev;
	gsl_vector *z;
	gsl_vector *q;
	gsl_vector *s;
	gsl_vector *p;
	double *alpha;
	double *gamma;
	double *gamma_prev;
	double *delta;
	double *rr;

	size_t *iteration;
};


//Struct for solver output

//info about what happened in the solver
struct PCG_info
{
	double abs_residual_norm;
	int iterations;
	int converged_iterations;
	int converged_atol;
	int converged_rtol;
	double reconstruction_time;
};

#endif
