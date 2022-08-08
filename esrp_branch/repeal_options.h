#ifndef REPEAL_OPTIONS
#define REPEAL_OPTIONS

#include "repeal_utils.h"

// *****everything related to handling the different possible settings*****

struct PC_options *get_pc_options();
struct PC_options *get_pc_options_from_values(enum PC_type pc_type, int blocksize);
void free_pc_options(struct PC_options *pc_opts);

struct solver_options *get_solver_options();
struct solver_options *get_solver_options_from_values(enum Solver solvertype, double atol, double rtol, size_t max_iter, int verbose, int with_residual_replacement);
void free_solver_options(struct solver_options *solver_opts);

struct ESR_options *get_esr_options();
void free_esr_options(struct ESR_options *esr_opts);

struct Input_options *get_input_options();
void free_input_options(struct Input_options *input_opts);

void set_from_options(int argc, char *argv[], int comm_rank, int comm_size,
                      enum COMM_strategy *comm_strategy,
                      struct PC_options *pc_opts,
                      struct solver_options *solver_opts,
                      struct ESR_options *esr_opts,
                      struct NVESR_options *nvesr_opts,
                      struct Input_options *input_opts);

struct NVESR_options *get_nvesr_options();
void free_nvesr_options(struct NVESR_options *nvesr_options);

#endif
