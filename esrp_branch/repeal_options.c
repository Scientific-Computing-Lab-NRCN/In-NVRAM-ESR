#include "repeal_options.h"

#include <getopt.h>
#include <gsl/gsl_sort.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "repeal_commstrategy.h"

//"default constructor"
struct PC_options *get_pc_options() {
  struct PC_options *pc_opts = malloc(sizeof(struct PC_options));
  if (!pc_opts) failure("PC_options allocation failed.");

  pc_opts->pc_type = PC_none;
  pc_opts->pc_bj_blocksize = -1;
  return pc_opts;
}

struct PC_options *get_pc_options_from_values(enum PC_type pc_type,
                                              int blocksize) {
  struct PC_options *pc_opts = malloc(sizeof(struct PC_options));
  if (!pc_opts) failure("PC_options allocation failed.");

  pc_opts->pc_type = pc_type;
  pc_opts->pc_bj_blocksize = blocksize;
  // TODO check that blocksize value is valid

  return pc_opts;
}

void free_pc_options(struct PC_options *pc_opts) { free(pc_opts); }

struct solver_options *get_solver_options() {
  struct solver_options *solver_opts = malloc(sizeof(struct solver_options));
  if (!solver_opts) failure("Solver options allocation failed.");

  solver_opts->solvertype = SOLVER_pcg;  // default behaviour: ordinary PCG
  solver_opts->atol = 1e-8;
  solver_opts->rtol = 1e-8;
  solver_opts->max_iter = 10000;
  solver_opts->verbose = 0;
  solver_opts->output_rank = 0;  // default: just use rank 0 to print
  solver_opts->with_residual_replacement =
      0;  // default behaviour: no residual replacement
  return solver_opts;
}

struct solver_options *get_solver_options_from_values(
    enum Solver solvertype, double atol, double rtol, size_t max_iter,
    int verbose, int with_residual_replacement) {
  struct solver_options *solver_opts = malloc(sizeof(struct solver_options));
  if (!solver_opts) failure("Solver options allocation failed.");

  solver_opts->solvertype = solvertype;
  solver_opts->atol = atol;
  solver_opts->rtol = rtol;
  solver_opts->max_iter = max_iter;
  solver_opts->verbose = verbose;
  solver_opts->with_residual_replacement = with_residual_replacement;
  return solver_opts;
}

void free_solver_options(struct solver_options *solver_opts) {
  free(solver_opts);
}

struct ESR_options *get_esr_options() {
  struct ESR_options *esr_opts = malloc(sizeof(struct ESR_options));
  if (!esr_opts) failure("ESR options allocation failed.");

  esr_opts->redundant_copies = 0;
  esr_opts->period = 0;  // default: saving the state every iteration
  esr_opts->reconstruction_strategy =
      REC_inplace;  // default: replacement nodes are available
  esr_opts->redistribution_strategy =
      REDIST_none;  // default: no redistribution necessary
  esr_opts->broken_iteration = -1;
  esr_opts->num_broken_nodes = 0;
  esr_opts->broken_node_ranks = NULL;
  esr_opts->buddies_that_save_me = NULL;
  esr_opts->buddies_that_I_save = NULL;
  esr_opts->innertol = 1e-11;

  esr_opts->verbose_reconstruction =
      0;  // default: don't print time measurements
  esr_opts->reconstruction_output_rank = 0;  // default: use rank 0 to print

  esr_opts->binary_matrix_filename = NULL;
  esr_opts->binary_rhs_filename = NULL;

  esr_opts->checkpoint_sendcounts = NULL;
  esr_opts->checkpoint_senddispls = NULL;
  esr_opts->checkpoint_recvcounts = NULL;
  esr_opts->checkpoint_recvdispls = NULL;

  return esr_opts;
}

void free_esr_options(struct ESR_options *esr_opts) {
  if (esr_opts->broken_node_ranks) free(esr_opts->broken_node_ranks);
  if (esr_opts->buddies_that_save_me) free(esr_opts->buddies_that_save_me);

  if (esr_opts->checkpoint_sendcounts) {
    free(esr_opts->checkpoint_sendcounts);
    free(esr_opts->checkpoint_senddispls);
    free(esr_opts->checkpoint_recvcounts);
    free(esr_opts->checkpoint_recvdispls);
    free(esr_opts->buddies_that_I_save);
  }

  free(esr_opts);
}

struct Input_options *get_input_options() {
  struct Input_options *input_opts = malloc(sizeof(struct Input_options));
  if (!input_opts) failure("Input options allocation failed");

  input_opts->matrix_file_name = NULL;
  input_opts->matrix_factor = 1;

  // no further default values, instead a failure should be thrown if those
  // options are not explicitly set

  return input_opts;
}

void free_input_options(struct Input_options *input_opts) { free(input_opts); }

struct NVESR_options *get_nvesr_options() {
  struct NVESR_options *nvesr_opts = calloc(1, sizeof(struct NVESR_options));
  if (!nvesr_opts) failure("NVESR options allocation failed");
  nvesr_opts->pmem_pool_path = NULL;
  return nvesr_opts;
}

void free_nvesr_options(struct NVESR_options *nvesr_opts) { free(nvesr_opts); }

void set_from_options(int argc, char *argv[], int comm_rank, int comm_size,
                      enum COMM_strategy *comm_strategy,
                      struct PC_options *pc_opts,
                      struct solver_options *solver_opts,
                      struct ESR_options *esr_opts,
                      struct NVESR_options *nvesr_opts,
                      struct Input_options *input_opts) {
  // TODO: add flag for factor
  // define valid options
  static struct option long_options[] = {
      {"f", required_argument, 0, OPT_file},  // matrix file name
      {"generate", required_argument, 0,
       OPT_generate},  // generate an input matrix; argument has the form
                       // <PDE><stencil size>-<grid size>; for example:
                       // poisson7-100 discretizes a poisson equation with a
                       // 7-point stencil on a grid with x/y/z dimensions
                       // 100/100/<number of MPI processes>; currently available
                       // PDE-stencil combinations are poisson7 and poisson27
      {"fb", required_argument, 0, OPT_bfile},  // binary matrix file name
      {"rhs", required_argument, 0, OPT_rhs},   // binary rhs file name
      {"pc", required_argument, 0, OPT_pc},     // preconditioner type
      {"comm", required_argument, 0,
       OPT_comm},  // communication strategy for SPMV
      {"atol", required_argument, 0, OPT_atol},  // absolute tolerance
      {"rtol", required_argument, 0, OPT_rtol},  // relative tolerance
      {"innertol", required_argument, 0,
       OPT_innertol},  // relative tolerance for the inner system during
                       // reconstruction
      {"maxit", required_argument, 0,
       OPT_maxit},                               // maximum number of iterations
      {"verbose", no_argument, 0, OPT_verbose},  // verbosity
      {"verbosereconstruction", no_argument, 0, OPT_rec_verbose},  // verbosity
      {"copies", required_argument, 0, OPT_copies},  // redundant copies for ESR
      {"breakiter", required_argument, 0,
       OPT_breakiter},  // iteration where the node failures happen
      {"breaknodes", required_argument, 0,
       OPT_breaknodes},  // the nodes that should break (comma separated)
      {"blocksize", required_argument, 0,
       OPT_blocksize},  // maximum blocksize for block jacobi preconditioner
                        // (note: the actual block size may be smaller than this
                        // value; only makes sense with block jacobi
                        // preconditioner, if this option is used with any other
                        // preconditioner, it is simply ignored)
      {"with_residual_replacement", required_argument, 0,
       OPT_residual_replacement},  // use residual replacement (only meaningful
                                   // with the pipelined solver)
      {"solver", required_argument, 0,
       OPT_solver_mode},  // which PCG version to use ("pcg" or "pipelined")
      {"esr", required_argument, 0,
       OPT_esr_strategy},  //"inplace" / "rowcol" / "keeporder": don't replace
                           // failed nodes, but shrink the communicator instead;
                           //"checkpoint": use in-memory checkpointing instead
                           // of reconstruction / "checkpoint": in-memory
                           // checkpointing / "checkpoint-disk"
      {"period", required_argument, 0,
       OPT_period},  // periodic ESR / checkpointing: don't save the state in
                     // every single iteration
      {"pmem_path", required_argument, 0, OPT_pmem_pool_path},
      {0, 0, 0, 0}};

  // set default communication strategy
  *comm_strategy = COMM_minimal;

  // read command line options and replace default options with the user input
  int opt;
  int long_index = 0;
  int matrix_filename_provided = 0, matrix_generator_provided = 0,
      esr_iter_provided = 0, esr_nodes_provided = 0;  // flags for sanity checks
  int bmatrix_filename_provided = 0, brhs_filename_provided = 0;
  int pmem_pool_path_provided = false;
  while ((opt = getopt_long_only(argc, argv, "", long_options, &long_index)) !=
         -1) {
    switch (opt) {
      case OPT_file:  // matrix file name
        input_opts->input_matrix_source = INPUT_from_mtx_file;
        input_opts->matrix_file_name = optarg;
        matrix_filename_provided = 1;
        break;
      case OPT_generate: {
        // a matrix should be generated
        input_opts->input_matrix_source = INPUT_generate;

        // create a copy of optarg, just to make sure this does not mess up
        // optarg
        int len = strlen(optarg);
        char *optargcopy = (char *)malloc(len * sizeof(char));
        strcpy(optargcopy, optarg);

        // extract the part before the hyphen (i.e. description of PDE and
        // stencil)
        char *token = strtok(optargcopy, "-");
        if (strcmp(token, "poisson7") == 0)
          input_opts->generation_strategy = GEN_poisson_7;
        else if (strcmp(token, "poisson27") == 0)
          input_opts->generation_strategy = GEN_poisson_27;
        else
          failure("Invalid matrix generation strategy");

        // extract the part after the comma (= the grid dimension in x and y
        // direction)
        token = strtok(NULL, "-");
        int dim = atoi(token);
        if (dim <= 0)
          failure("Invalid grid dimension");  // sanity check: grid dimensions
                                              // have to be positive
        input_opts->grid_dimension = dim;

        // set flag to show a matrix generation strategy has been selected
        matrix_generator_provided = 1;

        break;
      }
      case OPT_bfile:
        input_opts->input_matrix_source = INPUT_from_bin_file;
        input_opts->matrix_file_name = optarg;
        bmatrix_filename_provided = 1;
        break;
      case OPT_rhs:
        esr_opts->binary_rhs_filename = optarg;
        brhs_filename_provided = 1;
        break;
      case OPT_pc:  // preconditioner type
        if (strcmp(optarg, "none") == 0)
          pc_opts->pc_type = PC_none;
        else if (strcmp(optarg, "blockjacobi") == 0)
          pc_opts->pc_type = PC_BJ;
        else if (strcmp(optarg, "jacobi") == 0)
          pc_opts->pc_type = PC_Jacobi;
        else
          failure("Invalid preconditioner type.");
        break;
      case OPT_blocksize:
        pc_opts->pc_bj_blocksize = atoi(optarg);
        if ((pc_opts->pc_bj_blocksize <= 0) && (pc_opts->pc_bj_blocksize != -1))
          failure("Invalid block size for block jacobi preconditioner");
        break;
      case OPT_comm:  // communication strategy
        if (strcmp(optarg, "allgatherv") == 0)
          *comm_strategy = COMM_allgatherv;
        else if (strcmp(optarg, "alltoallv") == 0)
          *comm_strategy = COMM_alltoallv;
        else if (strcmp(optarg, "minimal") == 0)
          *comm_strategy = COMM_minimal;
        else if (strcmp(optarg, "split") == 0)
          *comm_strategy = COMM_minimal_split;
        else
          failure("Invalid communication strategy.");
        break;
      case OPT_solver_mode:  // ordinary or pipelined pcg
        if (strcmp(optarg, "pcg") == 0)
          solver_opts->solvertype = SOLVER_pcg;
        else if (strcmp(optarg, "pipelined") == 0)
          solver_opts->solvertype = SOLVER_pipelined;
        else
          failure("Invalid solver");
        break;
      case OPT_atol:
        solver_opts->atol = atof(optarg);
        break;
      case OPT_rtol:
        solver_opts->rtol = atof(optarg);
        break;
      case OPT_innertol:
        esr_opts->innertol = atof(optarg);
        break;
      case OPT_maxit:
        solver_opts->max_iter = atoi(optarg);
        break;
      case OPT_verbose:
        solver_opts->verbose = 1;
        break;
      case OPT_rec_verbose:
        esr_opts->verbose_reconstruction = 1;
        break;
      case OPT_residual_replacement:
        solver_opts->with_residual_replacement = atoi(optarg);
        if (solver_opts->with_residual_replacement < 0)
          failure("Argument for residual replacement cannot be less than zero");
        break;
      case OPT_copies:
        esr_opts->redundant_copies = atoi(optarg);
        if (esr_opts->redundant_copies < 0)
          failure("Invalid number of redundant copies");
        // sanity check: in order to be able to store k remote copies of each
        // element, there have to be at least k+1 processes
        if (esr_opts->redundant_copies >= comm_size)
          failure(
              "Cannot achieve required resilience with given number of "
              "processes");
        break;
      case OPT_breakiter:
        esr_opts->broken_iteration = atoi(optarg);
        esr_iter_provided = 1;
        break;
      case OPT_breaknodes: {
        // count the number of broken nodes in the input
        int i = 0;
        int num_nodes = 1;
        for (; optarg[i]; i++)
          if (optarg[i] == ',') num_nodes++;
        if (num_nodes > comm_size)
          failure(
              "Cannot simulate required number of node failures with the given "
              "number of processors");
        esr_opts->num_broken_nodes = num_nodes;

        // allocate required memory
        esr_opts->broken_node_ranks = (int *)malloc(num_nodes * sizeof(int));

        if (!(esr_opts->broken_node_ranks))
          failure("Memory allocation failed.");

        // save the ranks of the broken nodes
        int len = strlen(optarg);
        char *optargcopy = (char *)malloc(len * sizeof(char));
        strcpy(optargcopy,
               optarg);  // just to make sure this does not mess up optarg
        char *token = strtok(optargcopy, ",");
        i = 0;
        while (token) {
          esr_opts->broken_node_ranks[i] = atoi(token);  // process current
                                                         // token
          if (esr_opts->broken_node_ranks[i] >= comm_size ||
              esr_opts->broken_node_ranks[i] < 0)
            failure("Invalid node rank");
          i++;
          token = strtok(NULL, ",");  // get next token
        }
        free(optargcopy);
        esr_nodes_provided = 1;
        break;
      }
      case OPT_period:
        esr_opts->period = atoi(optarg);
        if (esr_opts->period < 0)
          failure("Invalid value: period must not be less than 0");
        else if (esr_opts->period <= 2)
          esr_opts->period =
              0;  // we need the values from two subsequent iterations, so
                  // anything less than 3 is equivalent to saving the state in
                  // every iteration
        break;
      case OPT_esr_strategy:
        if (strcmp(optarg, "inplace") == 0) {
          esr_opts->reconstruction_strategy = REC_inplace;
          esr_opts->redistribution_strategy = REDIST_none;
        } else if (strcmp(optarg, "restart") == 0) {
          failure("not yet implemented");
        } else if (strcmp(optarg, "rowcol") == 0) {
          esr_opts->reconstruction_strategy = REC_on_survivors;
          esr_opts->redistribution_strategy = REDIST_rowcol;
        } else if (strcmp(optarg, "keeporder") == 0) {
          esr_opts->reconstruction_strategy = REC_on_survivors;
          esr_opts->redistribution_strategy = REDIST_keeporder;
        } else if (strcmp(optarg, "checkpoint-in-memory") == 0) {
          esr_opts->reconstruction_strategy = REC_checkpoint_in_memory;
          esr_opts->redistribution_strategy = REDIST_none;
        } else if (strcmp(optarg, "checkpoint-disk") == 0) {
          esr_opts->reconstruction_strategy = REC_checkpoint_on_disk;
          esr_opts->redistribution_strategy = REDIST_none;
        } else if (strcmp(optarg, "checkpoint-nvram-homogenous") == 0) {
          esr_opts->redundant_copies = 0;
          esr_opts->reconstruction_strategy = REC_checkpoint_on_nvram_homogenous;
          esr_opts->redistribution_strategy = REDIST_none;
        } else if (strcmp(optarg, "checkpoint-nvram-RDMA") == 0) {
          esr_opts->redundant_copies = 0;
          esr_opts->reconstruction_strategy = REC_checkpoint_on_nvram_RDMA;
          esr_opts->redistribution_strategy = REDIST_none;
        } else if (strcmp(optarg, "nvesr-homogenous") == 0) {
          esr_opts->redundant_copies = 0;
          esr_opts->reconstruction_strategy = REC_nvesr_homogenous;
          esr_opts->redistribution_strategy = REDIST_none;
        } else if (strcmp(optarg, "nvesr-RDMA") == 0) {
          esr_opts->redundant_copies = 0;
          esr_opts->reconstruction_strategy = REC_nvesr_RDMA;
          esr_opts->redistribution_strategy = REDIST_none;
        
        } else
          failure("invalid argument for esr");
        break;
      case OPT_pmem_pool_path:
        nvesr_opts->pmem_pool_path = optarg;
        pmem_pool_path_provided = true;
    }
  }

  // check that a matrix source was entered
  if (!matrix_filename_provided && !matrix_generator_provided &&
      !bmatrix_filename_provided)
    failure(
        "Please provide a matrix source (file name or generation strategy)");

  // check that only one matrix source was entered
  if (matrix_filename_provided + matrix_generator_provided +
          bmatrix_filename_provided >
      1)
    failure(
        "Multiple matrix sources detected. Please provide either a file name, "
        "or a binary filename, or a generation strategy (but not more than one "
        "of them).");

  // check that ESR information is complete and makes sense
  if ((esr_iter_provided && !esr_nodes_provided) ||
      (!esr_iter_provided && esr_nodes_provided))  // XOR
    failure(
        "Cannot simulate node failures without iteration number or node ranks");
  //if (esr_opts->num_broken_nodes > esr_opts->redundant_copies) //TO DO YONI
  //  failure(
  //      "Given number of redundant copies is too small to support given number "
  //      "of node failures");

  // make sure ranks of broken nodes are sorted in ascending order (some
  // communication during reconstruction relies on that assumption)
  if (esr_nodes_provided)
    gsl_sort_int(esr_opts->broken_node_ranks, 1, esr_opts->num_broken_nodes);

  // resilience is only supported for certain communication strategies
  if (esr_opts->redundant_copies &&
      (*comm_strategy == COMM_alltoallv || *comm_strategy == COMM_allgatherv))
    failure(
        "Resilience is not supported for the chosen communication strategy");

  // residual replacement is only implemented for the pipelined version
  if (solver_opts->with_residual_replacement &&
      (solver_opts->solvertype != SOLVER_pipelined))
    if (comm_rank == 0)
      printf(
          "WARNING: Residual replacement is only implemented for pipelined "
          "PCG. Your residual replacement setting will be ignored.\n");

  // if redundant copies should be stored, we also need to determine where to
  // store them
  if (esr_opts->redundant_copies)
    get_buddy_info(esr_opts, comm_rank, comm_size);

  // If reconstructing without replacement nodes, binary filenames must be
  // provided
  if (esr_opts->reconstruction_strategy == REC_on_survivors &&
      (!bmatrix_filename_provided || !brhs_filename_provided))
    failure(
        "Recovery without spares requires both a binary matrix and a binary "
        "rhs");

  // if a period is set, any node failures should only be introduced after the
  // first time the state was saved ESR reconstruction needs data from two
  // subsequent iterations, therefore broken_iteration needs to be greater than
  // period (not >=)
  if (esr_opts->period && esr_opts->broken_iteration != -1) {
    if (esr_opts->broken_iteration <= esr_opts->period)
      failure(
          "Node failure cannot be simulated before redundant information has "
          "been stored");
  }

  // if necessary, adapting the rank of the node that prints to standard output
  // to make sure all information can be printed by the same node, this has to
  // be a reconstruction node (only an issue if something gets printed during
  //reconstruction)
  if (esr_opts->broken_iteration != -1 && esr_opts->verbose_reconstruction) {
    solver_opts->output_rank =
        esr_opts->broken_node_ranks[0];  // simply using the first replacement
                                         // node (broken node ranks are already
                                         // in ascending order)
    esr_opts->reconstruction_output_rank =
        esr_opts->broken_node_ranks[0];  // same node should print during
                                         // reconstruction
  }

  // TODO: if pipelined solver was chosen, check that the chosen reconstruction
  // strategy is available (i.e. no checkpointing etc.)

  if ((esr_opts->reconstruction_strategy == REC_nvesr_homogenous ||
       esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_homogenous) &&
      (!pmem_pool_path_provided || nvesr_opts->pmem_pool_path == NULL)) {
    failure(
        "missing a path for the persistent object pool");
  }
    if ((esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_RDMA ||
       esr_opts->reconstruction_strategy == REC_checkpoint_on_nvram_RDMA) &&
      (!pmem_pool_path_provided || nvesr_opts->pmem_pool_path == NULL)) {
    failure(
        "missing a path for persistent memory");
  }
   if ((esr_opts->reconstruction_strategy == REC_checkpoint_on_disk) &&
      (!pmem_pool_path_provided || nvesr_opts->pmem_pool_path == NULL)) {
    failure(
        "missing a path for storage");
  }
}
