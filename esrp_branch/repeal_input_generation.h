#ifndef REPEAL_INPUT_GEN
#define REPEAL_INPUT_GEN

#include "repeal_utils.h"




gsl_spmatrix *get_input_problem(MPI_Comm comm, struct Input_options *input_opts);


#endif