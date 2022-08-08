#ifndef REPEAL_IO
#define REPEAL_IO

gsl_spmatrix* read_matrix(int comm_size, int comm_rank, const char* file_name);

#endif
