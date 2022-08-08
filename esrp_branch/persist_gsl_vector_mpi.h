#include <mpi.h>
#include <mpi_one_sided_extension/mpi_win_pmem.h>
#include <mpi_one_sided_extension/defines.h>

#include <stdio.h>
#include <gsl/gsl_vector.h>

#include <stdint.h>
#include <limits.h>

#if SIZE_MAX == UCHAR_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "what is happening here?"
#endif

MPI_Aint put_gsl_vector_in_win(int comm_rank, int target, MPI_Win_pmem win, bool win_ram_on, MPI_Aint target_disp, gsl_vector *v, size_t local_size);
MPI_Aint put_scalar_in_win(int comm_rank, int target, MPI_Win_pmem win, bool win_ram_on, MPI_Aint target_disp, double value);
double* get_vector_from_win(int comm_rank, MPI_Win_pmem win, bool win_ram_on, MPI_Aint target_disp, size_t local_size);
double get_scalar_from_win(int comm_rank, MPI_Win_pmem win, bool win_ram_on, MPI_Aint target_disp);

