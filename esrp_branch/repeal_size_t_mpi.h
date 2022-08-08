#ifndef REPEAL_SIZE_T_MPI
#define REPEAL_SIZE_T_MPI

// Use a macro to fix the size of size_t for MPI, as described in
// https://stackoverflow.com/questions/40807833/sending-size-t-type-data-with-mpi (Gilles)

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

#endif
