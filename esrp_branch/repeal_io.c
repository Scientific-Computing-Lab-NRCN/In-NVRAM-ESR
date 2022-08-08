#include <stdio.h>
#include <gsl/gsl_spmatrix.h>

#include "mmio.h"
#include "repeal_io.h"
#include "repeal_utils.h"




//*************************************************
//Reading matrix from file
//*************************************************
gsl_spmatrix*
read_matrix(int comm_size, int comm_rank, const char* file_name)
{
    // Open matrix file stream
    FILE* matrix_stream = fopen(file_name, "r");
    if (matrix_stream == NULL)
        failure("Matrix Market file cannot be read.");

    // Check matrix file
    MM_typecode matrix_type;
    if (mm_read_banner(matrix_stream, &matrix_type) != 0)
        failure("Matrix Market file banner cannot be read.");
    if (!mm_is_matrix(matrix_type) || !mm_is_sparse(matrix_type) ||
            !mm_is_real(matrix_type) || !mm_is_symmetric(matrix_type))
        failure("Only sparse, real, and symmetric matrices supported.");

    // Determine rows for current process
    int m, n, nnz;
    if (mm_read_mtx_crd_size(matrix_stream, &m, &n, &nnz) != 0)
        failure("Matrix size cannot be read.");
    if (m != n)
        failure("Only square matrices supported.");
    int n_rows = n / comm_size;
    int n_rem_rows = n % comm_size;
    int first_row = comm_rank * n_rows + min(comm_rank, n_rem_rows) + 1;
    int last_row = (comm_rank + 1) * n_rows + min(comm_rank + 1, n_rem_rows);
    n_rows = last_row - first_row + 1;

    // Keep only coordinate triplets for current process
    char* triplets_buffer;
    size_t buffer_size;
    FILE* triplets_stream = open_memstream(&triplets_buffer, &buffer_size);
    char triplet[MM_MAX_LINE_LENGTH];
    int row, column, row_t, column_t;
    char value[MM_MAX_LINE_LENGTH];
    nnz = 0;
    while (fgets(triplet, MM_MAX_LINE_LENGTH, matrix_stream) != NULL)
	{
        if (sscanf(triplet, "%i %i %s", &row, &column, value) != 3)
            failure("Invalid coordinate triple encountered.");
		row_t = column;
		column_t = row;
        if (row >= first_row && row <= last_row)
		{
            nnz++;
            row = row - first_row + 1;
            if (fprintf(triplets_stream, "%i %i %s\n", row, column, value) < 0)
                failure("Coordinate triple cannot be written.");
        }

		//matrix market format: symmetric matrices only need to store the lower triangular portion
		//this means when reading a triplet, we also have to store its transposed
		if (row_t != column_t && row_t >= first_row && row_t <= last_row) //if it's a diagonal entry, no need to check it again
		{
            nnz++;
            row_t = row_t - first_row + 1;
            if (fprintf(triplets_stream, "%i %i %s\n", row_t, column_t, value) < 0)
                failure("Coordinate triple cannot be written.");
        }
    }
    fclose(matrix_stream);
    fclose(triplets_stream);

    // Build submatrix for current process in Matrix Market format
    char* submatrix_buffer;
    FILE* submatrix_stream = open_memstream(&submatrix_buffer, &buffer_size);
    if (mm_write_banner(submatrix_stream, matrix_type) != 0)
        failure("Matrix Market file banner cannot be written.");
    if (mm_write_mtx_crd_size(submatrix_stream, n_rows, n, nnz) != 0)
        failure("Matrix size cannot be written.");
    if (fputs(triplets_buffer, submatrix_stream) == EOF)
        failure("Coordinate triples cannot be written.");
    fflush(submatrix_stream);
    free(triplets_buffer);

    // Construct GSL matrix and convert matrix to compressed column format
    rewind(submatrix_stream);
    gsl_spmatrix* triplet_matrix = gsl_spmatrix_fscanf(submatrix_stream); //this also changes the indices from 1-based to 0-based

    //test output: print triplet_matrix
    //printf("Triplet matrix:\n");
    //gsl_spmatrix_fprintf(stdout, triplet_matrix, "%f");
    //printf("Storage type: %ld\n", triplet_matrix->sptype);

    fclose(submatrix_stream);
    free(submatrix_buffer);
    gsl_spmatrix* matrix = gsl_spmatrix_ccs(triplet_matrix); //compressed COLUMN storage because some communication later on relies on that

    //test output: print matrix
    //printf("\nMatrix:\n");
    //gsl_spmatrix_fprintf(stdout, matrix, "%f");
    //printf("Storage type: %ld\n", matrix->sptype);

    gsl_spmatrix_free(triplet_matrix);
    return matrix;
}



