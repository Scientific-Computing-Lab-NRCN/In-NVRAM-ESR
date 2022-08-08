#include "repeal_binary_io.h"

#include <assert.h>
#include <stdio.h>
#include <gsl/gsl_spmatrix.h>

#include "mmio.h"
#include "repeal_utils.h"

// Endianness conversion functions

static Int to_LE_Int(Int arg) {
  return ((arg & 0xFF000000) >> 24) | ((arg & 0xFF0000) >> 8) |
         ((arg & 0xFF00) << 8) | ((arg & 0xFF) << 24);
}

static Scalar to_LE_Scalar(Scalar arg) {
  Scalar output;
  char *outc = (char *)&output;
  char *inc = (char *)&arg;

  outc[7] = inc[0];
  outc[6] = inc[1];
  outc[5] = inc[2];
  outc[4] = inc[3];
  outc[3] = inc[4];
  outc[2] = inc[5];
  outc[1] = inc[6];
  outc[0] = inc[7];

  return output;
}

static void read_binary_matrix_metadata(FILE *binary_tape, size_t *rows,
                                        size_t *columns, size_t *nnz) {
  Int metadata[4];

  // Read in metadata: Type ID, rows, columns and nnz
  fread((void *)metadata, sizeof(Int), 4, binary_tape);
  const Int type_id = to_LE_Int(metadata[0]);
  const Int matrix_rows = to_LE_Int(metadata[1]);
  const Int matrix_cols = to_LE_Int(metadata[2]);
  const Int matrix_nnz = to_LE_Int(metadata[3]);

  assert(type_id == 1211216);  // As written by PETSc

  *rows = (size_t)matrix_rows;
  *columns = (size_t)matrix_cols;
  *nnz = (size_t)matrix_nnz;
}

static void read_binary_vector_metadata(FILE *binary_tape, size_t *size) {
  Int metadata[2];

  // Metadata: ID and size
  fread((void *)metadata, sizeof(Int), 2, binary_tape);

  const Int type_id = to_LE_Int(metadata[0]);
  const Int vec_size = to_LE_Int(metadata[1]);

  assert(type_id == 1211214);

  *size = (size_t)vec_size;
}

static void read_doubles(FILE *binary_tape, size_t start, size_t end,
                         size_t size, Scalar *const values) {
  assert(end <= size);
  assert(start >= 0);

  const size_t len = end - start;

  fseek(binary_tape, start * sizeof(Scalar), SEEK_CUR);

  fread((void *)values, sizeof(Scalar), len, binary_tape);

  for (size_t i = 0; i < len; i++) {
    values[i] = to_LE_Scalar(values[i]);
  }

  fseek(binary_tape, (size - end) * sizeof(Scalar), SEEK_CUR);
}

static void read_row_data(FILE *binary_tape, Int *const I, size_t start,
                          size_t end, size_t matrix_rows, size_t *nnz_start,
                          size_t *nnz_end, size_t *nnz) {
  fread((void *)I, sizeof(Int), matrix_rows,
        binary_tape);  // Offset for the leading zero

  // Change endianness
  for (Int i = 0; i < matrix_rows; i++) {
    I[i] = to_LE_Int(I[i]);
  }

  // Find the accumulated non-zeros at the start, end and total
  *nnz_start = 0;
  for (size_t i = 0; i < start; i++) {
    *nnz_start += (size_t)I[i];
  }
  *nnz_end = *nnz_start;
  for (size_t i = start; i < end; i++) {
    *nnz_end += (size_t)I[i];
  }

  assert(*nnz_end >= *nnz_start);
  *nnz = *nnz_end - *nnz_start;
}

void read_column_and_value_data(FILE *binary_tape, size_t nnz_start,
                                size_t nnz_end, size_t nnz, size_t matrix_nnz,
                                Int *J, Scalar *values) {
  // Load column indices
  fseek(binary_tape, nnz_start * sizeof(Int),
        SEEK_CUR);  // Skip the rows before start
  fread((void *)J, sizeof(Int), nnz, binary_tape);
  for (Int i = 0; i < nnz; i++) {
    J[i] = to_LE_Int(J[i]);
  }

  // Load data
  // Skip until start of data
  fseek(binary_tape,
        (matrix_nnz - nnz_end) * sizeof(Int) + nnz_start * sizeof(Scalar),
        SEEK_CUR);
  fread((void *)values, sizeof(Scalar), nnz, binary_tape);
  for (Int i = 0; i < nnz; i++) {
    values[i] = to_LE_Scalar(values[i]);
  }
}

void read_partial_petsc_matrix_single_block(const char *const filename,
                                            size_t start, size_t end,
                                            size_t *rows, size_t *columns,
                                            size_t *nz, size_t **row_nz,
                                            size_t **column_indices,
                                            double **values_array) {
  Int *I, *J;
  Scalar *values;
  size_t nnz_start, nnz_end, nnz;
  size_t matrix_rows, matrix_nnz;

  assert(start < end);

  // Some size assertions so that this works with my test file
  assert(sizeof(Int) == 4);
  assert(sizeof(Scalar) == 8);

  FILE *binary_tape;

  if (nz == NULL || rows == NULL || columns == NULL) {
    fprintf(stderr,
            "Warning: Passed a null pointer to the scalar values of partial "
            "matrix reader\n");
    return;
  }

  binary_tape = fopen(filename, "rb");

  read_binary_matrix_metadata(binary_tape, &matrix_rows, columns, &matrix_nnz);

  assert(matrix_rows >= end);

  // Since we will read the matrix partially, we need to know where the rows
  // are. For this, we need to scan the sizes array.

  // The file only contains the number of entries per row. It's not the row
  // pointer array.
  I = (Int *)malloc((matrix_rows) * sizeof(Int));

  read_row_data(binary_tape, I, start, end, matrix_rows, &nnz_start, &nnz_end,
                &nnz);

  // We can now allocate the other arrays
  J = (Int *)malloc(nnz * sizeof(Int));
  values = (Scalar *)malloc(nnz * sizeof(Scalar));

  read_column_and_value_data(binary_tape, nnz_start, nnz_end, nnz, matrix_nnz,
                             J, values);

  fclose(binary_tape);

  // Set return values
  *nz = nnz;
  *rows = end - start;
  // Columns were set already

  // Realistically speaking, _Scalar_ will always be double. Just return the
  // array
  *values_array = values;

  *row_nz = (size_t *)malloc((*rows) * sizeof(size_t));
  *column_indices = (size_t *)malloc(nnz * sizeof(size_t));

  for (size_t i = 0; i < *rows; i++) {
    (*row_nz)[i] = (size_t)(I[start + i]);
  }

  for (size_t i = 0; i < nnz; i++) {
    (*column_indices)[i] = (size_t)(J[i]);
  }

  free(I);
  free(J);
}

void read_binary_matrix(MPI_Comm comm, const int nof_groups, const int root,
                        const char *const filename, int **rows_per_process,
                        int **row_displacements, gsl_spmatrix **matrix) {
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  int sub_size, sub_rank;

  MPI_Comm subcomm;

  int color;
  int *startend;

  size_t matrix_rows, matrix_columns, matrix_nnz;

  FILE *binary_tape;

  *rows_per_process = (int *)malloc(comm_size * sizeof(int));
  *row_displacements = (int *)malloc(comm_size * sizeof(int));

  // Root will check the size of the matrix and decide the group distribution
  if (comm_rank == root) {
    binary_tape = fopen(filename, "rb");
    read_binary_matrix_metadata(binary_tape, &matrix_rows, &matrix_columns,
                                &matrix_nnz);
    fclose(binary_tape);

    const int base_size = (int)(matrix_rows / comm_size);
    const int modulo = (int)(matrix_rows % comm_size);
    for (int i = 0; i < comm_size; i++) {
      (*rows_per_process)[i] = base_size;
    }
    for (int i = 0; i < modulo; i++) {
      (*rows_per_process)[i]++;
    }

    (*row_displacements)[0] = 0;
    for (int i = 1; i < comm_size; i++) {
      (*row_displacements)[i] =
          (*row_displacements)[i - 1] + (*rows_per_process)[i - 1];
    }
  }

  MPI_Bcast(*rows_per_process, comm_size, MPI_INT, root, comm);
  MPI_Bcast(*row_displacements, comm_size, MPI_INT, root, comm);

  // Now, everyone must decide their color
  const int base_group_size = comm_size / nof_groups;
  const int group_remainder = comm_size % nof_groups;
  int group_start;
  if (comm_rank < group_remainder * (base_group_size + 1)) {
    color = comm_rank / (base_group_size + 1);
    group_start = color * (base_group_size + 1);
  } else {
    color = group_remainder;
    group_start = group_remainder * (base_group_size + 1);

    const int supplement =
        (comm_rank - group_remainder * (base_group_size + 1));

    color += supplement / base_group_size;
    group_start += (supplement / base_group_size) * base_group_size;
  }

  MPI_Comm_split(comm, color, comm_rank, &subcomm);
  MPI_Comm_size(subcomm, &sub_size);
  MPI_Comm_rank(subcomm, &sub_rank);

  startend = (int *)malloc((sub_size + 1) * sizeof(int));

  for (int i = 0; i < sub_size; i++) {
    startend[i] = (*row_displacements)[group_start + i];
  }
  startend[sub_size] =
      startend[sub_size - 1] + (*rows_per_process)[group_start + sub_size - 1];

  partial_read_matrix_block_split_in_comm(subcomm, 0, startend, filename,
                                          matrix);

  free(startend);
}

void partial_read_matrix_block_split_in_comm(MPI_Comm comm, int root,
                                             const int *const startend,
                                             const char *const filename,
                                             gsl_spmatrix **matrix) {
  int comm_size, comm_rank;
  size_t rows, columns, nz;
  size_t *row_nz = NULL, *all_column_indices = NULL;
  double *all_values;

  size_t *I;

  int *sizes_array = NULL, *displ_array = NULL;
  size_t local_nnz;
  size_t local_rows;

  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  local_rows = startend[comm_rank + 1] - startend[comm_rank];

  I = (size_t *)malloc(local_rows *
                       sizeof(size_t));  // One more for prefixing later

  if (comm_rank == root) {
    const size_t start_all = startend[0];
    const size_t end_all = startend[comm_size];
    read_partial_petsc_matrix_single_block(filename, start_all, end_all, &rows,
                                           &columns, &nz, &row_nz,
                                           &all_column_indices, &all_values);
    sizes_array = (int *)malloc(comm_size * sizeof(int));
    displ_array = (int *)malloc(comm_size * sizeof(int));

    // Create comm arrays to communicate the row_nz array
    displ_array[0] = 0;
    sizes_array[0] =
        startend[1] - startend[0];  // Should exist. Comm_s>1. stend>2
    for (int i = 1; i < comm_size; i++) {
      sizes_array[i] = startend[i + 1] - startend[i];
      displ_array[i] = displ_array[i - 1] + sizes_array[i - 1];
    }
  }

  MPI_Scatterv(row_nz, sizes_array, displ_array, my_MPI_SIZE_T, I, local_rows,
               my_MPI_SIZE_T, root, comm);
  MPI_Bcast(&columns, 1, my_MPI_SIZE_T, root, comm);

  local_nnz = 0;
  for (size_t i = 0; i < local_rows; i++) {
    local_nnz += I[i];
  }

  // Now that we have the nnz, we can allocate a matrix,
  // and then use its arrays to collect data into
  *matrix = gsl_spmatrix_alloc_nzmax(local_rows, columns, local_nnz,
                                     GSL_SPMATRIX_CRS);

  // Reconstruct the row pointer array
  (*matrix)->p[0] = 0;
  for (size_t i = 1; i <= local_rows; i++) {
    (*matrix)->p[i] = (*matrix)->p[i - 1] + I[i - 1];
  }

  // Prepare transfer of column indices and values
  if (comm_rank == root) {
    for (int i = 0; i < comm_size; i++) {
      sizes_array[i] = 0;
      for (int j = startend[i] - startend[0]; j < startend[i + 1] - startend[0];
           j++) {
        sizes_array[i] += (int)row_nz[j];
      }
    }
    displ_array[0] = 0;
    for (int i = 1; i < comm_size; i++) {
      displ_array[i] = displ_array[i - 1] + sizes_array[i - 1];
    }
  }

  // Transfer data
  MPI_Scatterv(all_values, sizes_array, displ_array, MPI_DOUBLE,
               (*matrix)->data, local_nnz, MPI_DOUBLE, root, comm);
  MPI_Scatterv(all_column_indices, sizes_array, displ_array, my_MPI_SIZE_T,
               (*matrix)->i, local_nnz, my_MPI_SIZE_T, root, comm);

  // Set other matrix details
  (*matrix)->nz = local_nnz;

  if (comm_rank == root) {
    free(sizes_array);
    free(displ_array);
  }
}

void read_partial_petsc_vector_single_block(const char *const filename,
                                            size_t start, size_t end,
                                            size_t *size, double **values) {
  size_t total_vector_length;
  const size_t len = end - start;
  assert(start < end);

  // Some size assertions so that this works with my test file
  assert(sizeof(Int) == 4);
  assert(sizeof(Scalar) == 8);

  FILE *binary_tape;

  if (size == NULL || values == NULL) {
    fprintf(stderr, "Warning: Passed a null pointer\n");
    return;
  }

  binary_tape = fopen(filename, "rb");

  read_binary_vector_metadata(binary_tape, &total_vector_length);

  assert(end <= total_vector_length);

  *values = (double *)malloc(len * sizeof(double));

  read_doubles(binary_tape, start, end, end, *values);

  *size = len;
}

void partial_read_vector_block_split_in_comm(MPI_Comm comm, int root,
                                             const int *const startend,
                                             const char *const filename,
                                             gsl_vector **vector) {
  int comm_rank, comm_size;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  double *all_values = NULL;

  int *sizes_array = NULL, *displ_array = NULL;

  size_t local_size = startend[comm_rank + 1] - startend[comm_rank];

  *vector = gsl_vector_alloc(local_size);

  if (comm_rank == root) {
    const size_t start_all = startend[0];
    const size_t end_all = startend[comm_size];

    size_t len;

    read_partial_petsc_vector_single_block(filename, start_all, end_all, &len,
                                           &all_values);

    sizes_array = (int *)malloc(comm_size * sizeof(int));
    displ_array = (int *)malloc(comm_size * sizeof(int));

    for (int i = 0; i < comm_size; i++) {
      sizes_array[i] = startend[i + 1] - startend[i];
      displ_array[i] = startend[i] - startend[0];
    }
  }

  MPI_Scatterv(all_values, sizes_array, displ_array, MPI_DOUBLE,
               (*vector)->data, local_size, MPI_DOUBLE, root, comm);

  free(all_values);

  if (comm_rank == root) {
    free(sizes_array);
    free(displ_array);
  }
}
