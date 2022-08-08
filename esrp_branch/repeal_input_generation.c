#include <mpi.h>

#include <stdio.h>
#include <gsl/gsl_spmatrix.h>

#include "repeal_input_generation.h"
#include "repeal_utils.h"
#include "repeal_io.h"
#include "repeal_binary_io.h"
#include "repeal_gsl_spmatrix_additions.h"

//for debugging: print the local matrix to standard output
void print_local_matrix(int comm_rank, gsl_spmatrix *A, int test_rank)
{
  if (comm_rank == test_rank)
  {
    printf("Local matrix on rank %d:\n", comm_rank);
    for (int i = 0; i < A->size1; ++i)
    {
      for (int j = 0; j < A->size2; ++j)
      {
        printf("%.0f ", gsl_spmatrix_get(A, i, j));
      }
      printf("\n");
    }
  }
}

//utility struct to model the points of the discretization grid
struct Point
{
  int x;
  int y;
  int z;
};

int get_index_from_coordinates(int x, int y, int z, int nx, int ny, int nz)
{
  //if the point lies outside the grid boundaries, return -1
  if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz)
    return -1;

  //calculate the global point index
  return z * ny * nx + y * nx + x;
}

struct Point get_coordinates_from_index(int index, int nx, int ny, int nz)
{
  //TODO check that the index lies in the range of legal values

  int z = index / (ny * nx);
  int remainder = index % (ny * nx);
  int y = remainder / nx;
  int x = remainder % nx;

  struct Point result = {.x = x, .y = y, .z = z};
  return result;
}

void generate_7point_stencil_input(int *num_neighbors_in_stencil, struct Point **neighbor_offsets)
{
  *num_neighbors_in_stencil = 6;

  *neighbor_offsets = malloc(6 * sizeof(struct Point));
  if (!(*neighbor_offsets))
    failure("Offset allocation failed");
  struct Point *neighbors = *neighbor_offsets; //for convenience, avoids having to constantly dereference neighbor_offsets

  //explicitly setting all offsets in the 7-point stencil
  //neighbors = [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]
  neighbors[0].x = 1;
  neighbors[0].y = 0;
  neighbors[0].z = 0;

  neighbors[1].x = -1;
  neighbors[1].y = 0;
  neighbors[1].z = 0;

  neighbors[2].y = 1;
  neighbors[2].x = 0;
  neighbors[2].z = 0;

  neighbors[3].y = -1;
  neighbors[3].x = 0;
  neighbors[3].z = 0;

  neighbors[4].z = 1;
  neighbors[4].x = 0;
  neighbors[4].y = 0;

  neighbors[5].z = -1;
  neighbors[5].x = 0;
  neighbors[5].y = 0;
}

void generate_27point_stencil_input(int *num_neighbors_in_stencil, struct Point **neighbor_offsets)
{
  *num_neighbors_in_stencil = 26;

  *neighbor_offsets = malloc(26 * sizeof(struct Point));
  if (!(*neighbor_offsets))
    failure("Offset allocation failed");
  // For convenience, avoids having to constantly dereference neighbor_offsets
  struct Point *neighbors = *neighbor_offsets;

  // Generating all offsets in the 27-point stencil
  int neighbor_index = 0;
  for (int x = -1; x <= 1; ++x)
  {
    for (int y = -1; y <= 1; ++y)
    {
      for (int z = -1; z <= 1; ++z)
      {
        if (x || y || z) //leave out the combination (0|0|0)
        {
          //neighbors[neighbor_index++] = {.x = x, .y = y, .z = z};
          neighbors[neighbor_index].x = x;
          neighbors[neighbor_index].y = y;
          neighbors[neighbor_index].z = z;
          neighbor_index++;
        }
      }
    }
  }
}

//TODO: remove test_rank
gsl_spmatrix *generate_local_matrix_from_stencil(int comm_size, int comm_rank, int num_neighbors_in_stencil, struct Point *neighbor_offsets, struct Input_options *input_opts /*, int test_rank*/)
{
  //problem size
  int grid_dim = input_opts->grid_dimension;
  size_t local_matrix_size = grid_dim * grid_dim;
  size_t global_matrix_size = local_matrix_size * comm_size;
  // This is an estimate, in reality there will be fewer nonzeros because the
  // points at the edges of the grid have fewer neighbors
  size_t matrix_nnz = (num_neighbors_in_stencil + 1) * local_matrix_size;

  //dimensions of the discretization grid (somewhat superfluous, but more legible to name them explicitly)
  int nx = grid_dim;
  int ny = grid_dim;
  int nz = comm_size;

  //allocate the matrix
  gsl_spmatrix *A_triplet = gsl_spmatrix_alloc_nzmax(local_matrix_size, global_matrix_size, matrix_nnz, GSL_SPMATRIX_TRIPLET);

  //fill with values
  double factor = input_opts->matrix_factor;
  double row_offset = comm_rank * local_matrix_size;                                    // = the global index of the first row owned by this process
  for (int local_row_index = 0; local_row_index < local_matrix_size; ++local_row_index) //for each row
  {
    //get the global index of the row
    int global_row_index = local_row_index + row_offset;

    //get the grid coordinates of the point represented by this row
    struct Point rowpoint = get_coordinates_from_index(global_row_index, nx, ny, nz);

    /*
		//testoutput - TODO: remove
		if (comm_rank == test_rank)
		{
			printf("\nRank %d: global row %d: x=%d, y=%d, z=%d\n", comm_rank, global_row_index, rowpoint.x, rowpoint.y, rowpoint.z);
		}
		*/

    //set the diagonal value
    gsl_spmatrix_set(A_triplet, local_row_index, global_row_index, -1 * num_neighbors_in_stencil * factor);

    //set the neighbors
    for (int offsetindex = 0; offsetindex < num_neighbors_in_stencil; offsetindex++)
    {
      struct Point current_offset = neighbor_offsets[offsetindex];

      int neighbor_x = rowpoint.x + current_offset.x;
      int neighbor_y = rowpoint.y + current_offset.y;
      int neighbor_z = rowpoint.z + current_offset.z;

      int neighborindex = get_index_from_coordinates(neighbor_x, neighbor_y, neighbor_z, nx, ny, nz);

      /*
			//testoutput - TODO: remove
			if (comm_rank == test_rank)
			{
				printf("\nRank %d: Marking neighbor point x=%d, y=%d, z=%d at global index %d\n", comm_rank, neighbor_x, neighbor_y, neighbor_z, neighborindex);
			}
			*/

      // An index of -1 means that the neighbor lies outside of the grid
      // boundaries and can therefore be ignored
      if (neighborindex != -1)
      {
        gsl_spmatrix_set(A_triplet, local_row_index, neighborindex, factor);
      }
    }
  }

  //convert to CCS
  gsl_spmatrix *A = gsl_spmatrix_ccs(A_triplet);
  gsl_spmatrix_free(A_triplet);

  //return
  return A;
}

gsl_spmatrix *get_input_problem(MPI_Comm comm, struct Input_options *input_opts)
{
  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  switch (input_opts->input_matrix_source)
  {

  case INPUT_from_mtx_file:
    return read_matrix(comm_size, comm_rank, input_opts->matrix_file_name);
    break;

  case INPUT_from_bin_file:
  {
    //read binary matrix from file
    int *elements_per_process, *displacements;
    gsl_spmatrix *A_crs, *A;
    int readin_groups = (comm_size / 10) + 1; // Have the groups be roughly 10 in size
    read_binary_matrix(comm, readin_groups, 0, input_opts->matrix_file_name, &elements_per_process, &displacements, &A_crs);

    //convert to CCS storage
    A = gsl_spmatrix_flip_compression(A_crs);
    gsl_spmatrix_free(A_crs);

    //these versions of elements_per_process and displacements are redundant
    if (elements_per_process)
      free(elements_per_process);
    if (displacements)
      free(displacements);
    //TODO: ideally, these vectors should not be needed in the function call

    return A;

    break;
  }
  case INPUT_generate:
  {
    //allocate the input data for the matrix generation
    int num_neighbors_in_stencil = 0;
    struct Point *neighbor_offsets = NULL;

    //Test output - TODO: remove
    //int test_rank = 0; //rank of the process that should print - only test one process at a time to avoid messy output

    //set the input data to the appropriate values
    switch (input_opts->generation_strategy)
    {
    case GEN_poisson_7:
      generate_7point_stencil_input(&num_neighbors_in_stencil, &neighbor_offsets);
      break;
    case GEN_poisson_27:
      generate_27point_stencil_input(&num_neighbors_in_stencil, &neighbor_offsets);
      break;
    default:
      failure("Unknown matrix generation strategy");
    }

    //generate the matrix from the input data
    gsl_spmatrix *local_matrix = generate_local_matrix_from_stencil(comm_size, comm_rank, num_neighbors_in_stencil, neighbor_offsets, input_opts /*, test_rank*/);

    //free any malloc'd input data
    free(neighbor_offsets);

    //test output - TODO: remove
    //print_local_matrix(comm_rank, local_matrix, test_rank);

    //return the matrix
    return local_matrix;

    break;
  }

  default:
    failure("Unknown input source");
  }

  return NULL; //TODO: remove - this is dead code and just exists to avoid compiler complaints
}
