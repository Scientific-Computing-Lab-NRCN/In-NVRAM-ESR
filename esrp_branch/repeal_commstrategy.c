#include <stdio.h>
#include <gsl/gsl_spmatrix.h>
#include <mpi.h>
#include <assert.h>

#include "repeal_utils.h"
#include "repeal_commstrategy.h"








struct SPMV_comm_structure *alloc_spmv_comm_structure(int comm_size)
{
	struct SPMV_comm_structure *comm_struct = malloc(sizeof(struct SPMV_comm_structure));
	if (!comm_struct) failure("Struct allocation failed.");

    comm_struct->recvcounts = (int*) malloc(comm_size * sizeof(int));
    comm_struct->recvdispl = (int*) malloc(comm_size * sizeof(int));
    comm_struct->sendcounts = (int*) malloc(comm_size * sizeof(int));
    comm_struct->senddispl = (int*) malloc(comm_size * sizeof(int));
    if ( !(comm_struct->recvcounts) || !(comm_struct->recvdispl) || !(comm_struct->sendcounts) || !(comm_struct->senddispl) )
        failure("Struct allocation failed.");

	return comm_struct;
}


struct SPMV_comm_structure_minimal *alloc_spmv_comm_structure_minimal(int comm_size)
{
	struct SPMV_comm_structure_minimal *comm_struct = malloc(sizeof(struct SPMV_comm_structure_minimal));
	if (!comm_struct) failure("Struct allocation failed.");

    comm_struct->sendcounts = (int*) malloc(comm_size * sizeof(int));
	comm_struct->recvcounts = (int*) malloc(comm_size * sizeof(int));
    comm_struct->senddispl = (int*) calloc(comm_size, sizeof(int)); //all send displacements are zero
	comm_struct->recvdispl = (int*) malloc(comm_size * sizeof(int));
	comm_struct->sendtypes = (MPI_Datatype*) malloc(comm_size * sizeof(MPI_Datatype));
	comm_struct->recvtypes = (MPI_Datatype*) malloc(comm_size * sizeof(MPI_Datatype));

    if ( !(comm_struct->sendcounts) || !(comm_struct->recvcounts) || !(comm_struct->recvdispl) || !(comm_struct->senddispl) || !(comm_struct->sendtypes) || !(comm_struct->recvtypes) )
        failure("Struct allocation failed.");

	return comm_struct;
}




struct SPMV_comm_structure *get_spmv_alltoallv_info(MPI_Comm comm, int comm_rank, int comm_size, const gsl_spmatrix* matrix, const int *elements_per_process, const int *displacements)
{
	size_t global_size = matrix->size2;

    //allocate struct to store the information necessary for alltoallv
	struct SPMV_comm_structure *comm_struct = alloc_spmv_comm_structure(comm_size);

    int* required = (int*) calloc(global_size, sizeof(int));
    if (!required) failure("Vector allocation failed.");

	//make sure the matrix is stored in compressed column storage
	assert(matrix->sptype == GSL_SPMATRIX_CCS);

	//compressed column storage: matrix->p holds the "pointers" to the start of new columns
	//if column j is empty, p[j] has the same value as p[j+1]
	//we want to mark the columns that are not empty
	for (int j = 0; j < global_size; ++j)
		if (matrix->p[j] != matrix->p[j+1])
			required[j] = 1;

    //calculate starting index and number of elements needed from each process
    for (int rank = 0; rank < comm_size; ++rank)
	{
        int limit = displacements[rank] + elements_per_process[rank]; //upper limit for elements owned by that process
        int first_required = -1, last_required = -1;

        //find the index of the first element needed from that process
        for (int i = displacements[rank]; i < limit; ++i)
            if (required[i] == 1)
			{
                first_required = last_required = i;
                break;
            }

        //find the index of the last element needed from that process
        for (int i = last_required + 1; i < limit; ++i)
            if (required[i] == 1)
                last_required = i;

        //calculate the number of elements needed from that process
        int num_required;
        if (first_required == -1)
		{
            num_required = 0;
            first_required = 0;
        }
        else
            num_required = last_required - first_required + 1; //need all elements from the first to the last nonzero

        comm_struct->recvdispl[rank] = first_required;
        comm_struct->recvcounts[rank] = num_required;
    }

    //buffers for distributing that information among all processes
    int* outbuffer = (int*) malloc(2*comm_size * sizeof(int));
    int* inbuffer = (int*) malloc(2*comm_size * sizeof(int));
    if (!(outbuffer) || !(inbuffer) ) failure("Vector allocation failed.");

    //Collective communication: telling each process which elements it should send
    for (int rank = 0; rank < comm_size; ++rank)
	{
        outbuffer[2*rank] = comm_struct->recvdispl[rank] - displacements[rank]; //local rank at receiving process
        outbuffer[2*rank + 1] = comm_struct->recvcounts[rank];
    }
    MPI_Alltoall(outbuffer, 2, MPI_INT, inbuffer, 2, MPI_INT, comm);

    //save results in struct
    for (int rank = 0; rank < comm_size; ++rank)
	{
        comm_struct->senddispl[rank] = inbuffer[2*rank];
        comm_struct->sendcounts[rank] = inbuffer[2*rank + 1];
    }

    free(outbuffer);
    free(inbuffer);

    return comm_struct;
}



struct SPMV_comm_structure_minimal *get_spmv_minimal_info(MPI_Comm comm, int comm_rank, int comm_size, const gsl_spmatrix* M, const int *elements_per_process, const int *displacements, const int redundant_copies, const int *buddies_that_save_me)
{
	size_t local_size = M->size1;
	size_t global_size = M->size2;

	/***** allocate struct that will hold all the information necessary for communication during SPMV *****/

	struct SPMV_comm_structure_minimal *comm_struct = alloc_spmv_comm_structure_minimal(comm_size);


	/***** find out which elements are required at the local process *****/

	//allocate memory
	int* required_by_me = (int*) calloc(global_size, sizeof(int)); //would probably be better to use a sparse representation here at some point
    if (!required_by_me) failure("Vector allocation failed.");

	//make sure the matrix is stored in compressed column storage
	assert(M->sptype == GSL_SPMATRIX_CCS);

	//compressed column storage: matrix->p holds the "pointers" to the start of new columns
	//if column j is empty, p[j] has the same value as p[j+1]
	//we want to mark the columns that are not empty
	for (int j = 0; j < global_size; ++j)
		if (M->p[j] != M->p[j+1])
			required_by_me[j] = 1;





	/***** resilience *****/

	if (redundant_copies)
	{

		//allocate required_from_me array (comm_size * M->size1)
		int *required_from_me = (int*) malloc(comm_size * local_size * sizeof(int));
		int *counts_localsize = (int*) malloc(comm_size * sizeof(int));
		int *displ_localsize = (int*) malloc(comm_size * sizeof(int));
		if (!required_from_me || !counts_localsize || !displ_localsize) failure("Memory allocation failed.");
		for (int i = 0; i < comm_size; ++i)
			counts_localsize[i] = local_size;
		for (int i = 0; i < comm_size; ++i)
			displ_localsize[i] = i * local_size;

		//alltoallv(required_by_me) to fill required_from_me
		MPI_Alltoallv(required_by_me, elements_per_process, displacements, MPI_INT, required_from_me, counts_localsize, displ_localsize, MPI_INT, comm);

		//count multiplicity of elements
		int *multiplicity = (int*) calloc(local_size, sizeof(int));
		if (!multiplicity) failure("Vector allocation failed.");
		for (int i = 0; i < comm_size; ++i) //rank
			if (i != comm_rank) //elements that are needed locally don't count, we're only interested in remote copies!
				for (int j = 0; j < local_size; ++j) //index of vector element
					if ( required_from_me[local_size * i + j] ) multiplicity[j]++;

		//for all entries that have less than the required multiplicity:
			//add additional entries into required_from_me to indicate that elements should be sent to buddies
		int buddyindex, next_buddy;
		for (int i = 0; i < local_size; ++i) //index of vector element
		{
			buddyindex = 0;
			while (multiplicity[i] < redundant_copies)
			{
				next_buddy = buddies_that_save_me[buddyindex++];
				assert(buddyindex <= redundant_copies); //if that's not the case, the logic is broken

				if (required_from_me[next_buddy * local_size + i] == 0) //the buddy does not yet get this element, we can send it there as an extra copy
				{
					required_from_me[next_buddy * local_size + i] = 1;
					multiplicity[i]++;
				}
			}
		}

		//alltoallv(required_from_me) to update required_by_me
		MPI_Alltoallv(required_from_me, counts_localsize, displ_localsize, MPI_INT, required_by_me, elements_per_process, displacements, MPI_INT, comm);
		//now required_by_me can be used to create the custom datatypes just like in the non-resilient case

		//free memory
		free(required_from_me);
		free(counts_localsize);
		free(displ_localsize);
		free(multiplicity);
	}



	/***** prepare everything needed to create the custom datatypes *****/

	int count, len, index, limit, current_pos;
	//allocating more space than needed for blocklengths and offsets --> don't have to keep allocating the exact amount of memory that is needed for each datatype
	int *num_blocks_received_by_me_per_process = malloc(comm_size * sizeof(int)); //number of blocks of data I will receive from each process
	int *my_blocklengths = malloc(global_size * sizeof(int)); //length of each block I will receive
	int *my_offsets = malloc(global_size * sizeof(int)); //local starting position (at sending process) of each block I will receive
	if (!num_blocks_received_by_me_per_process || !my_blocklengths || !my_offsets) failure("Memory allocation failed.");

	//from required_by_me array, read the appropriate values for num_blocks_received_by_me_per_process, my_blocklengths, and my_offsets
	current_pos = 0;
	for (int i = 0; i < comm_size; ++i)
	{
		count = 0;
		limit = displacements[i] + elements_per_process[i];
		for (index = displacements[i]; index < limit; ++index) //iterate over all indices owned by process i
		{
			if (required_by_me[index])
			{
				my_offsets[current_pos] = index - displacements[i]; //saving the local index (i.e. local at process i)
				len = 1;
				while (++index < limit && required_by_me[index])
					len++;
				my_blocklengths[current_pos] = len;
				count++;
				current_pos++;
			}
		}
		num_blocks_received_by_me_per_process[i] = count;
	}

	free(required_by_me);

	//collective communication: counts
	int *num_blocks_to_i_from_j = malloc(comm_size * comm_size * sizeof(int)); //will hold the number of blocks of data sent from each process to each process
	if (!num_blocks_to_i_from_j) failure("Vector allocation failed.");
	MPI_Allgather(num_blocks_received_by_me_per_process, comm_size, MPI_INT, num_blocks_to_i_from_j, comm_size, MPI_INT, comm);
	//num_blocks_to_i_from_j[i][j] = number of blocks sent from process j to process i
	free(num_blocks_received_by_me_per_process);

	//test output
	/*
	if (comm_rank == 0)
	{
		printf("num_blocks_to_i_from_j:\n");
		for (int i = 0; i < comm_size; ++i)
		{
			for (int j = 0; j < comm_size; ++j)
				printf("%02d  ", num_blocks_to_i_from_j[comm_size*i+j]);
			printf("\n");
		}
	}
	*/

	//sum up the counts to find out how much memory is needed for the next communication
	int num_blocks_sent_total = 0; //total number of data blocks that will be sent (needed for memory allocation)
	for (int i = 0; i < comm_size*comm_size; ++i)
		num_blocks_sent_total += num_blocks_to_i_from_j[i];

	//find out the total number of communications happening (i.e. the number of datatypes that have to be created)
	//communications are non-symmetric, i.e. if i sends to j and j sends to i, that counts as two communications
	int num_communications = 0;
	for (int i = 0; i < comm_size*comm_size; ++i)
		if (num_blocks_to_i_from_j[i] > 0) num_communications++;

	int* num_blocks_received_by_each_process = malloc(comm_size * sizeof(int)); //for each rank, stores the number of blocks of data that this rank will receive
	if (!num_blocks_received_by_each_process) failure("Vector allocation failed");
	for (int i = 0; i < comm_size; ++i)
	{
		int sum = 0;
		for (int j = 0; j < comm_size; ++j)
			sum += num_blocks_to_i_from_j[comm_size*i + j];
		num_blocks_received_by_each_process[i] = sum;
	}

	int* displs = malloc(comm_size * sizeof(int)); //displacements to determine where to put the data during the next communication rounds
	if (!displs) failure("Vector allocation failed");
	displs[0] = 0;
	for (int i = 1; i < comm_size; ++i)
		displs[i] = displs[i-1] + num_blocks_received_by_each_process[i-1];

	//allocate memory for blocklengths and offsets
	int *all_blocklengths = malloc(num_blocks_sent_total * sizeof(int));
	int *all_offsets = malloc(num_blocks_sent_total * sizeof(int));
	if (!all_blocklengths || !all_offsets) failure("Vector allocation failed");

	//communication: blocklengths and offsets
	MPI_Allgatherv(my_blocklengths, num_blocks_received_by_each_process[comm_rank], MPI_INT, all_blocklengths, num_blocks_received_by_each_process, displs, MPI_INT, comm);
	MPI_Allgatherv(my_offsets, num_blocks_received_by_each_process[comm_rank], MPI_INT, all_offsets, num_blocks_received_by_each_process, displs, MPI_INT, comm);

	free(my_blocklengths);
	free(my_offsets);
	free(num_blocks_received_by_each_process);
	free(displs);

	//allocate memory to store the custom datatypes
	MPI_Datatype* datatypes = malloc(num_communications * sizeof(MPI_Datatype));
	if (!datatypes) failure("Datatype allocation failed");

	//pointers to facilitate access into blocklengths and offsets arrays
	int *current_bl = all_blocklengths;
	int *current_offs = all_offsets;


	/***** create custom datatypes (one datatype for each communication between any two ranks) *****/

	int datatype_index = 0;
	for (int i = 0; i < comm_size; ++i)
	{
		for (int j = 0; j < comm_size; ++j)
		{
			//i is the receiving process
			//j is the sending process

			if (num_blocks_to_i_from_j[comm_size*i + j]) //data is being sent from process i to process j, need to create datatype
			{
				//create and commit datatype for sending from rank j to rank i
				MPI_Type_indexed(num_blocks_to_i_from_j[comm_size*i + j], current_bl, current_offs, MPI_DOUBLE, &datatypes[datatype_index]);
				MPI_Type_commit(&datatypes[datatype_index]);

				//make the appropriate processes remember which datatypes to use
				if (comm_rank == j) //I am the sending process
				{
					comm_struct->sendtypes[i] = datatypes[datatype_index];
					comm_struct->sendcounts[i] = 1;
				}
				if (comm_rank == i) //I am the receiving process
				{
					comm_struct->recvtypes[j] = datatypes[datatype_index];
					comm_struct->recvcounts[j] = 1;
				}

				//update pointers and index counter so they're in place for the next iteration
				current_bl += num_blocks_to_i_from_j[comm_size*i + j];
				current_offs += num_blocks_to_i_from_j[comm_size*i + j];
				datatype_index++;
			}
			else //no data is sent from process i to process j
			{
				if (comm_rank == j) //I am the sending process
				{
					comm_struct->sendtypes[i] = MPI_DOUBLE; //this communication is not happening, but we need a placeholder datatype to avoid an error
					comm_struct->sendcounts[i] = 0;
				}
				if (comm_rank == i) //I am the receiving process
				{
					comm_struct->recvtypes[j] = MPI_DOUBLE;
					comm_struct->recvcounts[j] = 0;
				}
			}
		}
	}

	free(all_blocklengths);
	free(all_offsets);
	free(num_blocks_to_i_from_j);

	//receive displacements for SPMV (needed in bytes)
	for (int i = 0; i < comm_size; ++i)
		comm_struct->recvdispl[i] = displacements[i] * sizeof(double);
	//send displacements are already filled with zeros upon allocation

	//provide a handle to the custom datatypes so they can be deallocated later on
	comm_struct->datatypes = datatypes;
	comm_struct->num_datatypes = num_communications;

	return comm_struct;
}


void free_spmv_comm_structure_minimal(struct SPMV_comm_structure_minimal *comm_struct)
{
	//deallocate the data types
	for (int i = 0; i < comm_struct->num_datatypes; ++i)
		MPI_Type_free(comm_struct->datatypes + i);
	free(comm_struct->datatypes);

    free(comm_struct->sendcounts);
	free(comm_struct->recvcounts);
    free(comm_struct->recvdispl);
    free(comm_struct->senddispl);
	free(comm_struct->sendtypes);
	free(comm_struct->recvtypes);

	free(comm_struct);
}


void free_spmv_comm_structure(struct SPMV_comm_structure *comm_struct)
{
    free(comm_struct->recvcounts);
    free(comm_struct->recvdispl);
    free(comm_struct->sendcounts);
    free(comm_struct->senddispl);

	free(comm_struct);
}

//convenience function to let a node calculate the buddies of some other rank
void get_buddies_for_rank(int rank, int comm_size, int redundant_copies, int *buddies)
{
	int i;
	int limit = redundant_copies/2;
	for (i = 0; i < limit; ++i)
	{
		buddies[2*i] = ( rank + (i+1) ) % comm_size; //neighbour i+1 steps to the right
		buddies[2*i + 1] = ( rank - (i+1) + comm_size ) % comm_size; //neighbour i+1 steps to the left
	}
	if (2*i < redundant_copies) //if redundant_copies is odd, the last entry is still missing
		buddies[2*i] = ( rank + (i+1) ) % comm_size;
}

void get_buddy_info(struct ESR_options *esr_opts, int comm_rank, int comm_size)
{
	esr_opts->buddies_that_save_me = (int*) malloc(esr_opts->redundant_copies * sizeof(int));
	if (!(esr_opts->buddies_that_save_me)) failure("Memory allocation failed.");

	int i;
	int limit = esr_opts->redundant_copies/2;
	for (i = 0; i < limit; ++i)
	{
		(esr_opts->buddies_that_save_me)[2*i] = ( comm_rank + (i+1) ) % comm_size; //neighbour i+1 steps to the right
		(esr_opts->buddies_that_save_me)[2*i + 1] = ( comm_rank - (i+1) + comm_size ) % comm_size; //neighbour i+1 steps to the left
	}
	if (2*i < esr_opts->redundant_copies) //if redundant_copies is odd, the last entry is still missing
		(esr_opts->buddies_that_save_me)[2*i] = ( comm_rank + (i+1) ) % comm_size;

	
	//for checkpointing, we also need the information the other way round
	if (esr_opts->reconstruction_strategy == REC_checkpoint_in_memory)
	{
		//printf("Allocating buddies_that_I_save array\n");

		esr_opts->buddies_that_I_save = (int*) malloc(esr_opts->redundant_copies * sizeof(int));
		if (!(esr_opts->buddies_that_I_save)) failure("Memory allocation failed.");

		//just calculating the indices in the opposite direction (starting with left neighbour)
		for (i = 0; i < limit; ++i)
		{
			(esr_opts->buddies_that_I_save)[2*i] = ( comm_rank - (i+1) + comm_size ) % comm_size; //neighbour i+1 steps to the left
			(esr_opts->buddies_that_I_save)[2*i + 1] = ( comm_rank + (i+1) ) % comm_size; //neighbour i+1 steps to the right
		}
		if (2*i < esr_opts->redundant_copies) //if redundant_copies is odd, the last entry is still missing
			(esr_opts->buddies_that_I_save)[2*i] = ( comm_rank - (i+1) + comm_size ) % comm_size;
	}
}
