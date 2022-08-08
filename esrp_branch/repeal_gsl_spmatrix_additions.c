#include "repeal_gsl_spmatrix_additions.h"
#include <assert.h>

gsl_spmatrix * gsl_spmatrix_flip_compression(const gsl_spmatrix * const A)
{
    // Algorithm from numerical recipes
    size_t new_order = A->sptype == GSL_SPMATRIX_CRS? GSL_SPMATRIX_CCS: GSL_SPMATRIX_CRS;

    assert(A->sptype != GSL_SPMATRIX_TRIPLET);

    gsl_spmatrix * At = gsl_spmatrix_alloc_nzmax(A->size1, A->size2, A->nz, new_order);

    size_t fiber_len = A->sptype == GSL_SPMATRIX_CRS ?
                       A->size2 : A->size1;
    size_t fiber_num = A->sptype == GSL_SPMATRIX_CRS ?
                       A->size1 : A->size2;

    size_t *count = (size_t*) calloc(fiber_len, sizeof(size_t));

    for(size_t i = 0; i < fiber_num; i++) {
        for(size_t j = A->p[i]; j < A->p[i+1]; j++) {
            count[A->i[j]]++;
        }
    }

    At->p[0] = 0;
    for(size_t j = 0; j < fiber_len; j++) {
        At->p[j+1] = At->p[j] + count[j];
    }

    for(size_t j = 0; j < fiber_len; j++) {
        count[j] = 0.0;
    }

    for(size_t i = 0; i < fiber_num; i++) {
        for(size_t j = A->p[i]; j < A->p[i+1]; j++) {
            size_t k = A->i[j];
            size_t index = At->p[k] + count[k];
            At->i[index] = i;
            At->data[index] = A->data[j];
            count[k]++;
        }
    }

    At->nz = A->nz;
    return At;
}
