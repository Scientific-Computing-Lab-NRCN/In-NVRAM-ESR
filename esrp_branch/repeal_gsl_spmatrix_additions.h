#ifndef REPEAL_GSL_SPMATRIX_ADDITIONS
#define REPEAL_GSL_SPMATRIX_ADDITIONS

#include <stdio.h>
#include <gsl/gsl_spmatrix.h>

//! @ brief Transform the representation of a matrix between CRS and CCS
gsl_spmatrix * gsl_spmatrix_flip_compression(const gsl_spmatrix * const A);

#endif