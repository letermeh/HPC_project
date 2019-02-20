#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>

int main () {
  gsl_vector* x[2];
  for (int i=0; i<2; i++) {
    x[i] = gsl_vector_alloc(2);
    gsl_vector_set(x[i], 0, 2.0*i);
    gsl_vector_set(x[i], 1, -1.0*i);
  }

  double x0 = gsl_vector_get(x[0], 0) + gsl_vector_get(x[1], 0);
  double x1 = gsl_vector_get(x[0], 1) + gsl_vector_get(x[1], 1);

  printf("Test0 = %lf\n", x0);
  printf("Test1 = %lf\n", x1);
  for (int i=0; i<2; i++){
    gsl_vector_free(x[i]);
  }
}
