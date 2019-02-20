#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

int print_matrix (gsl_matrix *m){
  int n1 = m->size1;
  int n2 = m->size2;
  for (int i=0; i<n1; i++){
    for (int j=0; j<n2; j++){
      double res = gsl_matrix_get(m, i, j);
      printf("%lf \t", res);
    }
    printf("\n");
  }
  return 0;
}

int main () {
  gsl_vector* x = gsl_vector_alloc(2);
  gsl_vector* y = gsl_vector_alloc(2);
  /*gsl_vector* mu = gsl_vector_alloc(2);
  gsl_matrix* sigma = gsl_matrix_alloc(2, 2);
  gsl_vector* work = gsl_vector_alloc(2);*/

  gsl_vector_set(x, 0, 1.0);
  gsl_vector_set(x, 1, -1.0);

  gsl_vector_set(y, 0, 2.0);
  gsl_vector_set(y, 1, 3.0);
  /*gsl_vector_set_zero(mu);
  gsl_matrix_set_identity(sigma);

  double* result = malloc(sizeof(double));
  gsl_ran_multivariate_gaussian_pdf(x, mu, sigma, result, work);*/

  gsl_matrix x_matr_view = gsl_matrix_view_vector(x, 2, 1).matrix;
  gsl_matrix y_matr_view = gsl_matrix_view_vector(y, 2, 1).matrix;

  gsl_matrix *x_matr = &x_matr_view;
  gsl_matrix *y_matr = &y_matr_view;

  gsl_matrix *x_outer_y = gsl_matrix_alloc(2, 2);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, x_matr, y_matr, 1.0, x_outer_y);

  print_matrix(x_outer_y);

  gsl_vector_free(x);
  gsl_vector_free(y);
  gsl_matrix_free(x_outer_y);
  /*gsl_vector_free(mu);
  gsl_matrix_free(sigma);
  gsl_vector_free(work);
  free(result);*/
}
