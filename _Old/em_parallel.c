/*
 **  PROGRAM: Parallelization of EM algorithm
 **
 **  COMPILATION: gcc -fopenmp em_parallel.c -o em_parallel
 **               The openmp option is needed because of the omp timer.
 **
 */

#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <omp.h>
#include <gsl/gsl_blas.h>

static long N = 1000000; // Number of examples
static long p = 10; // Number of dimensions (<< N)
static long K = 100; // Number of clusters (<< N)

static double CONV_CRIT = 0.01; // Convergence criterion (<< 1)
static int NTHREADS = 4; // Number of threads

int iter_em (gsl_vector *data[N], double pi[K], gsl_vector *mu[K], gsl_matrix *sigma[K])
{

  int n; // Index of the example
  int k; // Index of the variable

  // Input data
  gsl_vector *data[N]; // Dataset of N examples and d dimensions
  for (n=0; n<N; n++){
    data[n] = gsl_vector_alloc(p);
  }

  // Model parameters
  double pi[K]; // Array of mixing proportions
  gsl_vector *mu[K]; // Array of K mean vectors
  gsl_matrix *sigma[K]; // Array of K covariance matrices

  // Initialization of model parameters
  for (k=0; k<K; k++){
    pi[k] = 0; // TO CHANGE?

    mu[k] = gsl_vector_alloc(p);
    gsl_vector_set_zero(mu[k]); // TO CHANGE?

    sigma[k] = gsl_matrix_alloc(p, p);
    gsl_matrix_set_identity(sigma[k]); // TO CHANGE?
  }

  // Intermediate results (E-step)
  double gamma[N][K]; // Posterior probabilitites of the hidden variable having value k, given x_n

  // Intermediate results to be aggregated during M-step
  double sum_gamma[K][NTHREADS]; // Partial sums of posterior probablities
  gsl_vector *partial_unnormalized_mu[K][NTHREADS]; // Partial sums of (gamma_(n, k) * x_n)
  gsl_matrix *partial_unnormalized_sigma[K][NTHREADS]; // Partial sums of (gamma_(n, k) * (x_n - mu_k) * (x_n - mu_k)^T)

  // Bounds and iterators
  long N_min; // First training example to be considered by a given thread
  long N_max; // Last training example to be considered by a given thread (+1)

  double start_time, run_time;
  int th_id; // Thread ID

  omp_set_num_threads(NTHREADS);
  start_time = omp_get_wtime();

  long N_perthread = N / NTHREADS; // Number of training example per thread (except the last one)

#pragma omp parallel private(th_id, N_min, N_max, n, k) shared(*data, gamma, pi, *mu, *sigma, sum_gamma, *partial_unnormalized_mu, *partial_unnormalized_sigma)
  {
    th_id = omp_get_thread_num();

    // Determine the bounds
    N_min = th_id * N_perthread;
    if (th_id == NTHREADS-1) {
      N_max = N;
    }
    else {
      N_max = N_min + N_perthread;
    }

    // Intermediate sums that will be computed during E-step, to be aggregated during M-step
    // Initialization to 0
    for (k = 0; k < K; k++) {
      sum_gamma[k][th_id] = 0;
      partial_unnormalized_mu[k][th_id] = gsl_vector_alloc(p);
      gsl_vector_set_zero(partial_unnormalized_mu[k][th_id]);
    }

////////////////////////////////////////////////////////////////////////////
// E-step //////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
    for (n = N_min; n < N_max; n++) {
      // Computation of unregularized posterior probabilities
      double regul = 0;
      for (k = 0; k < K; k++) {
        // Get value from Gaussian PDF
        double *result = malloc(sizeof(double)); // Where the result will be stored
        gsl_vector *work = gsl_vector_alloc(p); // Additional workspace of length p
        gsl_ran_multivariate_gaussian_pdf(data[n], mu[k], sigma[k], result, work); // Multivariate Gaussian PDF (stored into result)
        gamma[n][k] = pi[k] * *result;
        regul += gamma[n][k];

        // Free memory
        free(result);
        gsl_vector_free(work);
      }

      for (k = 0; k < K; k++) {
        // Regularization of posterior probabilities
        gamma[n][k] = gamma[n][k] / regul;

        // Computation of intermediate results
        sum_gamma[k][th_id] += gamma[n][k]; // Add gamma_(n, k) (see expression of pi and denominators of mu and Sigma in section M-step)

        gsl_vector *intermediate_vector_for_mu = gsl_vector_alloc(p); // Gamma_(n, k) * x_n
        gsl_vector_memcpy(intermediate_vector_for_mu, data[n]); // Copy the n-th observation into the new vector intermediate_vector_for_mu
        gsl_vector_scale(intermediate_vector_for_mu, gamma[n][k]); // Multiply vector by gamma_(n, k)
        gsl_vector_add(partial_unnormalized_mu[k][th_id], intermediate_vector_for_mu); // Add gamma_(n, k) * x_n (see expression of mu_k in section M-step)

        // Free memory
        gsl_vector_free(intermediate_vector_for_mu);
      }
    }

////////////////////////////////////////////////////////////////////////////
// M-step //////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

#pragma omp barrier // Synchronization

    if (th_id == 0){
      for (k=0; k<K; k++){ // PARALLELIZE THIS TOO?

        // Aggregation of the intermediate sums
        //// Initialization
        double sum_gamma_aggreg = 0;
        gsl_vector_set_zero(mu[k]); // We don't need the values of mu computed at previous iteration anymore

        //// Aggregation over the threads
        for(int t=0; t<NTHREADS; t++){
          sum_gamma_aggreg += sum_gamma[k][t];
          gsl_vector_add(mu[k], partial_unnormalized_mu[k][t]);
        }
        double one_over_sum_gamma = 1 / sum_gamma_aggreg;

        // Computations of pi and mu
        pi[k] = sum_gamma_aggreg / N;
        gsl_vector_scale(mu[k], one_over_sum_gamma);
      }
    }

#pragma omp barrier // Wait for the master thread to finish its computations

    // Intermediate sums that will be aggregated to compute Sigma
    // Initialization to 0
    for (k = 0; k < K; k++) {
      partial_unnormalized_sigma[k][th_id] = gsl_matrix_alloc(p, p);
      gsl_matrix_set_zero(partial_unnormalized_sigma[k][th_id]);
    }

    for (n=N_min; n<N_max; n++){
      for (k=0; k<K; k++){

        // Compute x_n - mu_k
        gsl_vector *x_minus_mu = gsl_vector_alloc(p);
        gsl_vector_memcpy(x_minus_mu, data[n]);
        gsl_vector_sub(x_minus_mu, mu[k]);

        // Compute the outer product
        gsl_matrix x_minus_mu_matr0 = gsl_matrix_view_vector(x_minus_mu, p, 1).matrix; // Reshape x_minus_mu into a matrix of shape (p, 1)
        gsl_matrix *x_minus_mu_matr = &x_minus_mu_matr0; // Get pointer
        gsl_matrix *intermediate_matrix_for_sigma = gsl_matrix_alloc(p, p); // Will eventually store gamma_(n, k) * (x_n - mu_k) * (x_n - mu_k)^T
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, x_minus_mu_matr, x_minus_mu_matr, 1.0, intermediate_matrix_for_sigma); // Compute x_minus_mu_matr * x_minus_mu_matr^T

        // Compute gamma_(n, k) * (x_n - mu_k) * (x_n - mu_k)^T
        gsl_matrix_scale(intermediate_matrix_for_sigma, gamma[n][k]); // Multiply vector by gamma_(n, k)
        gsl_matrix_add(partial_unnormalized_sigma[k][th_id], intermediate_matrix_for_sigma); // Add gamma_(n, k) * (x_n - mu_k) * (x_n - mu_k)^T (see expression of Sigma_k in section M-step)

        // Free memory
        gsl_vector_free(x_minus_mu);
        gsl_matrix_free(intermediate_matrix_for_sigma);
      }
    }

#pragma omp barrier // Synchronization

    if (th_id == 0){
      for (k=0; k<K; k++){ // PARALLELIZE THIS TOO?

        // Aggregation of the intermediate sums
        //// Initialization
        gsl_matrix_set_zero(sigma[k]); // We don't need the values of Sigma computed at previous iteration anymore

        //// Aggregation over the threads
        for(int t=0; t<NTHREADS; t++){
          gsl_matrix_add(sigma[k], partial_unnormalized_sigma[k][t]);
        }

        // Computation of Sigma
        gsl_matrix_scale(sigma[k], one_over_sum_gamma);
      }
    }

#pragma omp barrier // Wait for the master thread to finish its computations

    // Free memory
    for (k = 0; k < K; k++) {
      gsl_vector_free(partial_unnormalized_mu[k][th_id]);
      gsl_matrix_free(partial_unnormalized_sigma[k][th_id]);
    }
  }

  run_time = omp_get_wtime() - start_time;

  return 0;
}
