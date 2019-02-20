#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h> // Probability density functions
#include <gsl/gsl_blas.h> // Basic linear algebra
#include <gsl/gsl_rng.h> // Random generators
#include <gsl/gsl_math.h> // Mathematical functions

#include "em_parallel.h"


double em_parallel (gsl_vector *data[N], double pi[K], gsl_vector *mu[K], gsl_matrix *sigma[K], double gamma[N][K], int *niter, double *run_time_init, double *run_time_em)
{
  // Get minimum and maximum values
  gsl_vector *min_data = gsl_vector_alloc(P);
  gsl_vector *max_data = gsl_vector_alloc(P);
  min_max_data(data, min_data, max_data);

  double start_time = omp_get_wtime();

////////////////////////////////////////////////////////////////////////////////
// Initialization of pi: all the mixing proportions are initialized to 1/K /////
////////////////////////////////////////////////////////////////////////////////

  double pi0 = 1.0/K;
  for (long k=0; k<K; k++){
    pi[k] = pi0;
  }

////////////////////////////////////////////////////////////////////////////////
// Initialization of Sigma to a diagonal matrix ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  //// Get initial standard deviations: (max_data - min_data) / 2
  gsl_vector *stdevs = gsl_vector_alloc(P);
  gsl_vector_memcpy(stdevs, max_data);
  gsl_vector_sub(stdevs, min_data);
  gsl_vector_scale(stdevs, 2);

  //// Create diagonal matrix
  gsl_matrix *init_sigma = gsl_matrix_alloc(P, P);
  gsl_matrix_set_all(init_sigma, 0.0);
  gsl_vector diag_sigma = gsl_matrix_diagonal(init_sigma).vector;
  gsl_vector_memcpy(&diag_sigma, stdevs);

  //// Initialize sigma
  for (long k=0; k<K; k++){
    gsl_matrix_memcpy(sigma[k], init_sigma);
  }

////////////////////////////////////////////////////////////////////////////////
// Random initialization of mu with respect to a multivariate Gaussian distribution
////////////////////////////////////////////////////////////////////////////////

  //// Get mean data: (max_data + min_data) / 2
  gsl_vector *mu0 = gsl_vector_alloc(P);
  gsl_vector_memcpy(mu0, max_data);
  gsl_vector_add(mu0, min_data);
  gsl_vector_scale(mu0, 2);

  //// Generate from multivariate Gaussian distribution
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus); // Random generator
  for (int k=0; k<K; k++){
    gsl_ran_multivariate_gaussian(r, mu0, init_sigma, mu[k]); // Initialize the mean vectors
  }

  //////////////////////////////////////////////////////////////////////////////

  // Free memory
  gsl_vector_free(min_data);
  gsl_vector_free(max_data);
  gsl_vector_free(stdevs);
  gsl_matrix_free(init_sigma);
  gsl_vector_free(mu0);
  gsl_rng_free(r);


////////////////////////////////////////////////////////////////////////////////
// Run EM algorithm ////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  // Compute initial log-likelihood
  double loglik = compute_loglik(data, pi, mu, sigma);
  int stop_iterations = 0; // Set to 1 when convergence is achieved

  *run_time_init = omp_get_wtime() - start_time;

  start_time = omp_get_wtime();
  *niter = 0;
  double new_loglik;
  while (stop_iterations == 0) {
    *niter += 1;
    new_loglik = iter_em(data, pi, mu, sigma, gamma, loglik); // One iteration of the EM algorithm
    if (new_loglik - loglik < CONV_CRIT) { // Convergence criterion met
      stop_iterations = 1;
    }
    else {
      loglik = new_loglik;
    }
  }
  *run_time_em = omp_get_wtime() - start_time;

  return new_loglik;

}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

int min_max_data (gsl_vector *data[N], gsl_vector *min_data, gsl_vector *max_data)
/*
This function computes the minimum and maximum values of the dataset for each variable, and store them in min_data[] and max_data[].
*/
{
  int th_id; // Thread ID
  gsl_vector *partial_min_data[NTHREADS];
  gsl_vector *partial_max_data[NTHREADS];

  omp_set_num_threads(NTHREADS);

#pragma omp parallel private(th_id) shared(*partial_min_data, *partial_max_data)
  {
    th_id = omp_get_thread_num();

    // Initialize partial_min_data and partial_max_data
    partial_min_data[th_id] = gsl_vector_alloc(P);
    partial_max_data[th_id] = gsl_vector_alloc(P);
    gsl_vector_memcpy(partial_min_data[th_id], data[0]);
    gsl_vector_memcpy(partial_max_data[th_id], data[0]);

#pragma omp for

    // Find min and max values within the elements of the thread, for each variable (0 <= j < P)
    for (int i=0; i<N; i++){
      for (int j=0; j<P; j++){
        double val = gsl_vector_get(data[i], j);
        if (val < gsl_vector_get(partial_min_data[th_id], j)){ // Update min value
          gsl_vector_set(partial_min_data[th_id], j, val);
        }
        if (val > gsl_vector_get(partial_max_data[th_id], j)){ // Update max value
          gsl_vector_set(partial_max_data[th_id], j, val);
        }
      }
    }

#pragma omp barrier // Synchronization

    // Aggragate values
    if (th_id == 0){
      gsl_vector_memcpy(min_data, partial_min_data[0]);
      gsl_vector_memcpy(max_data, partial_max_data[0]);

      for (int t=1; t<NTHREADS; t++){
        for (int j=0; j<P; j++){
          double val_min = gsl_vector_get(partial_min_data[t], j);
          double val_max = gsl_vector_get(partial_max_data[t], j);
          if (val_min < gsl_vector_get(min_data, j)){ // Update min value
            gsl_vector_set(min_data, j, val_min);
          }
          if (val_max > gsl_vector_get(max_data, j)){ // Update max value
            gsl_vector_set(max_data, j, val_max);
          }
        }
      }
    }

#pragma omp barrier

    // Free memory
    gsl_vector_free(partial_min_data[th_id]);
    gsl_vector_free(partial_max_data[th_id]);
  }
  return 0;
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

double iter_em (gsl_vector *data[N], double pi[K], gsl_vector *mu[K], gsl_matrix *sigma[K], double gamma[N][K], double loglik)
{
  long n; // Index of the example
  long k; // Index of the cluster

  // Partial to be aggregated by the master thread
  double sum_gamma[K][NTHREADS]; // Partial sums of posterior probablities
  gsl_vector *partial_unnormalized_mu[K][NTHREADS]; // Partial sums of (gamma_(n, k) * x_n)
  gsl_matrix *partial_unnormalized_sigma[K][NTHREADS]; // Partial sums of (gamma_(n, k) * (x_n - mu_k) * (x_n - mu_k)^T)
  double partial_loglik[NTHREADS]; // Partial log-likelihood

  int th_id; // Thread ID

  omp_set_num_threads(NTHREADS);

#pragma omp parallel private(th_id, n, k) shared(*data, gamma, pi, *mu, *sigma, sum_gamma, *partial_unnormalized_mu, *partial_unnormalized_sigma, partial_loglik)
  {
    th_id = omp_get_thread_num();

    // Intermediate sums that will be computed during E-step, to be aggregated during M-step
    // Initialization to 0
    for (k = 0; k < K; k++) {
      sum_gamma[k][th_id] = 0.0;
      partial_unnormalized_mu[k][th_id] = gsl_vector_alloc(P);
      gsl_vector_set_zero(partial_unnormalized_mu[k][th_id]);
    }

////////////////////////////////////////////////////////////////////////////////
// E-step //////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#pragma omp for

    for (n = 0; n < N; n++) {
      // Computation of unregularized posterior probabilities
      double regul = 0;
      for (k = 0; k < K; k++) {
        // Get value from Gaussian PDF
        double *result = malloc(sizeof(double)); // Where the result will be stored
        gsl_vector *work = gsl_vector_alloc(P); // Additional workspace of length P
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

        gsl_vector *intermediate_vector_for_mu = gsl_vector_alloc(P); // Gamma_(n, k) * x_n
        gsl_vector_memcpy(intermediate_vector_for_mu, data[n]); // Copy the n-th observation into the new vector intermediate_vector_for_mu
        gsl_vector_scale(intermediate_vector_for_mu, gamma[n][k]); // Multiply vector by gamma_(n, k)
        gsl_vector_add(partial_unnormalized_mu[k][th_id], intermediate_vector_for_mu); // Add gamma_(n, k) * x_n (see expression of mu_k in section M-step)

        // Free memory
        gsl_vector_free(intermediate_vector_for_mu);
      }
    }

////////////////////////////////////////////////////////////////////////////////
// M-step //////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

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
      partial_unnormalized_sigma[k][th_id] = gsl_matrix_alloc(P, P);
      gsl_matrix_set_zero(partial_unnormalized_sigma[k][th_id]);
    }

#pragma omp for

    for (n=0; n<N; n++){
      for (k=0; k<K; k++){

        // Compute x_n - mu_k
        gsl_vector *x_minus_mu = gsl_vector_alloc(P);
        gsl_vector_memcpy(x_minus_mu, data[n]);
        gsl_vector_sub(x_minus_mu, mu[k]);

        // Compute the outer product
        gsl_matrix x_minus_mu_matr0 = gsl_matrix_view_vector(x_minus_mu, P, 1).matrix; // Reshape x_minus_mu into a matrix of shape (P, 1)
        gsl_matrix *x_minus_mu_matr = &x_minus_mu_matr0; // Get pointer
        gsl_matrix *intermediate_matrix_for_sigma = gsl_matrix_alloc(P, P); // Will eventually store gamma_(n, k) * (x_n - mu_k) * (x_n - mu_k)^T
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
      for (k=0; k<K; k++){

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

////////////////////////////////////////////////////////////////////////////////
// Compute log-likelihood //////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  double new_loglik = compute_loglik(data, pi, mu, sigma);

  return new_loglik;
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

double compute_loglik(gsl_vector *data[N], double pi[K], gsl_vector *mu[K], gsl_matrix *sigma[K])
{
  int th_id; // Thread ID
  gsl_vector *partial_loglik[NTHREADS];

  omp_set_num_threads(NTHREADS);

#pragma omp parallel private(th_id, n, k) shared(*data, pi, *mu, *sigma, partial_loglik)
  {
    th_id = omp_get_thread_num();
    partial_loglik[th_id] = 0.0;

#pragma omp for

    for (n = 0; n < N; n++) {
      double res = 0.0;
      for (k = 0; k < K; k++) {

        // Multivariate normal PDF
        double *result = malloc(sizeof(double)); // Where the result will be stored
        gsl_vector *work = gsl_vector_alloc(P); // Additional workspace of length P
        gsl_ran_multivariate_gaussian_pdf(data[n], mu[k], sigma[k], result, work); // Multivariate Gaussian PDF (stored into result)

        // Add pi_k * N(x_n; mu_k, Sigma_k)
        res += pi[k] * *result;

        // Free memory
        free(result);
        gsl_vector_free(work);
      }
      // Compute the logarithm
      res = gsl_log1p(res-1);

      // Add to partial sum
      partial_loglik[th_id] += res;
    }

#pragma omp barrier // Synchronization

    if (th_id == 0){

      // Aggregation over the threads
      double loglik = 0.0;
      for(int t=0; t<NTHREADS; t++){
        new_loglik += partial_loglik[t];
      }
    }
  }

  return loglik;
}
