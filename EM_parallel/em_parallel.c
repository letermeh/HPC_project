/*
 **  PROGRAM: Parallelization of EM algorithm
 **
 **  COMPILATION: gcc -fopenmp main.c -lgsl -lgslcblas -lm
 **               The openmp option is needed because of the omp timer.
 **
 */

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h> // Probability density functions
#include <gsl/gsl_blas.h> // Basic linear algebra
#include <gsl/gsl_rng.h> // Random generators
#include <gsl/gsl_math.h> // Mathematical functions

#include <string.h>

static long N = 10000; // Number of examples
static long P = 10; // Number of dimensions (<< N)
static long K = 20; // Number of clusters (<< N)

static double CONV_CRIT = 0.00005; // Convergence criterion (<< 1)
static int NTHREADS = 4; // Number of threads

static char* DATA_FILE = "../dataset/data_n10000_p10_k20_covscale0.20_emax-1.00.csv";
static char* PI_FILE = "../dataset/pi_n10000_p10_k20_covscale0.20_emax-1.00.csv";
static char* CHOICES_FILE = "../dataset/choices_n10000_p10_k20_covscale0.20_emax-1.00.csv";

static long MAX_SIZE = 524288;
static char* DELIM = ";";

double em_parallel (gsl_vector *data[N], double pi[K], gsl_vector *mu[K], gsl_matrix *sigma[K], double gamma[N][K], int *niter, double *run_time_init, double *run_time_em);

int min_max_data (gsl_vector *data[N], gsl_vector *min_data, gsl_vector *max_data);

double iter_em (gsl_vector *data[N], double pi[K], gsl_vector *mu[K], gsl_matrix *sigma[K], double gamma[N][K], double loglik);

double compute_loglik(gsl_vector *data[N], double pi[K], gsl_vector *mu[K], gsl_matrix *sigma[K]);

int main()
{
    long n; // Index of the example
    long k; // Index of the cluster
    long p; // Index of the variable

////////////////////////////////////////////////////////////////////////////////
// Initialize dataset
////////////////////////////////////////////////////////////////////////////////

    gsl_vector *data[N];
    FILE *file = NULL;
    file = fopen(DATA_FILE, "r");
    if (file != NULL) {
        char str[MAX_SIZE];
        n = 0;
        while (fgets(str, MAX_SIZE, file) != NULL) { // Read one line
            // Initialize gsl vector
            data[n] = gsl_vector_alloc(P);

            // Get values
            char* substr;
            double val;
            substr = strtok(str, DELIM); // Get the first value
            sscanf(substr, "%lf", &val); // Convert into double
            gsl_vector_set(data[n], 0, val); // Update gsl vector
            for (p=1; p<P; p++) {
                substr = strtok(NULL, DELIM); // Get value
                sscanf(substr, "%lf", &val); // Convert into double
                gsl_vector_set(data[n], p, val); // Update gsl vector
            }
            n += 1; // Next row
        }
        fclose(file);
    }
    else {
        perror("fopen");
    }
    //printf("%.3f\n", gsl_vector_get(data[5], 1));

////////////////////////////////////////////////////////////////////////////////
// Initialize true mixing proportions
////////////////////////////////////////////////////////////////////////////////

    double true_pi[K];
    file = NULL;
    file = fopen(PI_FILE, "r");
    if (file != NULL) {

        // Read line
        char str[MAX_SIZE];
        fgets(str, MAX_SIZE, file);

        // Get values
        char* substr;
        double val;
        substr = strtok(str, DELIM); // Get the first value
        sscanf(substr, "%lf", &val); // Convert into double
        true_pi[0] = val; // Update array
        for (k=1; k<K; k++) {
            substr = strtok(NULL, DELIM); // Get value
            sscanf(substr, "%lf", &val); // Convert into double
            true_pi[k] = val; // Update array
        }
        n += 1; // Next row
        fclose(file);
    }
    else {
        perror("fopen");
    }
    //printf("%.3f\n", true_pi[1]);

////////////////////////////////////////////////////////////////////////////////
// Initialize true choice vector
////////////////////////////////////////////////////////////////////////////////

    long true_clusters[N];
    file = fopen(CHOICES_FILE, "r");
    if (file != NULL) {

        // Read line
        char str[MAX_SIZE];
        fgets(str, MAX_SIZE, file);

        // Get values
        char* substr;
        long val;
        substr = strtok(str, DELIM); // Get the first value
        sscanf(substr, "%ld", &val); // Convert into long
        true_clusters[0] = val; // Update array
        for (n=1; n<N; n++) {
            substr = strtok(NULL, DELIM); // Get value
            sscanf(substr, "%ld", &val); // Convert into long
            true_clusters[n] = val; // Update array
        }
        fclose(file);
    }
    else {
        perror("fopen");
    }
    //printf("%ld\n", true_clusters[6]);

////////////////////////////////////////////////////////////////////////////////
// Initialize parameters
////////////////////////////////////////////////////////////////////////////////

    double pi[K];
    gsl_vector *mu[K];
    gsl_matrix *sigma[K];
    for (k=0; k<K; k++) {
        mu[k] = gsl_vector_alloc(P);
        sigma[k] = gsl_matrix_alloc(P, P);
    }
    long clusters[N];

////////////////////////////////////////////////////////////////////////////////
// Initialize posterior probabilities
////////////////////////////////////////////////////////////////////////////////

    double gamma[N][K];

    // Initialize pointers to the results
    int *niter = malloc(sizeof(int));
    double *run_time_init = malloc(sizeof(double));
    double *run_time_em = malloc(sizeof(double));

    // Run parallel EM algorithm
    double loglik = em_parallel(data, pi, mu, sigma, gamma, niter, run_time_init, run_time_em);

    printf("EM algorithm with %u threads performed in %.0f seconds and %u iterations.\n", NTHREADS, *run_time_init + *run_time_em, *niter);

////////////////////////////////////////////////////////////////////////////////
// Compute accuracy of the model
////////////////////////////////////////////////////////////////////////////////

    //// Compute clusters (classification)
    for (n=0; n<N; n++) {
        double post_prob = 0; // Maximum posterior probability
        for (k=0; k<K; k++) {
            if (gamma[n][k] > post_prob) {
              post_prob = gamma[n][k];
              clusters[n] = k;
            }
        }
    }
    //// Compute accuracy rate
    double acc;
    for (n=0; n<N; n++) {
        if (clusters[n] == true_clusters[n]) {
            acc += 1;
      }
    }
    acc /= N;
    printf("Classification accuracy rate = %.2f.\n", acc);

////////////////////////////////////////////////////////////////////////////////
// Compute metrics
////////////////////////////////////////////////////////////////////////////////

    // Compute likelihood
    double lik = gsl_expm1(loglik) + 1;
    printf("Final likelihood = %.2f.\n", lik);

    // Compute run time per iteration
    double run_time_iter = *run_time_em / *niter;
    printf("Initialization run time = %.0f. Run time per iteration = %.0f.\n\n", *run_time_init, run_time_iter);

////////////////////////////////////////////////////////////////////////////////
// Free memory
////////////////////////////////////////////////////////////////////////////////

    for (k=0; k<K; k++) {
      gsl_vector_free(mu[k]);
      gsl_matrix_free(sigma[k]);
    }
    free(niter); free(run_time_init); free(run_time_em);

    return 0;
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

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

#pragma omp parallel private(th_id) shared(partial_min_data, partial_max_data)
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
  double one_over_sum_gamma[K]; // One over the aggregated sum
  gsl_vector *partial_unnormalized_mu[K][NTHREADS]; // Partial sums of (gamma_(n, k) * x_n)
  gsl_matrix *partial_unnormalized_sigma[K][NTHREADS]; // Partial sums of (gamma_(n, k) * (x_n - mu_k) * (x_n - mu_k)^T)
  double partial_loglik[NTHREADS]; // Partial log-likelihood

  int th_id; // Thread ID

  omp_set_num_threads(NTHREADS);

#pragma omp parallel private(th_id, n, k) shared(data, gamma, pi, mu, sigma, sum_gamma, partial_unnormalized_mu, partial_unnormalized_sigma, partial_loglik)
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
        one_over_sum_gamma[k] = 1 / sum_gamma_aggreg;

        // Computations of pi and mu
        pi[k] = sum_gamma_aggreg / N;
        gsl_vector_scale(mu[k], one_over_sum_gamma[k]);
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
        gsl_matrix_scale(sigma[k], one_over_sum_gamma[k]);
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
  long n; // Index of the example
  long k; // Index of the cluster

  int th_id; // Thread ID
  double partial_loglik[NTHREADS]; // Partial log-likelihoods
  double loglik; // Aggregated log-likelihood

  omp_set_num_threads(NTHREADS);

#pragma omp parallel private(th_id, n, k) shared(data, pi, mu, sigma, partial_loglik)
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
      for(int t=0; t<NTHREADS; t++){
        loglik += partial_loglik[t];
      }
    }
  }

  return loglik;
}
