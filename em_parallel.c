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

static long N = 1000000; // Number of examples
static long p = 10; // Number of dimensions (<< N)
static long K = 100; // Number of clusters (<< N)

static double CONV_CRIT = 0.01; // Convergence criterion (<< 1)
static int NTHREADS = 4; // Number of threads

int main ()
{
  // Input data
  gsl_vector* data[N]; // Dataset of N examples and d dimensions

  // Model parameters
  double pi[K]; // Array of mixing proportions
  gsl_vector* mu[K]; // Array of K mean vectors
  gsl_matrix* sigma[K]; // Array of K covariance matrices

  // Intermediate results (E-step)
  double regul; // Regularization factor used to compute posterior probabilities
  double gamma[N][K]; // Posterior probabilitites of the hidden variable having value k, given x_n

  // Bounds and iterators
  long N_min; // First training example to be considered by a given thread
  long N_max; // Last training example to be considered by a given thread (+1)
  long n; // Index of training example
  long k; // Possible value of hidden variable

  double start_time, run_time;
  int th_id; // Thread ID

  omp_set_num_threads(NTHREADS);
  start_time = omp_get_wtime();

  long N_perthread = N / NTHREADS; // Number of training example per thread (except the last one)

#pragma omp parallel private(th_id, N_min, N_max, n, k) shared(gamma, pi, mu, sigma, regul)
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

    ////////////////////////////////////////////////////////////////////////////
    // E-step //////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    for (n = N_min; n < N_max; n++) {
      // Computation of unregularized posterior probabilities
      regul = 0;
      for (k = 0; k < K; k++) {
        gamma[n][k] = pi[k] * gsl_ran_multivariate_gaussian_pdf(data[n], mu[k], sigma[k]);
        regul += gamma[n][k];
      }

      // Regularization of posterior probabilities
      for (k = 0; k < K; k++) {
        gamma[n][k] = gamma[n][k] / regul;
      }
    }
  }

  run_time = omp_get_wtime() - start_time;

  return 0;
}
