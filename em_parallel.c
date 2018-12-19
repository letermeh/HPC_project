/*
 **  PROGRAM: Parallelization of EM algorithm
 **
 **  COMPILATION: gcc -fopenmp em_parallel.c -o em_parallel
 **               The openmp option is needed because of the omp timer.
 **
 */

#include <stdio.h>
#include <omp.h>

static long N = 1000000; // Number of examples
static long p = 10; // Number of dimensions (<< N)
static long K = 100; // Number of clusters (<< N)

static double conv_crit = 0.01; // Convergence criterion (<< 1)
static int nthreads = 4; // Number of threads

int main ()
{
  double dataset[N][d]; // Dataset of N examples and d dimensions

  double mu[K][d]; // Array of K mean vectors
  double sigma[K][d][d]; // Array of K covariance matrices


  double start_time, run_time;

  omp_set_num_threads(nthreads);

  start_time = omp_get_wtime();

  int th_id;

#pragma omp parallel private() shared()
  {
    th_id = omp_get_thread_num();

  }

  run_time = omp_get_wtime() - start_time;

  return 0;
}
