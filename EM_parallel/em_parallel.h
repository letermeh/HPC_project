extern long N = 10000; // Number of examples
extern long P = 10; // Number of dimensions (<< N)
extern long K = 20; // Number of clusters (<< N)

extern double CONV_CRIT = 0.00005; // Convergence criterion (<< 1)
extern int NTHREADS = 4; // Number of threads

extern char* DATA_FILE = "../dataset/data_n10000_p10_k20_covscale0.20_emax-1.00.csv";
extern char* PI_FILE = "../dataset/pi_n10000_p10_k20_covscale0.20_emax-1.00.csv";
extern char* CHOICES_FILE = "../dataset/choices_n10000_p10_k20_covscale0.20_emax-1.00.csv";

double em_parallel (gsl_vector *data[N], double pi[K], gsl_vector *mu[K], gsl_matrix *sigma[K], double gamma[N][K], int *niter, double *run_time_init, double *run_time_em);

int min_max_data (gsl_vector *data[N], gsl_vector *min_data, gsl_vector *max_data);

double iter_em (gsl_vector *data[N], double pi[K], gsl_vector *mu[K], gsl_matrix *sigma[K], double gamma[N][K], double loglik);

double compute_loglik(gsl_vector *data[N], double pi[K], gsl_vector *mu[K], gsl_matrix *sigma[K]);
