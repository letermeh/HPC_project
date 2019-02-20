#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h> // Mathematical functions

#include <string.h>

static char* DATA_FILE = "../dataset/data_n10000_p10_k20_covscale0.20_emax-1.00.csv";
static long P = 10;

int main () {

  // Creat template string
  char temp_str[1024];
  strcpy(temp_str, "");
  for (long p=0; p<P; p++) {
      strcat(temp_str, "%.16f;");
  }
  double x[P];

  // Initialize dataset
  FILE *file = NULL;
  file = fopen(DATA_FILE, "r");
  if (file != NULL) {
      fscanf(file, temp_str, x);
      fclose(file);
  }
  else {
      perror("fopen");
  }
  printf("%.5f\n", x[5]);
}
