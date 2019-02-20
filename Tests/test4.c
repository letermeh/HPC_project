#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h> // Mathematical functions

#include <string.h>

static char* DATA_FILE = "../dataset/data_n10000_p10_k20_covscale0.20_emax-1.00.csv";
static long P = 10;
static int TAILLE_MAX = 1024;

int main () {

  // Create template string
  char str[1024];

  // Initialize dataset
  FILE *file = NULL;
  file = fopen(DATA_FILE, "r");
  if (file != NULL) {
      fgets(str, TAILLE_MAX, file);
      fclose(file);
  }
  else {
      perror("fopen");
  }
  printf("%s\n", str);
  char* delim = ";";
  char* ptr = strtok(str, delim);
  printf("%s\n", ptr);
  ptr = strtok(NULL, delim);
  printf("%s\n", ptr);
  ptr = strtok(NULL, delim);
  double d[2];
  sscanf(ptr, "%lf", &d[0]);
  printf("%.5f\n", d[0]);
}
