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
    char str[5];
    str = "";
    printf("%s\n", str);
    return 0;
}
