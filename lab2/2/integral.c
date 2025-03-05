#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double func(double x){
    return exp(-x * x);
}

double integrate_omp(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int threadid = omp_get_thread_num();
        int items_per_thread = n / NUM_THREADS;
        int lb = threadid * items_per_thread;
        int ub = (threadid == NUM_THREADS - 1) ? (n - 1) : (lb + items_per_thread - 1);
        double sumloc = 0.0;

        for (int i = lb; i <= ub; i++){
            sumloc += func(a + h * (i + 0.5));
        }
        #pragma omp atomic
        sum += sumloc;
    }
    sum *= h;
    return sum;
}


double run_parallel(){
    double start = omp_get_wtime();
    double res = integrate_omp(func, a, b, nsteps);
    double end = omp_get_wtime();
    return end - start;
}

int main(){

    double tparallel = run_parallel();
    printf("Execution time (threads=%d): %.6f seconds\n", NUM_THREADS, tparallel);
    
    return 0;
}