#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void matrix_vector_product(double *a, double *b, double *c, size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        c[i] = 0.0;
        for (size_t j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void run_serial(size_t n, size_t m) {
    double *a = (double*)malloc(sizeof(*a) * m * n);
    double *b = (double*)malloc(sizeof(*b) * n);
    double *c = (double*)malloc(sizeof(*c) * m);

    if (!a || !b || !c) {
        printf("Error allocating memory!\n");
        free(a); free(b); free(c);
        exit(1);
    }

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            a[i * n + j] = i + j;

    for (size_t j = 0; j < n; j++)
        b[j] = j;

    double start = omp_get_wtime();
    matrix_vector_product(a, b, c, m, n);
    double elapsed_time = omp_get_wtime() - start;

    printf("Elapsed time (serial): %.6f sec.\n", elapsed_time);

    free(a); free(b); free(c);
}

void run_parallel(size_t n, size_t m, int num_threads) {
    double *a = (double*)malloc(sizeof(*a) * m * n);
    double *b = (double*)malloc(sizeof(*b) * n);
    double *c = (double*)malloc(sizeof(*c) * m);

    if (!a || !b || !c) {
        printf("Error allocating memory!\n");
        free(a); free(b); free(c);
        exit(1);
    }

    // Инициализация
    #pragma omp parallel num_threads(num_threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < n; j++)
                a[i * n + j] = i + j;
        }

        for (size_t j = 0; j < n; j++)
            b[j] = j;
    }
    double start = omp_get_wtime();
    matrix_vector_product_omp(a, b, c, m, n);
    double elapsed_time = omp_get_wtime() - start;

    printf("Elapsed time (parallel, %d threads): %.6f sec.\n", num_threads, elapsed_time);

    free(a); free(b); free(c);
}

int main() {
    size_t sizes[] = {20000, 40000};
    int thread_counts[] = {2, 4, 7, 8, 16, 20, 40};

    for (int i = 0; i < 2; i++) {
        size_t size = sizes[i];
        printf("\nRunning tests for M = N = %lu\n", size);
        run_serial(size, size);

        for (int j = 0; j < 7; j++) {
            omp_set_num_threads(thread_counts[j]);
            run_parallel(size, size, thread_counts[j]);
        }
    }

    return 0;
}
