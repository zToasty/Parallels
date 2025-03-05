#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

const double ERROR = 1e-6;
const int MAX_ITER = 30000;

std::vector<double> simple_iteration(const std::vector<std::vector<double>>& A, const std::vector<double>& b, double tau, int n) {
    std::vector<double> x(n, 0.0);
    std::vector<double> new_x(n, 0.0);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        #pragma omp parallel num_threads(NUM_THREADS)
        {
            int nthreads = omp_get_num_threads();
            int threadid = omp_get_thread_num();

            int items_per_thread = n / nthreads;

            int lb = threadid * items_per_thread;
            int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

            for (int i = lb; i <= ub; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    sum += A[i][j] * x[j];
                }
                new_x[i] = x[i] - tau * (sum - b[i]);
            }
        }

        // Проверка на сходимость
        double norm_value = 0.0;
        #pragma omp parallel num_threads(NUM_THREADS)
        {
            int nthreads = omp_get_num_threads();
            int threadid = omp_get_thread_num();

            int items_per_thread = n / nthreads;

            int lb = threadid * items_per_thread;
            int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

            double local_sum = 0.0;
            for (int i = lb; i <= ub; i++) {
                local_sum += pow(new_x[i] - x[i], 2);
            }

            #pragma omp atomic
            norm_value += local_sum;
        }

        norm_value = sqrt(norm_value);
        if (norm_value < ERROR) {
            std::cout << "Сошлось за " << iter + 1 << " итераций.\n";
            return new_x;
        }

        x = new_x;
    }

    std::cout << "Не сошлось за " << MAX_ITER << " итераций.\n";
    return new_x;
}

int main() {
    int n;

    std::cout << "Введите размер матрицы: ";
    std::cin >> n;

    std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));
    std::vector<double> b(n, n + 1);

    for (int i = 0; i < n; i++) {
        A[i][i] = 2.0;
    }

    double tau = 0.001;

    double start_time = omp_get_wtime();
    std::vector<double> x = simple_iteration(A, b, tau, n);
    double end_time = omp_get_wtime();

    std::cout << "Время выполнения: " << (end_time - start_time) << " секунд, на " << NUM_THREADS << " потоках\n";

    std::cout << "Ответ:\n";
    for (double val : x) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
