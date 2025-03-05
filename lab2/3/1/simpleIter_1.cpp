#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

const double ERROR = 1e-6;
const int MAX_ITER = 30000;

std::vector<double> oneIter(const std::vector<std::vector<double>>& A, const std::vector<double>& b, const std::vector<double>& x, double t, int n) {
    std::vector<double> new_x(n, 0.0);

    #pragma omp parallel for num_threads(NUM_THREADS) schedule(guided, 50)
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        new_x[i] = x[i] - t * (sum - b[i]);
    }

    return new_x;
}

std::vector<double> simple_iteration(const std::vector<std::vector<double>>& A, const std::vector<double>& b, double tau, int n) {
    std::vector<double> x(n, 0.0);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        std::vector<double> x_new = oneIter(A, b, x, tau, n);

        double diff_norm = 0.0;
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(guided, 50) reduction(+:diff_norm) 
        for (int i = 0; i < n; i++) {
            diff_norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
        }
        diff_norm = std::sqrt(diff_norm);

        if (diff_norm < ERROR) {
            std::cout << "Сошлось за " << iter + 1 << " итераций.\n";
            return x_new;
        }

        x = std::move(x_new); 
    }

    std::cout << "Не сошлось за " << MAX_ITER << " итераций.\n";
    return x;
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
}