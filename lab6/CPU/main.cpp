#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>

namespace opt = boost::program_options;

double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}

void saveMatrixToFile(const double* matrix, int size, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Не удалось открыть файл " << filename << " для записи.\n";
        return;
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            outputFile << std::setw(10) << std::fixed << std::setprecision(4) << matrix[i * size + j];
        }
        outputFile << '\n';
    }
    outputFile.close();
}

void initializeMatrix(double* matrix, int size) {
    matrix[0] = 10.0;
    matrix[size - 1] = 20.0;
    matrix[(size - 1) * size + (size - 1)] = 30.0;
    matrix[(size - 1) * size] = 20.0;

    for (int i = 1; i < size - 1; ++i) {
        matrix[i] = linearInterpolation(i, 0, 10, size - 1, 20);
        matrix[i * size] = linearInterpolation(i, 0, 10, size - 1, 20);
        matrix[i * size + size - 1] = linearInterpolation(i, 0, 20, size - 1, 30);
        matrix[(size - 1) * size + i] = linearInterpolation(i, 0, 20, size - 1, 30);
    }

    for (int i = 1; i < size - 1; ++i) {
        for (int j = 1; j < size - 1; ++j) {
            matrix[i * size + j] = 0.0;
        }
    }
}

int main(int argc, char* argv[]) {
    opt::options_description desc("Параметры");
    desc.add_options()
        ("accuracy", opt::value<double>()->default_value(1e-6), "Точность")
        ("size", opt::value<int>()->default_value(1024), "Размер сетки")
        ("iterations", opt::value<int>()->default_value(1000000), "Макс. итераций")
        ("help", "Показать справку");

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    const int matrixSize = vm["size"].as<int>();
    const double accuracy = vm["accuracy"].as<double>();
    const int maxIterations = vm["iterations"].as<int>();

    auto currentMatrix = std::make_unique<double[]>(matrixSize * matrixSize);
    auto newMatrix = std::make_unique<double[]>(matrixSize * matrixSize);

    initializeMatrix(currentMatrix.get(), matrixSize);
    initializeMatrix(newMatrix.get(), matrixSize);

    double* previousMatrix = currentMatrix.get();
    double* updatedMatrix = newMatrix.get();

    double error = 1.0;
    int iteration = 0;

    auto start = std::chrono::high_resolution_clock::now();

    while (iteration < maxIterations && error > accuracy) {
        #pragma acc parallel loop collapse(2) vector vector_length(16) gang num_gangs(40) 
        for (int i = 1; i < matrixSize - 1; ++i) {
            for (int j = 1; j < matrixSize - 1; ++j) {
                updatedMatrix[i * matrixSize + j] = 0.25 * (
                    previousMatrix[(i - 1) * matrixSize + j] +
                    previousMatrix[(i + 1) * matrixSize + j] +
                    previousMatrix[i * matrixSize + (j - 1)] +
                    previousMatrix[i * matrixSize + (j + 1)]
                );
            }
        }

        if ((iteration + 1) % 1000 == 0 || iteration == 0) {
            error = 0.0;
            #pragma acc parallel loop collapse(2) vector vector_length(16) gang num_gangs(40)  reduction(max:error) 
            for (int i = 1; i < matrixSize - 1; ++i) {
                for (int j = 1; j < matrixSize - 1; ++j) {
                    double diff = std::fabs(updatedMatrix[i * matrixSize + j] - previousMatrix[i * matrixSize + j]);
                    if (diff > error) error = diff;
                }
            }
            std::cout << "Итерация: " << iteration + 1 << ", Ошибка: " << error << '\n';
        }

        std::swap(previousMatrix, updatedMatrix);
        ++iteration;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Завершено за " << iteration << " итераций. Итоговая ошибка: " << error << ". Время: " << time << " мс.\n";

    saveMatrixToFile(previousMatrix, matrixSize, "matrix.txt");

    return 0;
}
