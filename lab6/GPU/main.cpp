#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <openacc.h>

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
            outputFile << std::setw(10) << std::fixed << std::setprecision(4)
                       << matrix[i * size + j];
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
        ("gpu", opt::value<int>()->default_value(0), "Номер GPU (0-3)")
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
    const int gpuId = vm["gpu"].as<int>();

    // Установка GPU
    acc_set_device_num(gpuId, acc_device_nvidia);
    std::cout << "Используется GPU: " << gpuId << "\n";

    auto currentMatrix = std::make_unique<double[]>(matrixSize * matrixSize);
    auto newMatrix = std::make_unique<double[]>(matrixSize * matrixSize);

    initializeMatrix(currentMatrix.get(), matrixSize);
    initializeMatrix(newMatrix.get(), matrixSize);

    double* previousMatrix = currentMatrix.get();
    double* updatedMatrix = newMatrix.get();

    double error = 1.0;
    int iteration = 0;

    std::ostringstream logBuffer;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma acc data copyin(previousMatrix[0:matrixSize * matrixSize]) \
                     copy(updatedMatrix[0:matrixSize * matrixSize]) \
                     copy(error)
    {
        while (iteration < maxIterations && error > accuracy) {
            #pragma acc parallel loop collapse(2) present(previousMatrix, updatedMatrix)
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

            if ((iteration + 1) % 10000 == 0 || iteration == 0) {
                double local_error = 0.0;
                #pragma acc parallel loop collapse(2) reduction(max:local_error) present(previousMatrix, updatedMatrix) 
                for (int i = 1; i < matrixSize - 1; ++i) {
                    for (int j = 1; j < matrixSize - 1; ++j) {
                        double diff = std::fabs(updatedMatrix[i * matrixSize + j] -
                                                previousMatrix[i * matrixSize + j]);
                        if (diff > local_error) local_error = diff;
                    }
                }
                error = local_error;
                logBuffer << "Итерация: " << iteration + 1 << ", Ошибка: " << error << '\n';
            }

            // swap
            double* temp = updatedMatrix;
            updatedMatrix = previousMatrix;
            previousMatrix = temp;

            ++iteration;
        }

        #pragma acc update self(previousMatrix[0:matrixSize * matrixSize])
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << logBuffer.str();
    std::cout << "Завершено за " << iteration << " итераций. Итоговая ошибка: " << error
              << ". Время: " << elapsedTime << " мс.\n";

    saveMatrixToFile(previousMatrix, matrixSize, "matrix.txt");

    return 0;
}
