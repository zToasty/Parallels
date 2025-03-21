#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>
#include <algorithm>

void matMulVec(const std::vector<double>& matrix, const std::vector<double>& vector, std::vector<double>& result, size_t rows, size_t cols, size_t startRow, size_t endRow) {
    for (size_t i = startRow; i < endRow; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
}

double measureTime(size_t rows, size_t cols, size_t numThreads) {
    std::vector<double> matrix(rows * cols);
    std::vector<double> vector(cols);
    std::vector<double> result(rows, 0.0);

    // Заполняем матрицу
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            matrix[i * cols + j] = static_cast<double>(i + j);
        }
    }

    // Заполняем вектор
    for (size_t j = 0; j < cols; j++) {
        vector[j] = static_cast<double>(j);
    }

    std::vector<std::thread> threads;
    auto start = std::chrono::steady_clock::now();

    size_t actualThreads = std::min(numThreads, rows);
    size_t chunkSize = rows / actualThreads;

    for (size_t i = 0; i < actualThreads; ++i) {
        size_t startRow = i * chunkSize;
        size_t endRow = (i == actualThreads - 1) ? rows : startRow + chunkSize;

        if (startRow < endRow) {
            threads.emplace_back(&matMulVec, std::cref(matrix), std::cref(vector), std::ref(result), rows, cols, startRow, endRow);
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main() {
    const size_t rows1 = 20000;
    const size_t rows2 = 40000;
    std::vector<size_t> threadCounts = {1, 2, 4, 7, 8, 16, 20, 40};

    std::ofstream outFile("results.csv");
    outFile << "Threads,Time_20k,Speedup_20k,Time_40k,Speedup_40k\n";

    std::cout << "Started" << std::endl;

    double seqTime1 = measureTime(rows1, rows1, 1);
    double seqTime2 = measureTime(rows2, rows2, 1);
    outFile << "1," << seqTime1 << ",1," << seqTime2 << ",1\n";

    for (size_t numThreads : threadCounts) {
        double time1 = measureTime(rows1, rows1, numThreads);
        double time2 = measureTime(rows2, rows2, numThreads);
        double speedup1 = (time1 > 0) ? seqTime1 / time1 : 0;
        double speedup2 = (time2 > 0) ? seqTime2 / time2 : 0;

        outFile << numThreads << "," << time1 << "," << speedup1 << "," << time2 << "," << speedup2 << "\n";
    }

    outFile.close();
    std::cout << "Results saved to results.csv" << std::endl;

    return 0;
}
