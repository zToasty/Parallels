#include <iostream>
#include <cmath>

// Глобальная константа для выбора типа данных
#ifdef USE_FLOAT
typedef float DataType;
#else
typedef double DataType;
#endif

int main() {
    const size_t N = 10000000; // Количество элементов
    DataType sum = 0;

    for (size_t i = 0; i < N; ++i) {
        DataType value = std::sin(2 * M_PI * static_cast<DataType>(i) / N);
        sum += value;
    }

    std::cout << "Sum: " << sum << std::endl;

    return 0;
}