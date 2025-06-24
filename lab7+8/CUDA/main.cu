#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace opt = boost::program_options;

// cuda unique_ptr
template<typename T>
using cuda_unique_ptr = std::unique_ptr<T,std::function<void(T*)>>;

// Выделение памяти на GPU
template<typename T>
T* cuda_new(size_t size) {
    T *d_ptr;
    cudaMalloc((void **)&d_ptr, sizeof(T) * size);
    return d_ptr;
}

// Освобождение памяти на GPU
template<typename T>
void cuda_delete(T *dev_ptr) {
    cudaFree(dev_ptr);
}

// Функции для корректного удаления CUDA хэндлов (потока, графа, исполняемого графа)
// Эти хэндлы сами по себе являются указателями, поэтому их нужно удалять специальными функциями CUDA API
void cuda_delete_stream_handle(cudaStream_t stream) {
    if (stream) cudaStreamDestroy(stream);
}

void cuda_delete_graph_handle(cudaGraph_t graph) {
    if (graph) cudaGraphDestroy(graph);
}

void cuda_delete_graph_exec_handle(cudaGraphExec_t graphExec) {
    if (graphExec) cudaGraphExecDestroy(graphExec);
}

using cuda_stream_unique_ptr = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cuda_delete_stream_handle)>;
using cuda_graph_unique_ptr = std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, decltype(&cuda_delete_graph_handle)>;
using cuda_graph_exec_unique_ptr = std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, decltype(&cuda_delete_graph_exec_handle)>;

#define CHECK(call) \
    { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("Error: %s:%d, ", __FILE__, __LINE__); \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1); \
        } \
    }

double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}

void initMatrix(std::unique_ptr<double[]> &arr, int N) {
    std::fill(arr.get(), arr.get() + N * N, 0.0);
    if (N > 0) {
        arr[0] = 10.0;
        if (N > 1) {
            arr[N - 1] = 20.0;
            arr[(N - 1) * N + (N - 1)] = 30.0;
            arr[(N - 1) * N] = 20.0;
        } else {
             arr[0] = 10.0;
        }
    }
    if (N > 1) {
        for (int i = 1; i < N - 1; ++i) {
            arr[0 * N + i] = linearInterpolation(i, 0.0, arr[0], N - 1, arr[N - 1]);
            arr[i * N + 0] = linearInterpolation(i, 0.0, arr[0], N - 1, arr[(N - 1) * N]);
            arr[i * N + (N - 1)] = linearInterpolation(i, 0.0, arr[N - 1], N - 1, arr[(N - 1) * N + (N - 1)]);
            arr[(N - 1) * N + i] = linearInterpolation(i, 0.0, arr[(N - 1) * N], N - 1, arr[(N - 1) * N + (N - 1)]);
        }
    }
}

// Ядро для одной итерации метода Якоби
__global__ void computeOneIteration(double *prevmatrix, double *curmatrix, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && j > 0 && i < size - 1 && j < size - 1) {
        curmatrix[i * size + j] = 0.25 * (
            prevmatrix[i * size + j + 1] + prevmatrix[i * size + j - 1] +
            prevmatrix[(i - 1) * size + j] + prevmatrix[(i + 1) * size + j]);
    }
}
// Варпы, Dcopy, что будет если создать 32 потока в сетке, если ииндекс потока <16  то А а если >16 то Б

// Ядро для вычитания матриц и нахождения максимальной ошибки на уровне блока с использованием CUB
template <int THREADS_PER_BLOCK_X, int THREADS_PER_BLOCK_Y>
__global__ void matrixSubAndBlockReduceError(
    const double *matrix_latest_A, 
    const double *matrix_previous_B, 
    double *block_max_errors_output, // Output array for max error of each block
    int size) {

    // Temporary storage for cub::BlockReduce
    typedef cub::BlockReduce<double, THREADS_PER_BLOCK_X, cub::BLOCK_REDUCE_RAKING, THREADS_PER_BLOCK_Y> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    double local_error = 0.0;

    // Calculate error only for interior points
    if (i > 0 && j > 0 && i < size - 1 && j < size - 1) {
        local_error = fabs(matrix_latest_A[i * size + j] - matrix_previous_B[i * size + j]);
    }

    // Reduce errors within the block to find the maximum error for this block
    double block_max_error = BlockReduce(temp_storage).Reduce(local_error, cub::Max());

    // The first thread in each block writes the block's maximum error to the output array
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int flat_block_id = blockIdx.x + blockIdx.y * gridDim.x;
        block_max_errors_output[flat_block_id] = block_max_error;
    }
}


void swapMatrices(double*& a, double*& b) {
    double* tmp = a;
    a = b;
    b = tmp;
}

const int ITERATIONS_PER_GRAPH_BLOCK = 10000; 

// Размеры блоков CUDA
const int BLOCK_DIM_X = 32;
const int BLOCK_DIM_Y = 32;


int main(int argc, char** argv) {
    opt::options_description desc("Options");
    desc.add_options()
        ("accuracy", opt::value<double>()->default_value(1e-6), "Accuracy")
        ("size", opt::value<int>()->default_value(256), "Matrix size")
        ("iterations", opt::value<int>()->default_value(1000000), "Max iteration count")
        ("gpu", opt::value<int>()->default_value(0), "GPU index")
        ("help", "Print help message");

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    int N = vm["size"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int maxIter = vm["iterations"].as<int>();
    int gpuIndex = vm["gpu"].as<int>();

    CHECK(cudaSetDevice(gpuIndex));

    std::unique_ptr<double[]> A_host(new double[N * N]);
    std::unique_ptr<double[]> Anew_host(new double[N * N]);

    initMatrix(A_host, N);    
    initMatrix(Anew_host, N); 

    cuda_unique_ptr<double> curmatrix_GPU_data(cuda_new<double>(N * N), cuda_delete<double>);
    cuda_unique_ptr<double> prevmatrix_GPU_data(cuda_new<double>(N * N), cuda_delete<double>);
    
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y); 
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    int num_total_blocks = blocks.x * blocks.y;

    // Буфер для хранения максимальных ошибок от каждого блока
    cuda_unique_ptr<double> block_max_errors_GPU(cuda_new<double>(num_total_blocks), cuda_delete<double>);
    
    // Буфер для хранения одной итоговой максимальной ошибки после всех редукций
    cuda_unique_ptr<double> maxError_GPU_data(cuda_new<double>(1), cuda_delete<double>); // Stores the final single max error

    // --- Копирование начальных данных с хоста на устройство ---
    CHECK(cudaMemcpy(curmatrix_GPU_data.get(), A_host.get(), sizeof(double) * N * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(prevmatrix_GPU_data.get(), Anew_host.get(), sizeof(double) * N * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(block_max_errors_GPU.get(), 0, sizeof(double) * num_total_blocks)); 

    // Temporary storage for the final DeviceReduce call (reducing block_max_errors_GPU)
    size_t tmp_storage_bytes_device_reduce = 0;
    
    // Первый вызов CUB для получения размера необходимой временной памяти
    cub::DeviceReduce::Max(nullptr, tmp_storage_bytes_device_reduce, block_max_errors_GPU.get(), maxError_GPU_data.get(), num_total_blocks);
    // Выделение временной памяти на GPU
    cuda_unique_ptr<uint8_t> tmp_storage_GPU_device_reduce(cuda_new<uint8_t>(tmp_storage_bytes_device_reduce), cuda_delete<uint8_t>);

    std::ostringstream logBuffer;
    int iter = 0;
    double max_err_host = 1.0;

    cudaStream_t stream_handle = nullptr;
    cudaGraph_t graph_handle = nullptr;
    cudaGraphExec_t graph_exec_handle = nullptr;

    CHECK(cudaStreamCreate(&stream_handle));
    cuda_stream_unique_ptr stream(stream_handle, cuda_delete_stream_handle);
    cuda_graph_unique_ptr graph(nullptr, cuda_delete_graph_handle); 
    cuda_graph_exec_unique_ptr graph_exec(nullptr, cuda_delete_graph_exec_handle);

    // Указатели на данные GPU, которые будут использоваться при записи графа.
    // Их значения (адреса памяти) будут "заморожены" в графе.
    // Локальные swapMatrices будут менять эти C++ указатели, чтобы правильно записать последовательность операций.
    double* graph_capture_prev_ptr = prevmatrix_GPU_data.get();
    double* graph_capture_cur_ptr = curmatrix_GPU_data.get();

    CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal));

    for (int i = 0; i < ITERATIONS_PER_GRAPH_BLOCK; ++i) {
        computeOneIteration<<<blocks, threads, 0, stream.get()>>>(graph_capture_prev_ptr, graph_capture_cur_ptr, N);
        swapMatrices(graph_capture_prev_ptr, graph_capture_cur_ptr);
    }
    
    // Записываем в граф вызов ядра для вычитания матриц и нахождения макс. ошибки поблочно
    matrixSubAndBlockReduceError<BLOCK_DIM_X, BLOCK_DIM_Y><<<blocks, threads, 0, stream.get()>>>(
        graph_capture_prev_ptr, // latest data
        graph_capture_cur_ptr,  // previous data
        block_max_errors_GPU.get(), 
        N);
    
    // Записываем в граф финальную редукцию (DeviceReduce) массива максимальных ошибок блоков
    cub::DeviceReduce::Max(
        tmp_storage_GPU_device_reduce.get(), 
        tmp_storage_bytes_device_reduce, 
        block_max_errors_GPU.get(),     // Вход: массив ошибок блоков
        maxError_GPU_data.get(),        // Выход: одна глобальная максимальная ошибка
        num_total_blocks,               // Количество элементов для редукции
        stream.get());

    cudaGraph_t captured_graph_handle = nullptr; 
    CHECK(cudaStreamEndCapture(stream.get(), &captured_graph_handle));
    graph.reset(captured_graph_handle); 

    cudaGraphExec_t instantiated_graph_exec_handle = nullptr; 
    CHECK(cudaGraphInstantiate(&instantiated_graph_exec_handle, graph.get(), NULL, NULL, 0));
    graph_exec.reset(instantiated_graph_exec_handle); 
    
    auto start_time = std::chrono::high_resolution_clock::now();

    while (iter < maxIter && max_err_host > accuracy) {
        CHECK(cudaGraphLaunch(graph_exec.get(), stream.get()));
        CHECK(cudaStreamSynchronize(stream.get())); 

        CHECK(cudaMemcpy(&max_err_host, maxError_GPU_data.get(), sizeof(double), cudaMemcpyDeviceToHost));
        
        iter += ITERATIONS_PER_GRAPH_BLOCK;
        logBuffer << "Iteration: " << iter << ", Max Error: " << std::fixed << std::setprecision(8) << max_err_host << "\n";
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << logBuffer.str();
    std::cout << "Total execution time: " << duration.count() << " seconds\n";
    std::cout << "Final Iteration: " << iter << ", Final Max Error: " << std::fixed << std::setprecision(8) << max_err_host << "\n";

    double* final_result_device_buffer;
    if (ITERATIONS_PER_GRAPH_BLOCK % 2 != 0) {
        final_result_device_buffer = curmatrix_GPU_data.get();
    } else {
        final_result_device_buffer = prevmatrix_GPU_data.get();
    }
    CHECK(cudaMemcpy(A_host.get(), final_result_device_buffer, sizeof(double) * N * N, cudaMemcpyDeviceToHost));

    std::ofstream fout("matrix.txt");
    if (!fout.is_open()) {
        std::cerr << "Error: Unable to open matrix.txt for writing." << std::endl;
    } else {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                fout << std::setw(10) << std::fixed << std::setprecision(4) << A_host[i * N + j];
            }
            fout << "\n";
        }
        fout.close();
        std::cout << "Result matrix saved to matrix.txt" << std::endl;
    }
    
    return 0;
}