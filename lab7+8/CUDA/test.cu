#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
namespace opt = boost::program_options;

#include <cuda_runtime.h>
#include <cub/cub.cuh>


// cuda unique_ptr
template<typename T>
using cuda_unique_ptr = std::unique_ptr<T,std::function<void(T*)>>;

// new
template<typename T>
T* cuda_new(size_t size)
{
    T *d_ptr;
    cudaMalloc((void **)&d_ptr, sizeof(T) * size);
    return d_ptr;
}

// delete
template<typename T>
void cuda_delete(T *dev_ptr)
{
    cudaFree(dev_ptr);
}

cudaStream_t* cuda_new_stream()
{
    cudaStream_t* stream = new cudaStream_t;
    cudaStreamCreate(stream);
    return stream;
}

void cuda_delete_stream(cudaStream_t* stream)
{
    cudaStreamDestroy(*stream);
    delete stream;
}

cudaGraph_t* cuda_new_graph()
{
    cudaGraph_t* graph = new cudaGraph_t;
    return graph;
}

void cuda_delete_graph(cudaGraph_t* graph)
{
    cudaGraphDestroy(*graph);
    delete graph;
}

cudaGraphExec_t* cuda_new_graph_exec()
{
    cudaGraphExec_t* graphExec = new cudaGraphExec_t;
    return graphExec;
}

void cuda_delete_graph_exec(cudaGraphExec_t* graphExec)
{
    cudaGraphExecDestroy(*graphExec);
    delete graphExec;
}


#define CHECK(call)                                                             \
    {                                                                           \
        const cudaError_t error = call;                                         \
        if (error != cudaSuccess)                                               \
        {                                                                       \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                            \
        }                                                                       \
    }

// собственно возвращает значение линейной интерполяции
double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    // делаем значение y(щначение клетки)используя формулу линейной интерполяции
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}



void initMatrix(std::unique_ptr<double[]> &arr ,int N){
        
        for (size_t i = 0; i < N*N-1; i++)
        {
            arr[i] = 0;
        }
        


          arr[0] = 10.0;
          arr[N-1] = 20.0;
          arr[(N-1)*N + (N-1)] = 30.0;
          arr[(N-1)*N] = 20.0;
              // инициализируем и потом сразу отправим на девайс
        for (size_t i = 1; i < N-1; i++)
        {
            arr[0*N+i] = linearInterpolation(i,0.0,arr[0],N-1,arr[N-1]);
            arr[i*N+0] = linearInterpolation(i,0.0,arr[0],N-1,arr[(N-1)*N]);
            arr[i*N+(N-1)] = linearInterpolation(i,0.0,arr[N-1],N-1,arr[(N-1)*N + (N-1)]);
            arr[(N-1)*N+i] = linearInterpolation(i,0.0,arr[(N-1)*N],N-1,arr[(N-1)*N + (N-1)]);
        }
}




void saveMatrixToFile(const double* matrix, int N, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // Устанавливаем ширину вывода для каждого элемента
    int fieldWidth = 10; // Ширина поля вывода, можно настроить по вашему усмотрению

    // Записываем матрицу в файл с выравниванием столбцов
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix[i * N + j];
        }
        outputFile << std::endl;
    }

    outputFile.close();
}


void swapMatrices(double* &prevmatrix, double* &curmatrix) {
    double* temp = prevmatrix;
    prevmatrix = curmatrix;
    curmatrix = temp;
    
}





__global__ void computeOneIteration(double *prevmatrix, double *curmatrix, int size){
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // чтобы не изменять границы
    if (!(j == 0 || i == 0 || i >= size-1 || j >= size-1))
        curmatrix[i*size+j]  = 0.25 * (prevmatrix[i*size+j+1] + prevmatrix[i*size+j-1] + prevmatrix[(i-1)*size+j] + prevmatrix[(i+1)*size+j]);
        

}


// вычитание из матрицы, результат сохраняем в матрицу пред. значений
__global__ void matrixSub(double *prevmatrix, double *curmatrix,double *error,int size){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // чтобы не изменять границы
    if (!(j == 0 || i == 0 || i >= size-1 || j >= size-1))
        error[i*size + j] = fabs(curmatrix[i*size+j] - prevmatrix[i*size+j]);
        

}


int main(int argc, char const *argv[])
{
    // парсим аргументы
    opt::options_description desc("опции");
    desc.add_options()
        ("accuracy",opt::value<double>()->default_value(1e-6),"точность")
        ("cellsCount",opt::value<int>()->default_value(256),"размер матрицы")
        ("iterCount",opt::value<int>()->default_value(1000000),"количество операций")
        ("help","помощь")
    ;

    opt::variables_map vm;

    opt::store(opt::parse_command_line(argc, argv, desc), vm);

    opt::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    
    // и это всё было только ради того чтобы спарсить аргументы.......

    int N = vm["cellsCount"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int countIter = vm["iterCount"].as<int>();
   
    
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    
    cuda_unique_ptr<cudaStream_t> stream(cuda_new_stream(),cuda_delete_stream);
    cuda_unique_ptr<cudaGraph_t>graph(cuda_new_graph(),cuda_delete_graph);
    cuda_unique_ptr<cudaGraphExec_t>g_exec(cuda_new_graph_exec(),cuda_delete_graph_exec);
    // cudaStream_t* stream = (stream_ptr.get());
    // double *prevmatrix_GPU  = NULL;
    // double *error_GPU  = NULL;
    // tmp будет буфером для хранения результатов редукции , по блокам и общий
    size_t tmp_size = 0;
    // double *curmatrix_GPU = NULL;
    double* tmp = NULL;

    double error =1.0;
    int iter = 0;

    std::unique_ptr<double[]> A(new double[N*N]);
    std::unique_ptr<double[]> Anew(new double[N*N]);
    std::unique_ptr<double[]> B(new double[N*N]);

    initMatrix(std::ref(A),N);
    initMatrix(std::ref(Anew),N);
    
    double* curmatrix = A.get();
    double* prevmatrix = Anew.get();
    double* error_matrix = B.get();
    // double* error_gpu;
    // CHECK(cudaMalloc(&curmatrix_GPU,sizeof(double)*N*N));
    // CHECK(cudaMalloc(&prevmatrix_GPU,sizeof(double)*N*N));
    // CHECK(cudaMalloc(&error_gpu,sizeof(double)*N*N));
    // CHECK(cudaMalloc(&error_GPU,sizeof(double)*1));
    cuda_unique_ptr<double> curmatrix_GPU_ptr(cuda_new<double>(N*N),cuda_delete<double>);
    cuda_unique_ptr<double> prevmatrix_GPU_ptr(cuda_new<double>(N*N),cuda_delete<double>);
    cuda_unique_ptr<double> error_gpu_ptr(cuda_new<double>(N*N),cuda_delete<double>);
    cuda_unique_ptr<double>error_GPU_ptr(cuda_new<double>(1),cuda_delete<double>);
    // 
    double* curmatrix_GPU = curmatrix_GPU_ptr.get();
    double* prevmatrix_GPU = prevmatrix_GPU_ptr.get();
    double* error_gpu = error_gpu_ptr.get();
    double* error_GPU = error_GPU_ptr.get();


    CHECK(cudaMemcpy(curmatrix_GPU,curmatrix,N*N*sizeof(double),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(prevmatrix_GPU,prevmatrix,N*N*sizeof(double),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(error_gpu,error_matrix,N*N*sizeof(double),cudaMemcpyHostToDevice));
    
    
    cub::DeviceReduce::Max(tmp,tmp_size,prevmatrix_GPU,error_GPU,N*N);

    cuda_unique_ptr<double>tmp_ptr(cuda_new<double>(tmp_size),cuda_delete<double>);
    // CHECK(cudaMalloc(&tmp,tmp_size));
    tmp = tmp_ptr.get();


    dim3 threads_in_block = dim3(32, 32);
    dim3 blocks_in_grid((N + threads_in_block.x - 1) / threads_in_block.x, (N + threads_in_block.y - 1) / threads_in_block.y);



// начало записи вычислительного графа
    cudaStreamBeginCapture(*stream,cudaStreamCaptureModeGlobal);
    
        // 999 + 1 - считаем ошибку через 1000 итераций

    for(size_t i =0 ; i<999;i++){
        
        // cudaDeviceSynchronize();
        computeOneIteration<<<blocks_in_grid, threads_in_block,0,*stream>>>(prevmatrix_GPU,curmatrix_GPU,N);
        swapMatrices(prevmatrix_GPU,curmatrix_GPU);
        // cudaDeviceSynchronize();
        // cudaMemcpy(prevmatrix_GPU,curmatrix_GPU,N*N*sizeof(double),cudaMemcpyDeviceToDevice);

    }

    computeOneIteration<<<blocks_in_grid, threads_in_block,0,*stream>>>(prevmatrix_GPU,curmatrix_GPU,N);
    matrixSub<<<blocks_in_grid, threads_in_block,0,*stream>>>(prevmatrix_GPU,curmatrix_GPU,error_gpu,N);
    


    // cudaDeviceSynchronize();
    cub::DeviceReduce::Max(tmp,tmp_size,error_gpu,error_GPU,N*N,*stream);
    cudaStreamEndCapture(*stream, graph.get());


    // закончили построение выч. графа
    
    
    // получили экземпляр выч.графа
    cudaGraphInstantiate(g_exec.get(), *graph, NULL, NULL, 0);

    auto start = std::chrono::high_resolution_clock::now();
    while(error > accuracy && iter < countIter){
        cudaGraphLaunch(*g_exec,*stream);
        // matrixSub<<<blocks_in_grid, threads_in_block,0,stream>>>(prevmatrix_GPU,curmatrix_GPU,error_gpu,N);
        // cub::DeviceReduce::Max(tmp,tmp_size,error_gpu,error_GPU,N*N,stream);
        // cudaDeviceSynchronize();
        cudaMemcpy(&error,error_GPU,1*sizeof(double),cudaMemcpyDeviceToHost);
        iter+=1000;
        std::cout << "iteration: "<<iter << ' ' <<"error: "<<error << std::endl;

    }
    
    


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
    
    std::cout<<"time: " << time_s<<" error: "<<error << " iterarion: " << iter<<std::endl;
    
    CHECK(cudaMemcpy(prevmatrix,prevmatrix_GPU,sizeof(double)*N*N,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(error_matrix,error_gpu,sizeof(double)*N*N,cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(curmatrix,curmatrix_GPU,sizeof(double)*N*N,cudaMemcpyDeviceToHost));
    if (N <=13){
        
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                /* code */
                std::cout << A[i*N+j] << ' ';
                
            }
            std::cout << std::endl;
        }

        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                /* code */
                std::cout << Anew[i*N+j] << ' ';
                
            }
            std::cout << std::endl;
        }

    }
    saveMatrixToFile(curmatrix, N , "matrix.txt");
    return 0;
}