# Имя исполняемого файла
TARGET = jacobi_cuda

# Компилятор CUDA
NVCC = nvcc

# Исходные файлы .cu (если есть .cpp файлы, их можно добавить в CXX_SOURCES)
CU_SOURCES = main.cu

# Флаги для компилятора CUDA
# -O3: Уровень оптимизации
# -arch=compute_70 -code=sm_70: Генерация PTX и SASS для Tesla V100 (Volta)
# -std=c++17: Стандарт C++ (убедитесь, что ваш код ему соответствует)
# --ptxas-options=-v: Показать информацию от PTX ассемблера (полезно для анализа регистров/памяти)
# --compiler-options "-Wall": Передать флаг -Wall хост-компилятору для вывода всех предупреждений
NVCCFLAGS = -O3 \
            -arch=compute_70 \
            -code=sm_70 \
            -std=c++17 \
            --ptxas-options=-v \
            --compiler-options "-Wall"
            # --use-fast-math


# Флаги для линковщика
# -lboost_program_options: Линковка с библиотекой Boost program_options
LDFLAGS = -lboost_program_options

# Объектные файлы
CU_OBJS = $(CU_SOURCES:.cu=.o)
OBJS = $(CU_OBJS)


# Цель по умолчанию
all: $(TARGET)

# Правило для сборки .cu файлов в .o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# Правило для линковки
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@ $(LDFLAGS)

# Цель для запуска (предполагается, что параметры передаются через командную строку)
run: $(TARGET)
	./$(TARGET) --size=512 --accuracy=1e-6 --iterations=1000000

# Цель для очистки
clean:
	rm -f $(TARGET) $(OBJS) *.o

# Объявление целей, которые не являются файлами
.PHONY: all clean run
