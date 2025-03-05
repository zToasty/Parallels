CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -fopenmp

TARGET = build/simple_iteration_1

# Исходные файлы
SRCS = simpleIter_1.cpp 
OBJS = $(SRCS:%.cpp=build/%.o)

# Количество потоков (по умолчанию 8)
NUM_THREADS ?= 8

$(shell mkdir -p build)

# Сборка исполняемого файла
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -DNUM_THREADS=$(NUM_THREADS) -o $(TARGET) $(OBJS)

# Компиляция объектных файлов
build/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -DNUM_THREADS=$(NUM_THREADS) -c $< -o $@

# Очистка
clean:
	rm -rf build/*.o $(TARGET)