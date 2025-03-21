CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -pthread

TARGET = build/task2

# Исходные файлы
SRCS = task2.cpp 
OBJS = $(SRCS:%.cpp=build/%.o)


$(shell mkdir -p build)

# Сборка исполняемого файла
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS)  -o $(TARGET) $(OBJS)

# Компиляция объектных файлов
build/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Очистка
clean:
	rm -rf build/*.o $(TARGET)