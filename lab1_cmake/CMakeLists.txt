cmake_minimum_required(VERSION 3.10)
project(FloatOrDouble)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(USE_FLOAT "Use Float, not Double" OFF)

if(USE_FLOAT)
    add_compile_definitions(USE_FLOAT)
endif()

# Добавление исполняемого файла
add_executable(SinusArray main.cpp)