#!/bin/bash

# Проверяем, передан ли аргумент
if [ "$1" == "float" ]; then
    BUILD_DIR="build_float"
    OPTIONS="-DUSE_FLOAT=ON"
elif [ "$1" == "double" ]; then
    BUILD_DIR="build_double"
    OPTIONS=""
else
    echo "Usage: $0 {float|double}"
    exit 1
fi

mkdir -p $BUILD_DIR && cd $BUILD_DIR

cmake $OPTIONS .. && cmake --build .

cd ..