cmake_minimum_required(VERSION 3.31)
project(Kalman)

set(CMAKE_CXX_STANDARD 20)

add_executable(Kalman main.cpp
        Kalman.cpp
        Kalman.h
        matrix_math.h
        matrix_math.cpp
        test_matrix.cpp)
