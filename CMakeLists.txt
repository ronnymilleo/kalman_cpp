cmake_minimum_required(VERSION 3.14)
project(Kalman)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(tests)

# Main Kalman executable
add_executable(Kalman
    main.cpp
    kalman.cpp
    matrix_math.cpp
)

target_include_directories(Kalman PRIVATE
    ${CMAKE_SOURCE_DIR}
)