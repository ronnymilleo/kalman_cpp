cmake_minimum_required(VERSION 3.14)
project(Kalman)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add Google Test
add_subdirectory(vendor/googletest)

# Main Kalman executable
add_executable(Kalman
    main.cpp
    Kalman.cpp
    Kalman.h
    matrix_math.cpp
    matrix_math.h
)

# Test executable
add_executable(test_matrix
    test_matrix.cpp
    matrix_math.cpp
    matrix_math.h
)

# Link Google Test to the test executable
target_link_libraries(test_matrix
    PRIVATE
    gtest_main
    gtest
)

# Include directories for the test
target_include_directories(test_matrix
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Enable testing
enable_testing()

# Add test to CTest
add_test(NAME MatrixTests COMMAND test_matrix)
