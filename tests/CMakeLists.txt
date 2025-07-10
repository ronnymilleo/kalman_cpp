cmake_minimum_required(VERSION 3.14)
project(kalman_tests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0 # Or specify a different tag/commit
)

FetchContent_MakeAvailable(googletest)

# Test executable
add_executable(test_matrix
        test_matrix.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../matrix_math.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../matrix_math.h
)

# Link Google Test to the test executable
target_link_libraries(test_matrix
        PRIVATE
        GTest::gtest
        GTest::gtest_main
)

# Include directories for the test
target_include_directories(test_matrix
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}../
)

include(GoogleTest)
gtest_discover_tests(test_matrix)

