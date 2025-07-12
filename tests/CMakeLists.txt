# =============================================================================
# Kalman Filter Implementation - Test Suite
# =============================================================================
# Comprehensive unit tests for the Kalman filter implementation using 
# Google Test framework with modern CMake practices.
#
# This test suite validates:
# - Matrix operations (addition, multiplication, transpose, inversion)
# - Kalman filter state estimation accuracy
# - Exception handling and error conditions
# - Memory management and RAII compliance
# =============================================================================

# =============================================================================
# Test Configuration
# =============================================================================

# Test project should not redefine C++ standard if parent project already set it
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# =============================================================================
# Google Test Framework Setup
# =============================================================================

include(FetchContent)

# Configure Google Test with specific version for reproducible builds
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS NAMES GTest
)

# Set Google Test options before making it available
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Make Google Test available
FetchContent_MakeAvailable(googletest)

# =============================================================================
# Test Executables
# =============================================================================

# Matrix operations test suite
add_executable(test_matrix
    test_matrix.cpp
)

# Link test executable with the main library and Google Test
target_link_libraries(test_matrix
    PRIVATE
        kalman_lib          # Link with main library target
        GTest::gtest
        GTest::gtest_main
)

# Configure include directories for tests
target_include_directories(test_matrix
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../  # Access to source headers
)

# Set test executable properties
set_target_properties(test_matrix PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "Tests"
)

# Apply same warning flags as main project if enabled
if(KALMAN_ENABLE_WARNINGS AND DEFINED KALMAN_WARNING_FLAGS)
    target_compile_options(test_matrix PRIVATE ${KALMAN_WARNING_FLAGS})
endif()

# =============================================================================
# Test Discovery and Registration
# =============================================================================

# Include Google Test's CMake integration
include(GoogleTest)

# Automatically discover and register tests
gtest_discover_tests(test_matrix
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES
        LABELS "unit"
        TIMEOUT 30
)

# =============================================================================
# Custom Test Targets
# =============================================================================

# Create a custom target to run all tests with verbose output
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --output-on-failure
    DEPENDS test_matrix
    COMMENT "Running all unit tests with verbose output"
    VERBATIM
)

# Create a custom target for running tests with memory checking (if available)
find_program(VALGRIND_PROGRAM valgrind)
if(VALGRIND_PROGRAM)
    add_custom_target(run_tests_memcheck
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose --output-on-failure 
                -T memcheck
        DEPENDS test_matrix
        COMMENT "Running tests with memory checking using Valgrind"
        VERBATIM
    )
    message(STATUS "Memory checking target 'run_tests_memcheck' available")
endif()

# =============================================================================
# Test Coverage Configuration (Optional)
# =============================================================================

option(KALMAN_ENABLE_COVERAGE "Enable code coverage analysis" OFF)

if(KALMAN_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Add coverage flags
        target_compile_options(test_matrix PRIVATE --coverage -fprofile-arcs -ftest-coverage)
        target_link_libraries(test_matrix PRIVATE --coverage)
        
        # Find gcov or llvm-cov
        find_program(GCOV_PROGRAM gcov)
        find_program(LLVM_COV_PROGRAM llvm-cov)
        
        if(GCOV_PROGRAM OR LLVM_COV_PROGRAM)
            add_custom_target(coverage
                COMMAND ${CMAKE_CTEST_COMMAND} --quiet
                COMMAND echo "Generating coverage report..."
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                DEPENDS test_matrix
                COMMENT "Generating code coverage report"
                VERBATIM
            )
            message(STATUS "Code coverage target 'coverage' available")
        endif()
    else()
        message(WARNING "Code coverage is only supported with GCC or Clang")
    endif()
endif()

# =============================================================================
# Test Configuration Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== Test Configuration Summary ===")
message(STATUS "Google Test version:  ${googletest_VERSION}")
message(STATUS "Coverage enabled:     ${KALMAN_ENABLE_COVERAGE}")
if(VALGRIND_PROGRAM)
    message(STATUS "Valgrind available:   Yes (${VALGRIND_PROGRAM})")
else()
    message(STATUS "Valgrind available:   No")
endif()
message(STATUS "Test timeout:         30 seconds")
message(STATUS "===================================")
message(STATUS "")

