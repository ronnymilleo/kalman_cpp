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
# Test Executables
# =============================================================================

# Matrix operations test suite
add_executable(test_matrix
    test_matrix.cpp
)

# Link test executable with the main library and Google Test
target_link_libraries(test_matrix
    PRIVATE
        kalman          # Link with main library target
        GTest::gtest
        GTest::gtest_main
)

# Configure include directories for tests
target_include_directories(test_matrix
    PRIVATE
        ${CMAKE_SOURCE_DIR}/library/include
)

# Standard Kalman Filter test suite
add_executable(test_kalman
    test_kalman.cpp
)

# Link test executable with the main library and Google Test
target_link_libraries(test_kalman
    PRIVATE
        kalman          # Link with main library target
        GTest::gtest
        GTest::gtest_main
)

# Configure include directories for tests
target_include_directories(test_kalman
    PRIVATE
        ${CMAKE_SOURCE_DIR}/library/include
)

# Extended Kalman Filter test suite
add_executable(test_extended_kalman
    test_extended_kalman.cpp
)

# Link test executable with the main library and Google Test
target_link_libraries(test_extended_kalman
    PRIVATE
        extended_kalman          # Link with main library target
        GTest::gtest
        GTest::gtest_main
)

# Configure include directories for tests
target_include_directories(test_extended_kalman
    PRIVATE
        ${CMAKE_SOURCE_DIR}/library/include
)

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

# Discover Standard Kalman Filter tests
gtest_discover_tests(test_kalman
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES
        LABELS "unit"
        TIMEOUT 30
)

# Discover Extended Kalman Filter tests
gtest_discover_tests(test_extended_kalman
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES
        LABELS "unit"
        TIMEOUT 30
)

gtest_discover_tests(test_extended_kalman
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
    DEPENDS test_matrix test_kalman test_extended_kalman
    COMMENT "Running all unit tests with verbose output"
    VERBATIM
)

# Create a custom target for running tests with memory checking (if available)
find_program(VALGRIND_PROGRAM valgrind)
if(VALGRIND_PROGRAM)
    add_custom_target(run_tests_memcheck
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose --output-on-failure 
                -T memcheck
        DEPENDS test_matrix test_kalman test_extended_kalman
        COMMENT "Running tests with memory checking using Valgrind"
        VERBATIM
    )
    message(STATUS "Memory checking target 'run_tests_memcheck' available")
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
