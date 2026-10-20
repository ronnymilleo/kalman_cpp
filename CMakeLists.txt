# =============================================================================
# KalmanCpp
# =============================================================================
# Kalman and Extended Kalman filters in C++
#
# Author: Ronny Milleo
# License: See LICENSE file in project root
# =============================================================================

cmake_minimum_required(VERSION 3.28)

project(KalmanCpp
    VERSION 2.0.0
    DESCRIPTION "Kalman and Extended Kalman filters in C++"
    LANGUAGES CXX
)

# Refuse in-source builds, so a stray `cmake .` never spreads build files among the sources
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Configure in a separate build folder: cmake -S . -B build")
endif()

# =============================================================================
# C++ Standard and Compiler Requirements
# =============================================================================

# The code needs C++17 only: structured bindings, std::optional, std::variant and fold expressions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The project uses no C++ modules; scanning for them only slows the build down and puts compiler specific flags in
# compile_commands.json that clang-tidy cannot read
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# =============================================================================
# Default Build Type
# =============================================================================

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# =============================================================================
# Project Options
# =============================================================================

option(KALMAN_CPP_BUILD_TESTS "Build unit tests" OFF)
option(KALMAN_CPP_BUILD_EXAMPLES "Build example programs" ON)
option(KALMAN_CPP_ENABLE_COVERAGE "Instrument the code for coverage reports" OFF)
option(KALMAN_CPP_ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(KALMAN_CPP_WARNINGS_AS_ERRORS "Treat warnings in the project's own code as errors" OFF)

# =============================================================================
# Compiler Configuration
# =============================================================================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/${BUILD_TYPE_LOWER}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/${BUILD_TYPE_LOWER}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/${BUILD_TYPE_LOWER}/lib")

# Strict warnings, sanitizers and coverage for the project's own targets; vendored code is left alone
function(kalman_cpp_configure_target target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(KALMAN_CPP_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
        if(KALMAN_CPP_ENABLE_SANITIZERS)
            target_compile_options(${target} PRIVATE /fsanitize=address)
        endif()
        return()
    endif()

    target_compile_options(${target} PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wswitch-enum -Wswitch-default
    )
    if(KALMAN_CPP_WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE -Werror)
    endif()
    if(KALMAN_CPP_ENABLE_SANITIZERS)
        target_compile_options(${target} PRIVATE
            -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined
        )
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
    if(KALMAN_CPP_ENABLE_COVERAGE)
        target_compile_options(${target} PRIVATE --coverage)
        target_link_options(${target} PRIVATE --coverage)
    endif()
endfunction()

# =============================================================================
# Formatting Target (clang-format)
# =============================================================================

set(KALMAN_CPP_FORMAT_SOURCES
    ${CMAKE_SOURCE_DIR}/examples/demo_support.cpp
    ${CMAKE_SOURCE_DIR}/examples/demo_support.h
    ${CMAKE_SOURCE_DIR}/examples/extended_kalman_demo.cpp
    ${CMAKE_SOURCE_DIR}/examples/kalman_demo.cpp
    ${CMAKE_SOURCE_DIR}/library/core/result.h
    ${CMAKE_SOURCE_DIR}/library/core/text.h
    ${CMAKE_SOURCE_DIR}/library/filters/extended_kalman_filter.cpp
    ${CMAKE_SOURCE_DIR}/library/filters/extended_kalman_filter.h
    ${CMAKE_SOURCE_DIR}/library/filters/kalman_filter.cpp
    ${CMAKE_SOURCE_DIR}/library/filters/kalman_filter.h
    ${CMAKE_SOURCE_DIR}/library/filters/kalman_math.cpp
    ${CMAKE_SOURCE_DIR}/library/filters/kalman_math.h
    ${CMAKE_SOURCE_DIR}/library/matrix/matrix.cpp
    ${CMAKE_SOURCE_DIR}/library/matrix/matrix.h
    ${CMAKE_SOURCE_DIR}/tests/extended_kalman_filter_tests.cpp
    ${CMAKE_SOURCE_DIR}/tests/kalman_filter_tests.cpp
    ${CMAKE_SOURCE_DIR}/tests/kalman_math_tests.cpp
    ${CMAKE_SOURCE_DIR}/tests/matrix_tests.cpp
    ${CMAKE_SOURCE_DIR}/tests/test_support.cpp
    ${CMAKE_SOURCE_DIR}/tests/test_support.h
)

find_program(CLANG_FORMAT NAMES clang-format)
if(CLANG_FORMAT)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i ${KALMAN_CPP_FORMAT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Formatting source code with clang-format"
        VERBATIM
    )
else()
    message(STATUS "clang-format not found; the format target does nothing")
    add_custom_target(format
        COMMAND ${CMAKE_COMMAND} -E echo "clang-format not found, skipping formatting"
    )
endif()

# =============================================================================
# Dependencies
# =============================================================================

if(KALMAN_CPP_BUILD_TESTS)
    add_subdirectory(vendor)
endif()

# =============================================================================
# Library
# =============================================================================

add_subdirectory(library)

# =============================================================================
# Examples
# =============================================================================

# Enabled before the examples, so that the demonstrations that check themselves can register as tests
if(KALMAN_CPP_BUILD_TESTS)
    enable_testing()
endif()

if(KALMAN_CPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# =============================================================================
# Testing
# =============================================================================

if(KALMAN_CPP_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# =============================================================================
# Post-Build Commands
# =============================================================================

# Lets clangd and clang-tidy find the compile commands from the project root
add_custom_target(copy_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_SOURCE_DIR}/compile_commands.json"
    COMMENT "Copying compile_commands.json to the project root"
    VERBATIM
)

# =============================================================================
# Project Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== KalmanCpp Configuration ===")
message(STATUS "Version:              ${PROJECT_VERSION}")
message(STATUS "Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler:             ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build tests:          ${KALMAN_CPP_BUILD_TESTS}")
message(STATUS "Build examples:       ${KALMAN_CPP_BUILD_EXAMPLES}")
message(STATUS "Sanitizers:           ${KALMAN_CPP_ENABLE_SANITIZERS}")
message(STATUS "Coverage:             ${KALMAN_CPP_ENABLE_COVERAGE}")
message(STATUS "Warnings as errors:   ${KALMAN_CPP_WARNINGS_AS_ERRORS}")
message(STATUS "===============================")
message(STATUS "")
