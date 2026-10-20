# =============================================================================
# Unit Tests
# =============================================================================
# Each executable holds the tests of one module and is registered with CTest.
# =============================================================================

include(GoogleTest)

function(kalman_cpp_add_test name)
    add_executable(${name}
        ${name}.cpp
        test_support.cpp
        test_support.h
    )
    target_link_libraries(${name} PRIVATE kalman-cpp-core-lib GTest::gtest_main)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    kalman_cpp_configure_target(${name})
    gtest_discover_tests(${name}
        PROPERTIES
            LABELS "unit"
            TIMEOUT 30
    )
endfunction()

kalman_cpp_add_test(extended_kalman_filter_tests)
kalman_cpp_add_test(kalman_filter_tests)
kalman_cpp_add_test(kalman_math_tests)
kalman_cpp_add_test(matrix_tests)
