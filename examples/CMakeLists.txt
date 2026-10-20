# =============================================================================
# Examples
# =============================================================================

add_library(kalman-cpp-demo-support-lib STATIC
    demo_support.cpp
    demo_support.h
)

target_include_directories(kalman-cpp-demo-support-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kalman-cpp-demo-support-lib PUBLIC kalman-cpp-core-lib)

kalman_cpp_configure_target(kalman-cpp-demo-support-lib)

# The demonstrations compare the filter with the truth they simulate and exit with an error when it does not beat the
# raw measurements, so running them is a test
function(kalman_cpp_add_demo name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE kalman-cpp-core-lib kalman-cpp-demo-support-lib)
    kalman_cpp_configure_target(${name})
    if(KALMAN_CPP_BUILD_TESTS)
        add_test(NAME ${name} COMMAND ${name})
        set_tests_properties(${name} PROPERTIES LABELS "demo" TIMEOUT 30)
    endif()
endfunction()

kalman_cpp_add_demo(extended_kalman_demo)
kalman_cpp_add_demo(kalman_demo)
