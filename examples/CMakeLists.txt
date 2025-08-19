# =============================================================================
# Examples
# =============================================================================

add_executable(kalman_demo
    kalman_demo.cpp
)

target_include_directories(kalman_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/library/include
)

target_link_libraries(kalman_demo
    PRIVATE
        kalman
)

add_executable(extended_kalman_demo
    extended_kalman_demo.cpp
)

target_include_directories(extended_kalman_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/library/include
)

target_link_libraries(extended_kalman_demo
    PRIVATE
        extended_kalman
)
