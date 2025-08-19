# =============================================================================
# Main Library Target
# =============================================================================

add_library(matrix_math
    src/matrix_math.cpp
)

target_include_directories(matrix_math
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(kalman
    src/kalman.cpp
)

target_include_directories(kalman
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kalman
    PUBLIC
        matrix_math
)

add_library(extended_kalman
    src/extended_kalman.cpp
)

target_include_directories(extended_kalman
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(extended_kalman
    PUBLIC
        matrix_math
)
