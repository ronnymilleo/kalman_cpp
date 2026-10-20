# =============================================================================
# Core Library
# =============================================================================

add_library(kalman-cpp-core-lib STATIC
    filters/extended_kalman_filter.cpp
    filters/extended_kalman_filter.h
    filters/kalman_filter.cpp
    filters/kalman_filter.h
    filters/kalman_math.cpp
    filters/kalman_math.h
    matrix/matrix.cpp
    matrix/matrix.h
)

# Headers are included relative to this folder, such as "matrix/matrix.h"
target_include_directories(kalman-cpp-core-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

kalman_cpp_configure_target(kalman-cpp-core-lib)

# Formatting runs before every build, so unformatted code never reaches a commit
add_dependencies(kalman-cpp-core-lib format)
