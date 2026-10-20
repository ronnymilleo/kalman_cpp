# =============================================================================
# Third-party dependencies
# =============================================================================

# Keep GoogleTest from installing itself, building GoogleMock, or linking a different C runtime than the project on
# Windows
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# SYSTEM keeps its headers out of the warnings and clang-tidy checks, which are for our own code
add_subdirectory(googletest SYSTEM)
