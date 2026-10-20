[![CI](https://github.com/ronnymilleo/kalman_cpp/actions/workflows/cmake-multi-platform.yml/badge.svg?branch=main)](https://github.com/ronnymilleo/kalman_cpp/actions/workflows/cmake-multi-platform.yml)

# KalmanCpp

Kalman and Extended Kalman filters in C++23, for real-time state estimation.

## Features

- **Kalman filter** for linear systems and **Extended Kalman filter** for non-linear ones, with optional control input.
- **No exceptions**: operations that can fail return `std::expected<T, std::string>` with a message that says what was
  wrong, and a filter is left unchanged when an operation fails.
- **`Core::Matrix`**: a dense row-major matrix with the arithmetic the filters need (`+`, `-`, `*`, transpose and
  inverse), so the filter equations read like the textbook.
- **Unit tests** with GoogleTest, and two demonstration programs.

## Project structure

```
├── library/
│   ├── matrix/               # Core::Matrix
│   └── filters/              # KalmanFilter, ExtendedKalmanFilter and the equations they share
├── examples/                 # kalman_demo and extended_kalman_demo
├── tests/                    # One <module>_tests.cpp per module
├── vendor/googletest         # GoogleTest (git submodule)
└── CMakeLists.txt
```

## Building

You need CMake 3.28 or newer, Ninja and a compiler with C++23 support for `std::expected` and `std::format` (GCC 13+,
Clang 17+, a current Visual Studio 2022).

```bash
# GoogleTest is only needed for the tests
git submodule update --init vendor/googletest

cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The presets are `debug`, `release`, `clang-debug` and `clang-release`, and build into `build/<preset>`. Binaries and
libraries end up in `out/<build type>/bin` and `out/<build type>/lib`.

Without presets:

```bash
cmake -S . -B build -G Ninja -DKALMAN_CPP_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

| Option                         | Default | Purpose                                            |
|--------------------------------|---------|----------------------------------------------------|
| `KALMAN_CPP_BUILD_TESTS`       | `OFF`   | Build the unit tests                               |
| `KALMAN_CPP_BUILD_EXAMPLES`    | `ON`    | Build the demonstration programs                   |
| `KALMAN_CPP_WARNINGS_AS_ERRORS`| `OFF`   | Make warnings in the project's code fail the build |
| `KALMAN_CPP_ENABLE_SANITIZERS` | `OFF`   | AddressSanitizer and UndefinedBehaviorSanitizer    |
| `KALMAN_CPP_ENABLE_COVERAGE`   | `OFF`   | Instrument the code for `gcov`/`llvm-cov`          |

Building also runs `clang-format` over the sources when it is installed, through the `format` target.

To use the library in another CMake project, add this folder with `add_subdirectory` and link `kalman-cpp-core-lib`.
Headers are included relative to `library/`, such as `#include "filters/kalman_filter.h"`.

## Usage

### Linear filter

```cpp
#include "filters/kalman_filter.h"

// Track position and velocity from position measurements
constexpr double TimeStep = 0.1;

std::expected<Core::KalmanFilter, std::string> created = Core::KalmanFilter::Create(2, 1);
if (!created) {
    return created.error();
}
Core::KalmanFilter &filter = *created;

// Each setter returns a message when it fails; and_then stops at the first one
const std::expected<void, std::string> configured =
    filter.InitializeState({0.0, 0.0})
        .and_then([&] { return filter.SetStateTransitionMatrix({{1.0, TimeStep}, {0.0, 1.0}}); })
        .and_then([&] { return filter.SetObservationMatrix({{1.0, 0.0}}); })
        .and_then([&] { return filter.SetProcessNoiseCovariance({{0.01, 0.0}, {0.0, 0.01}}); })
        .and_then([&] { return filter.SetMeasurementNoiseCovariance({{0.25}}); })
        .and_then([&] { return filter.SetErrorCovariance({{1.0, 0.0}, {0.0, 1.0}}); }); // ready once P is set
if (!configured) {
    return configured.error();
}

if (filter.Predict() && filter.Update({1.2})) {
    const double position = filter.GetState()[0];
    const double velocity = filter.GetState()[1];
}
```

### Extended filter

The model is given as functions, together with their Jacobians:

```cpp
#include "filters/extended_kalman_filter.h"

Core::ExtendedKalmanFilter filter = Core::ExtendedKalmanFilter::Create(2, 1).value();

filter.SetStateTransitionFunction([](const Core::Vector &state, const Core::Vector & /*control*/) {
    return Core::Vector{state[0] + state[1], state[1]};
}).value();
filter.SetStateJacobianFunction([](const Core::Vector & /*state*/) {
    return Core::Matrix{{1.0, 1.0}, {0.0, 1.0}};
}).value();
filter.SetObservationFunction([](const Core::Vector &state) {
    return Core::Vector{std::hypot(state[0], state[1])};
}).value();
filter.SetObservationJacobianFunction([](const Core::Vector &state) {
    const double range = std::hypot(state[0], state[1]);
    return Core::Matrix{{state[0] / range, state[1] / range}};
}).value();
```

`.value()` is shorthand here; check the results as in the example above in real code. `Predict()` and `Update()` fail
with a message until the four functions and the error covariance are set.

## Mathematical background

### Prediction

- **Linear**: `x = F x + B u`
- **Extended**: `x = f(x, u)`, with `F` the Jacobian of `f` evaluated at the estimate the step starts from
- **Covariance**: `P = F P Fᵀ + Q`

### Update

- **Innovation**: `y = z - H x` (linear) or `y = z - h(x)` (extended, with `H` the Jacobian of `h`)
- **Innovation covariance**: `S = H P Hᵀ + R`
- **Kalman gain**: `K = P Hᵀ S⁻¹`
- **State**: `x = x + K y`
- **Covariance**: `P = (I - K H) P`

## License

This project is licensed under the GNU GPLv3. See the [LICENSE](LICENSE) file for details.
