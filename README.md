# KalmanCpp

[![CI](https://github.com/ronnymilleo/kalman_cpp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ronnymilleo/kalman_cpp/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/ronnymilleo/kalman_cpp/badge.svg?branch=main)](https://coveralls.io/github/ronnymilleo/kalman_cpp?branch=main)
[![CodeFactor](https://www.codefactor.io/repository/github/ronnymilleo/kalman_cpp/badge)](https://www.codefactor.io/repository/github/ronnymilleo/kalman_cpp)

Kalman and Extended Kalman filters in C++17, for real-time state estimation.

## Features

- **Kalman filter** for linear systems and **Extended Kalman filter** for non-linear ones, with optional control input.
- **No exceptions**: operations that can fail return a `Core::Result<T>` (a value, or a message) that says what was
  wrong, and a filter is left unchanged when an operation fails.
- **`Core::Matrix`**: a dense row-major matrix with the arithmetic the filters need (`+`, `-`, `*`, transpose and
  inverse), so the filter equations read like the textbook.
- **Unit tests** with GoogleTest, and two demonstration programs.

## Project structure

```
├── library/
│   ├── core/                 # Core::Result, Core::Concat and the angle helpers
│   ├── matrix/               # Core::Matrix
│   └── filters/              # KalmanFilter, ExtendedKalmanFilter and the equations they share
├── examples/                 # kalman_demo and extended_kalman_demo
├── tests/                    # One <module>_tests.cpp per module
├── vendor/googletest         # GoogleTest (git submodule)
└── CMakeLists.txt
```

## Building

You need CMake 3.28 or newer, Ninja and a C++17 compiler (GCC, Clang or MSVC).

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
| `KALMAN_CPP_ENABLE_COVERAGE`   | `OFF`   | Instrument the library for `gcov` (GCC or Clang)   |

Building also runs `clang-format` over the sources when it is installed, through the `format` target.

To use the library in another CMake project, add this folder with `add_subdirectory` and link `kalman-cpp-core-lib`.
Headers are included relative to `library/`, such as `#include "filters/kalman_filter.h"`.

## Usage

### Linear filter

```cpp
#include "filters/kalman_filter.h"

// Track position and velocity from position measurements
constexpr double TimeStep = 0.1;

Core::Result<Core::KalmanFilter> created = Core::KalmanFilter::Create(2, 1);
if (!created) {
    return created.Error();
}
Core::KalmanFilter &filter = *created;

// Each setter returns a message when it fails; AndThen stops at the first one
const Core::Result<void> configured =
    filter.InitializeState({0.0, 0.0})
        .AndThen([&] { return filter.SetStateTransitionMatrix({{1.0, TimeStep}, {0.0, 1.0}}); })
        .AndThen([&] { return filter.SetObservationMatrix({{1.0, 0.0}}); })
        .AndThen([&] { return filter.SetProcessNoiseCovariance({{0.01, 0.0}, {0.0, 0.01}}); })
        .AndThen([&] { return filter.SetMeasurementNoiseCovariance({{0.25}}); })
        .AndThen([&] { return filter.SetErrorCovariance({{1.0, 0.0}, {0.0, 1.0}}); }); // ready once P is set
if (!configured) {
    return configured.Error();
}

if (filter.Predict() && filter.Update({1.2})) {
    const double position = filter.GetState()[0];
    const double velocity = filter.GetState()[1];
}
```

### Extended filter

The model is given as functions, together with their Jacobians. The setters return a `Core::Result<void>`; check them as
in the example above.

```cpp
#include "filters/extended_kalman_filter.h"

Core::Result<Core::ExtendedKalmanFilter> created = Core::ExtendedKalmanFilter::Create(2, 1);
Core::ExtendedKalmanFilter &filter = *created; // after checking `created`, as above

const Core::Result<void> configured =
    filter
        .SetStateTransitionFunction([](const Core::Vector &state, const Core::Vector & /*control*/) {
            return Core::Vector{state[0] + state[1], state[1]};
        })
        .AndThen([&] {
            return filter.SetStateJacobianFunction(
                [](const Core::Vector & /*state*/) { return Core::Matrix{{1.0, 1.0}, {0.0, 1.0}}; });
        })
        .AndThen([&] {
            return filter.SetObservationFunction(
                [](const Core::Vector &state) { return Core::Vector{std::hypot(state[0], state[1])}; });
        })
        .AndThen([&] {
            return filter.SetObservationJacobianFunction([](const Core::Vector &state) {
                const double range = std::hypot(state[0], state[1]);
                return Core::Matrix{{state[0] / range, state[1] / range}};
            });
        });
```

`Predict()` and `Update()` fail with a message until the four functions and the error covariance are set.

### Angles

A heading or a bearing lives on a circle: 179 degrees measured against a prediction of -179 degrees is a 2 degree error,
not 358, and a plain subtraction gets it wrong. Two optional hooks fix that, in the linear and the extended filter alike:

```cpp
#include "core/angle.h"

filter.SetResidualFunction([](const Core::Vector &measurement, const Core::Vector &expected) {
    return Core::Vector{Core::AngleDifference(measurement[0], expected[0])}; // computes the innovation y
});
filter.SetStateNormalizer([](Core::Vector &state) {
    state[0] = Core::WrapAngle(state[0]); // adjusts the state before Predict() and Update() store it
});
```

The residual function replaces `y = z - h(x)`, and the state normalizer is the place to wrap angle states, or to
renormalize any other state with a constraint, once the filter has moved it. Both are empty by default, which keeps the
plain subtraction and leaves the state as computed; passing an empty function clears them.

## Mathematical background

### Prediction

- **Linear**: `x = F x + B u`
- **Extended**: `x = f(x, u)`, with `F` the Jacobian of `f` evaluated at the estimate the step starts from
- **Covariance**: `P = F P Fᵀ + Q`

### Update

- **Innovation**: `y = z - H x` (linear) or `y = z - h(x)` (extended, with `H` the Jacobian of `h`); a residual
  function can replace the subtraction
- **Innovation covariance**: `S = H P Hᵀ + R`
- **Kalman gain**: `K = P Hᵀ S⁻¹`
- **State**: `x = x + K y`
- **Covariance**: `P = (I - K H) P`

## License

This project is licensed under the GNU GPLv3. See the [LICENSE](LICENSE) file for details.
