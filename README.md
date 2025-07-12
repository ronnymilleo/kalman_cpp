[![CMake on multiple platforms](https://github.com/ronnymilleo/kalman_cpp/actions/workflows/cmake-multi-platform.yml/badge.svg?branch=main)](https://github.com/ronnymilleo/kalman_cpp/actions/workflows/cmake-multi-platform.yml)

# Kalman Filter Implementation in C++

This repository contains a modern C++ implementation of both standard Kalman filters and Extended Kalman filters (EKF) for real-time state estimation.

## Features

- **Modern C++ Implementation**: Uses STL containers and RAII principles
- **Dual Filter Support**: Both linear Kalman filter and Extended Kalman filter implementations
- **Non-Linear System Support**: EKF handles non-linear state transitions and observation functions
- **Dynamic Matrix Management**: Efficient std::vector-based matrix storage
- **Comprehensive Matrix Operations**: Addition, subtraction, multiplication, transpose, and inversion
- **Exception-Safe Design**: Robust error handling with proper exception management
- **Function Object Support**: EKF uses std::function for flexible non-linear function specification
- **Well-Documented API**: Complete Doxygen documentation with usage examples
- **Comprehensive Unit Tests**: Full test suite using Google Test framework
- **Example Applications**: 
  - Linear KF: 1D position tracking with velocity estimation
  - EKF: Circular motion tracking and pendulum dynamics

## Project Structure

```
├── kalman.h               # Standard Kalman filter class interface
├── kalman.cpp             # Standard Kalman filter implementation
├── extended_kalman.h      # Extended Kalman filter class interface  
├── extended_kalman.cpp    # Extended Kalman filter implementation
├── matrix_math.h          # Matrix operations interface
├── matrix_math.cpp        # Matrix operations implementation
├── kalman_demo.cpp        # Standard Kalman filter demo
├── extended_kalman_demo.cpp # Extended Kalman filter demonstrations
├── tests/
│   ├── test_matrix.cpp            # Unit tests for matrix operations
│   ├── test_kalman.cpp            # Unit tests for standard Kalman filter
│   ├── test_extended_kalman.cpp   # Unit tests for EKF
│   └── CMakeLists.txt
├── example-c/             # C implementation examples
│   ├── kalman.c
│   └── kalman.h
├── CMakeLists.txt         # Build configuration
└── CMakePresets.json      # CMake presets for different configurations
```

## Building

### Prerequisites
- CMake 3.14 or higher
- C++11-compatible compiler (GCC, Clang, MSVC)
- Google Test framework (for unit tests)
- Doxygen (optional, for generating documentation)

### Build Instructions

```bash
# Configure
cmake --preset debug
cmake --preset release

# Build
cmake --build --preset debug
cmake --build --preset release

# Run tests
ctest --preset debug          # Run tests (debug build)
ctest --preset release        # Run tests (release build)
```

## Usage Examples

### Standard Kalman Filter

```cpp
#include "kalman.h"

// Create a 2D state, 1D measurement Kalman filter
kalman kf(2, 1, 0);

// Initialize state [position=0, velocity=0]
std::vector<double> initialState = {0.0, 0.0};
kf.initializeState(initialState);

// Set up state transition matrix (constant velocity model)
std::vector<std::vector<double>> F = {
    {1.0, 1.0},  // dt = 1.0
    {0.0, 1.0}
};
kf.setStateTransitionMatrix(F);

// Set up observation matrix (measure position only)
std::vector<std::vector<double>> H = {
    {1.0, 0.0}
};
kf.setObservationMatrix(H);

// Set process noise covariance
std::vector<std::vector<double>> Q = {
    {0.1, 0.0},
    {0.0, 0.1}
};
kf.setProcessNoiseCovariance(Q);

// Set measurement noise covariance
std::vector<std::vector<double>> R = {{0.5}};
kf.setMeasurementNoiseCovariance(R);

// Set initial error covariance
std::vector<std::vector<double>> P = {
    {1.0, 0.0},
    {0.0, 1.0}
};
kf.setErrorCovariance(P);

// Filter cycle
for (auto measurement : measurements) {
    kf.predict();              // Prediction step
    kf.update({measurement});  // Update step
    
    auto state = kf.getState();
    std::cout << "Position: " << state[0] << ", Velocity: " << state[1] << std::endl;
}
```

### Extended Kalman Filter

```cpp
#include "extended_kalman.h"

// Create a 2D state, 1D measurement Extended Kalman filter
ExtendedKalman ekf(2, 1, 0);

// Define non-linear state transition function
auto stateFunc = [](const std::vector<double>& x, const std::vector<double>& u) {
    double dt = 0.1;
    return std::vector<double>{
        x[0] + x[1] * dt,  // position = position + velocity * dt
        x[1]               // velocity (constant)
    };
};

// Define state Jacobian matrix
auto stateJacFunc = [](const std::vector<double>& x) {
    double dt = 0.1;
    return std::vector<std::vector<double>>{
        {1.0, dt},
        {0.0, 1.0}
    };
};

// Define non-linear observation function (range measurement)
auto obsFunc = [](const std::vector<double>& x) {
    double range = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    return std::vector<double>{range};
};

// Define observation Jacobian matrix
auto obsJacFunc = [](const std::vector<double>& x) {
    double range = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    if (range < 1e-10) range = 1e-10; // Avoid division by zero
    
    return std::vector<std::vector<double>>{
        {x[0] / range, x[1] / range}
    };
};

// Set up EKF functions
ekf.setStateTransitionFunction(stateFunc);
ekf.setStateJacobianFunction(stateJacFunc);
ekf.setObservationFunction(obsFunc);
ekf.setObservationJacobianFunction(obsJacFunc);

// Set noise covariance matrices (same as standard KF)
// ... initialize Q, R, P matrices ...

ekf.setProcessNoiseCovariance(Q);
ekf.setMeasurementNoiseCovariance(R);
ekf.setErrorCovariance(P);

// Initialize state
std::vector<double> initialState = {1.0, 0.5};
ekf.initializeState(initialState);

// Filter cycle
for (auto measurement : measurements) {
    ekf.predict();              // Non-linear prediction step
    ekf.update({measurement});  // Non-linear update step
    
    auto state = ekf.getState();
    std::cout << "Position: " << state[0] << ", Velocity: " << state[1] << std::endl;
}
```

## Running the Demonstrations

The project includes two demonstration programs:

### Standard Kalman Filter Demo
```bash
./Release/bin/kalman_demo
```
This demonstrates 1D position tracking with velocity estimation using a linear constant velocity model.

### Extended Kalman Filter Demo
```bash
./Release/bin/extended_kalman_demo
```
This demonstrates two non-linear scenarios:
1. **Circular Motion Tracking**: Tracks an object moving in a circle using range-only measurements
2. **Pendulum Dynamics**: Estimates pendulum angle and angular velocity from noisy angle measurements

## Running Tests

```bash
# Run all tests
ctest

# Run specific test suite
ctest -R "Kalman"          # Standard KF tests only
ctest -R "ExtendedKalman"  # EKF tests only
ctest -R "MatrixMath"      # Matrix operation tests only

# Run tests with verbose output
ctest --verbose
```

## Key Differences: Kalman Filter vs Extended Kalman Filter

| Feature | Kalman Filter | Extended Kalman Filter |
|---------|---------------|------------------------|
| **System Model** | Linear | Non-linear |
| **State Transition** | Matrix multiplication (F·x) | Non-linear function f(x,u) |
| **Observation Model** | Matrix multiplication (H·x) | Non-linear function h(x) |
| **Linearization** | Not required | Uses Jacobian matrices |
| **Computational Cost** | Lower | Higher (due to Jacobian computation) |
| **Accuracy** | Optimal for linear systems | Approximation for non-linear systems |
| **Use Cases** | Linear tracking, simple models | Radar tracking, robotics, GPS/INS |

## Mathematical Background

### Standard Kalman Filter Equations

**Prediction:**
- x̂(k|k-1) = F·x̂(k-1|k-1) + B·u(k-1)
- P(k|k-1) = F·P(k-1|k-1)·F^T + Q

**Update:**
- K(k) = P(k|k-1)·H^T·[H·P(k|k-1)·H^T + R]^(-1)
- x̂(k|k) = x̂(k|k-1) + K(k)·[z(k) - H·x̂(k|k-1)]
- P(k|k) = [I - K(k)·H]·P(k|k-1)

### Extended Kalman Filter Equations

**Prediction:**
- x̂(k|k-1) = f(x̂(k-1|k-1), u(k-1))
- F(k) = ∂f/∂x |_(x̂(k-1|k-1))
- P(k|k-1) = F(k)·P(k-1|k-1)·F(k)^T + Q

**Update:**
- H(k) = ∂h/∂x |_(x̂(k|k-1))
- K(k) = P(k|k-1)·H(k)^T·[H(k)·P(k|k-1)·H(k)^T + R]^(-1)
- x̂(k|k) = x̂(k|k-1) + K(k)·[z(k) - h(x̂(k|k-1))]
- P(k|k) = [I - K(k)·H(k)]·P(k|k-1)

## Documentation

This project includes comprehensive Doxygen documentation.

### Documentation Features
- **Complete API Reference**: All classes, methods, and parameters documented
- **Usage Examples**: Practical C++ code examples for each function
- **Mathematical Background**: Detailed explanation of Kalman filter equations
- **Group Organization**: Methods organized by purpose (constructors, operations, etc.)
- **Cross-References**: Links between related classes and functions
- **Exception Documentation**: All possible exceptions and error conditions

## API Reference

### Kalman Filter Class

- `kalman()` - Constructors (default and parameterized)
- `initializeState()` - Set initial state vector
- `setStateTransitionMatrix()` - Set F matrix (system dynamics)
- `setControlMatrix()` - Set B matrix (control inputs)
- `setObservationMatrix()` - Set H matrix (measurement model)
- `setProcessNoiseCovariance()` - Set Q matrix (process uncertainty)
- `setMeasurementNoiseCovariance()` - Set R matrix (measurement uncertainty)
- `setErrorCovariance()` - Set P matrix (initial state uncertainty)
- `predict()` - Prediction step of the filter
- `update()` - Correction step of the filter
- `getState()` - Get current state vector
- `getStateElement()` - Get specific state element
- `getErrorCovariance()` - Get current error covariance matrix
- `reset()` - Reset filter to uninitialized state
- `isInitialized()` - Check if filter is ready for use

### Extended Kalman Filter Class

- `ExtendedKalman()` - Constructors (default and parameterized)
- `initializeState()` - Set initial state vector
- `setStateTransitionFunction()` - Set non-linear state transition function f(x,u)
- `setStateJacobianFunction()` - Set Jacobian matrix function ∂f/∂x
- `setObservationFunction()` - Set non-linear observation function h(x)
- `setObservationJacobianFunction()` - Set observation Jacobian function ∂h/∂x
- `setControlMatrix()` - Set B matrix (control inputs)
- `setProcessNoiseCovariance()` - Set Q matrix (process uncertainty)
- `setMeasurementNoiseCovariance()` - Set R matrix (measurement uncertainty)
- `setErrorCovariance()` - Set P matrix (initial state uncertainty)
- `predict()` - Non-linear prediction step with Jacobian linearization
- `update()` - Non-linear update step with observation Jacobian
- `getState()` - Get current state vector
- `getStateElement()` - Get specific state element
- `getErrorCovariance()` - Get current error covariance matrix
- `reset()` - Reset filter to uninitialized state
- `isInitialized()` - Check if filter is ready for use

### Matrix Functions

- `matrix_add()` - Matrix addition (A + B)
- `matrix_subtract()` - Matrix subtraction (A - B)
- `matrix_multiply()` - Matrix multiplication (A * B)
- `matrix_vector_multiply()` - Matrix-vector multiplication (A * v)
- `matrix_transpose()` - Matrix transpose (A^T)
- `matrix_inverse()` - Matrix inversion (A^-1) using Gaussian elimination
- `matrix_identity()` - Create identity matrix

## Mathematical Background

The Kalman filter implementation follows the standard discrete-time formulation:

**Prediction Step:**
- State prediction: `x(k|k-1) = F * x(k-1|k-1) + B * u(k-1)`
- Covariance prediction: `P(k|k-1) = F * P(k-1|k-1) * F^T + Q`

**Update Step:**
- Innovation (residual): `y(k) = z(k) - H * x(k|k-1)`
- Innovation covariance: `S(k) = H * P(k|k-1) * H^T + R`
- Kalman gain: `K(k) = P(k|k-1) * H^T * S(k)^(-1)`
- State update: `x(k|k) = x(k|k-1) + K(k) * y(k)`
- Covariance update: `P(k|k) = (I - K(k) * H) * P(k|k-1)`

Where:
- `x` = state vector
- `F` = state transition matrix
- `B` = control matrix (optional)
- `u` = control vector (optional)
- `Q` = process noise covariance
- `H` = observation matrix
- `R` = measurement noise covariance
- `P` = error covariance matrix
- `z` = measurement vector
- `I` = identity matrix

## License

This project is licensed under the GNU GPLv3 - see the [LICENSE](LICENSE) file for details.
