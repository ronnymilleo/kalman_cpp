# Kalman Filter Implementation in C++

This repository contains a modern C++ implementation of a Kalman filter for real-time state estimation.

## Features

- **Modern C++ Implementation**: Uses STL containers and RAII principles
- **Dynamic Matrix Management**: Efficient std::vector-based matrix storage
- **Comprehensive Matrix Operations**: Addition, subtraction, multiplication, transpose, and inversion
- **Exception-Safe Design**: Robust error handling with proper exception management
- **Well-Documented API**: Complete Doxygen documentation with usage examples
- **Unit Tests**: Comprehensive test suite using Google Test framework
- **Example Application**: 1D position tracking with velocity estimation

## Project Structure

```
├── kalman.h          # Main Kalman filter class interface
├── kalman.cpp        # Kalman filter class implementation
├── matrix_math.h     # Matrix operations interface
├── matrix_math.cpp   # Matrix operations implementation
├── main.cpp          # Example usage and demonstration
├── tests/
│   ├── test_matrix.cpp # Unit tests for matrix operations
│   └── CMakeLists.txt
├── CMakeLists.txt    # Build configuration
└── .gitattributes    # Git line ending configuration
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
ctest --preset tests
```

## Usage Example

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
std::vector<std::vector<double>> R = {{1.0}};
kf.setMeasurementNoiseCovariance(R);

// Set initial error covariance
std::vector<std::vector<double>> P = {
    {1.0, 0.0},
    {0.0, 1.0}
};
kf.setErrorCovariance(P);

// Prediction and update cycle
kf.predict();                           // Predict step
std::vector<double> measurement = {1.2};
kf.update(measurement);                 // Update step

// Get results
double position = kf.getStateElement(0);
double velocity = kf.getStateElement(1);
```

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

This project is licensed under the GNU GPLv3 - see the LICENSE file for details.
