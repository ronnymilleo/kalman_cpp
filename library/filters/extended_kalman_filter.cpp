/**
 * @file extended_kalman.cpp
 * @brief Extended Kalman filter class implementation
 * @author ronnymilleo
 * @date 12/07/25
 * @version 1.0
 * 
 * This file implements the Extended Kalman filter class for real-time state 
 * estimation of non-linear systems. The implementation follows the standard 
 * discrete-time Extended Kalman filter algorithm with prediction and update 
 * steps using linearization via Jacobian matrices. All matrix operations are 
 * delegated to the matrix_math module for modularity and maintainability.
 * 
 * @details Implementation features:
 * - Standard discrete-time Extended Kalman filter equations
 * - Non-linear state transition and observation functions
 * - Automatic linearization using user-provided Jacobian functions
 * - Comprehensive error checking and validation
 * - Efficient memory management with pre-allocated matrices
 * - Exception-safe design with proper error handling
 * - Support for optional control inputs
 * - Robust initialization and state management
 * 
 * @note Uses std::vector for dynamic matrix storage
 * @note All matrix operations are performed through matrix_math functions
 * @note Filter state is maintained between predict/update cycles
 * @note Non-linear functions are provided via std::function objects
 * 
 * @see extended_kalman.h for class interface documentation
 * @see matrix_math.h for matrix operation implementations
 * @see kalman.h for standard linear Kalman filter
 */

#include "extended_kalman.h"
#include "matrix_math.h"
#include <stdexcept>
#include <cstddef>

// Constructor with dimensions
ExtendedKalman::ExtendedKalman(const size_t stateDimension, const size_t measurementDimension, const size_t controlDimension)
    : stateDim(stateDimension), measurementDim(measurementDimension), controlDim(controlDimension), initialized(false) {
    
    if (stateDimension < 1 || measurementDimension < 1) {
        throw std::invalid_argument("State and measurement dimensions must be at least 1");
    }

    // Initialize state vector
    state.resize(stateDim, 0.0);

    // Initialize matrices with appropriate dimensions
    errorCovariance.resize(stateDim, std::vector<double>(stateDim, 0.0));
    processNoise.resize(stateDim, std::vector<double>(stateDim, 0.0));
    measurementNoise.resize(measurementDim, std::vector<double>(measurementDim, 0.0));

    // Initialize identity matrix
    identity.resize(stateDim, std::vector<double>(stateDim, 0.0));
    matrix_identity(identity);
}

// Initialize state vector
void ExtendedKalman::initializeState(const std::vector<double>& initialState) {
    if (initialState.size() != stateDim) {
        throw std::invalid_argument("Initial state size mismatch");
    }
    state = initialState;
}

// Set non-linear state transition function f(x, u)
void ExtendedKalman::setStateTransitionFunction(const StateTransitionFunction& func) {
    if (!func) {
        throw std::invalid_argument("State transition function cannot be null");
    }
    stateTransitionFunc = func;
}

// Set non-linear observation function h(x)
void ExtendedKalman::setObservationFunction(const ObservationFunction& func) {
    if (!func) {
        throw std::invalid_argument("Observation function cannot be null");
    }
    observationFunc = func;
}

// Set state Jacobian function ∂f/∂x
void ExtendedKalman::setStateJacobianFunction(const StateJacobianFunction& func) {
    if (!func) {
        throw std::invalid_argument("State Jacobian function cannot be null");
    }
    stateJacobianFunc = func;
}

// Set observation Jacobian function ∂h/∂x
void ExtendedKalman::setObservationJacobianFunction(const ObservationJacobianFunction& func) {
    if (!func) {
        throw std::invalid_argument("Observation Jacobian function cannot be null");
    }
    observationJacobianFunc = func;
}

// Set process noise covariance Q
void ExtendedKalman::setProcessNoiseCovariance(const std::vector<std::vector<double>>& Q) {
    if (Q.size() != stateDim || Q[0].size() != stateDim) {
        throw std::invalid_argument("Process noise matrix size mismatch");
    }
    processNoise = Q;
}

// Set measurement noise covariance R
void ExtendedKalman::setMeasurementNoiseCovariance(const std::vector<std::vector<double>>& R) {
    if (R.size() != measurementDim || R[0].size() != measurementDim) {
        throw std::invalid_argument("Measurement noise matrix size mismatch");
    }
    measurementNoise = R;
}

// Set error covariance P
void ExtendedKalman::setErrorCovariance(const std::vector<std::vector<double>>& P) {
    if (P.size() != stateDim || P[0].size() != stateDim) {
        throw std::invalid_argument("Error covariance matrix size mismatch");
    }
    errorCovariance = P;
    initialized = true;
}

// Prediction step
void ExtendedKalman::predict(const std::vector<double>& control) {
    if (!initialized) {
        throw std::runtime_error("Extended Kalman filter not initialized");
    }
    
    if (!areAllFunctionsSet()) {
        throw std::runtime_error("All required functions must be set before prediction");
    }

    // Check control input size if control is provided
    if (!control.empty() && controlDim > 0 && control.size() != controlDim) {
        throw std::invalid_argument("Control vector size mismatch");
    }

    // Predict state using non-linear function: x = f(x, u)
    state = stateTransitionFunc(state, control);

    // Compute state Jacobian matrix F = ∂f/∂x at current state
    auto F = stateJacobianFunc(state);
    
    // Validate Jacobian dimensions
    if (F.size() != stateDim || F[0].size() != stateDim) {
        throw std::runtime_error("State Jacobian matrix has incorrect dimensions");
    }

    // Predict error covariance: P = F * P * F^T + Q
    std::vector<std::vector<double>> FP(stateDim, std::vector<double>(stateDim, 0.0));
    std::vector<std::vector<double>> FT(stateDim, std::vector<double>(stateDim, 0.0));
    std::vector<std::vector<double>> FPFT(stateDim, std::vector<double>(stateDim, 0.0));

    matrix_multiply(F, errorCovariance, FP);
    matrix_transpose(F, FT);
    matrix_multiply(FP, FT, FPFT);
    matrix_add(FPFT, processNoise, errorCovariance);
}

// Update step (correction)
void ExtendedKalman::update(const std::vector<double>& measurement) {
    if (!initialized) {
        throw std::runtime_error("Extended Kalman filter not initialized");
    }
    
    if (!areAllFunctionsSet()) {
        throw std::runtime_error("All required functions must be set before update");
    }

    if (measurement.size() != measurementDim) {
        throw std::invalid_argument("Measurement vector size mismatch");
    }

    // Predict measurement using non-linear function: z_pred = h(x)
    auto predictedMeasurement = observationFunc(state);
    
    // Validate predicted measurement dimensions
    if (predictedMeasurement.size() != measurementDim) {
        throw std::runtime_error("Observation function returned incorrect measurement dimensions");
    }

    // Calculate innovation: y = z - h(x)
    std::vector<double> innovation(measurementDim);
    for (size_t i = 0; i < measurementDim; i++) {
        innovation[i] = measurement[i] - predictedMeasurement[i];
    }

    // Compute observation Jacobian matrix H = ∂h/∂x at current state
    auto H = observationJacobianFunc(state);
    
    // Validate Jacobian dimensions
    if (H.size() != measurementDim || H[0].size() != stateDim) {
        throw std::runtime_error("Observation Jacobian matrix has incorrect dimensions");
    }

    // Calculate innovation covariance: S = H * P * H^T + R
    std::vector<std::vector<double>> HP(measurementDim, std::vector<double>(stateDim, 0.0));
    std::vector<std::vector<double>> HT(stateDim, std::vector<double>(measurementDim, 0.0));
    std::vector<std::vector<double>> HPHT(measurementDim, std::vector<double>(measurementDim, 0.0));
    std::vector<std::vector<double>> S(measurementDim, std::vector<double>(measurementDim, 0.0));

    matrix_multiply(H, errorCovariance, HP);
    matrix_transpose(H, HT);
    matrix_multiply(HP, HT, HPHT);
    matrix_add(HPHT, measurementNoise, S);

    // Calculate Kalman gain: K = P * H^T * S^(-1)
    std::vector<std::vector<double>> PHT(stateDim, std::vector<double>(measurementDim, 0.0));
    std::vector<std::vector<double>> SInv(measurementDim, std::vector<double>(measurementDim, 0.0));
    std::vector<std::vector<double>> K(stateDim, std::vector<double>(measurementDim, 0.0));

    matrix_multiply(errorCovariance, HT, PHT);
    
    // Attempt matrix inversion - handle potential failure
    try {
        matrix_inverse(S, SInv);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to invert innovation covariance matrix - matrix may be singular");
    }
    
    matrix_multiply(PHT, SInv, K);

    // Update state: x = x + K * y
    std::vector<double> Ky(stateDim);
    matrix_vector_multiply(K, innovation, Ky);
    for (size_t i = 0; i < stateDim; i++) {
        state[i] += Ky[i];
    }

    // Update error covariance: P = (I - K * H) * P
    std::vector<std::vector<double>> KH(stateDim, std::vector<double>(stateDim, 0.0));
    std::vector<std::vector<double>> IKH(stateDim, std::vector<double>(stateDim, 0.0));
    std::vector<std::vector<double>> result(stateDim, std::vector<double>(stateDim, 0.0));

    matrix_multiply(K, H, KH);
    matrix_subtract(identity, KH, IKH);
    matrix_multiply(IKH, errorCovariance, result);

    // Copy result back to error covariance
    for (size_t i = 0; i < stateDim; i++) {
        for (size_t j = 0; j < stateDim; j++) {
            errorCovariance[i][j] = result[i][j];
        }
    }
}

// Getters
std::vector<double>& ExtendedKalman::getState() {
    if (!initialized) {
        throw std::runtime_error("Extended Kalman filter not initialized");
    }
    return state;
}

std::vector<std::vector<double>>& ExtendedKalman::getErrorCovariance() {
    if (!initialized) {
        throw std::runtime_error("Extended Kalman filter not initialized");
    }
    return errorCovariance;
}

double ExtendedKalman::getStateElement(const size_t index) const {
    if (!initialized) {
        throw std::runtime_error("Extended Kalman filter not initialized");
    }
    if (index >= stateDim) {
        throw std::out_of_range("State index out of range");
    }
    return state[index];
}

// Utility methods
void ExtendedKalman::reset() {
    // Clear state vector
    for (auto& element : state) {
        element = 0.0;
    }
    
    // Clear error covariance matrix
    for (auto& row : errorCovariance) {
        for (auto& element : row) {
            element = 0.0;
        }
    }
    
    // Clear all function objects
    stateTransitionFunc = nullptr;
    observationFunc = nullptr;
    stateJacobianFunc = nullptr;
    observationJacobianFunc = nullptr;
    
    initialized = false;
}

bool ExtendedKalman::isInitialized() const {
    return initialized && areAllFunctionsSet();
}

bool ExtendedKalman::areAllFunctionsSet() const {
    return stateTransitionFunc && observationFunc && 
           stateJacobianFunc && observationJacobianFunc;
}
