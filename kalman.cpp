/**
 * @file kalman.cpp
 * @brief Kalman filter class implementation
 * @author ronnymilleo
 * @date 08/07/25
 * @version 1.0
 * 
 * This file implements the Kalman filter class for real-time state estimation.
 * The implementation follows the standard discrete-time Kalman filter algorithm
 * with prediction and update steps. All matrix operations are delegated to
 * the matrix_math module for modularity and maintainability.
 * 
 * @details Implementation features:
 * - Standard discrete-time Kalman filter equations
 * - Comprehensive error checking and validation
 * - Efficient memory management with pre-allocated matrices
 * - Exception-safe design with proper error handling
 * - Support for optional control inputs
 * - Robust initialization and state management
 * 
 * @note Uses std::vector for dynamic matrix storage
 * @note All matrix operations are performed through matrix_math functions
 * @note Filter state is maintained between predict/update cycles
 * 
 * @see kalman.h for class interface documentation
 * @see matrix_math.h for matrix operation implementations
 */

#include "kalman.h"
#include <stdexcept>

#include "matrix_math.h"

// Constructor with dimensions
kalman::kalman(const int stateDimension, const int measurementDimension, const int controlDimension)
    : stateDim(stateDimension), measurementDim(measurementDimension),
      controlDim(controlDimension), initialized(false) {

    // Initialize state vector
    state.resize(stateDim, 0.0);

    // Initialize matrices with appropriate dimensions
    errorCovariance.resize(stateDim, std::vector<double>(stateDim, 0.0));
    stateTransition.resize(stateDim, std::vector<double>(stateDim, 0.0));
    processNoise.resize(stateDim, std::vector<double>(stateDim, 0.0));
    observationMatrix.resize(measurementDim, std::vector<double>(stateDim, 0.0));
    measurementNoise.resize(measurementDim, std::vector<double>(measurementDim, 0.0));

    if (controlDim > 0) {
        controlMatrix.resize(stateDim, std::vector<double>(controlDim, 0.0));
    }

    identity.resize(stateDim, std::vector<double>(stateDim, 0.0));
    matrix_identity(identity);
}

// Initialize state vector
void kalman::initializeState(const std::vector<double>& initialState) {
    if (initialState.size() != stateDim) {
        throw std::invalid_argument("Initial state size mismatch");
    }
    state = initialState;
}

// Set state transition matrix F
void kalman::setStateTransitionMatrix(const std::vector<std::vector<double>>& F) {
    if (F.size() != stateDim || F[0].size() != stateDim) {
        throw std::invalid_argument("State transition matrix size mismatch");
    }
    stateTransition = F;
}

// Set control matrix B
void kalman::setControlMatrix(const std::vector<std::vector<double>>& B) {
    if (B.size() != stateDim || B[0].size() != controlDim) {
        throw std::invalid_argument("Control matrix size mismatch");
    }
    controlMatrix = B;
}

// Set observation matrix H
void kalman::setObservationMatrix(const std::vector<std::vector<double>>& H) {
    if (H.size() != measurementDim || H[0].size() != stateDim) {
        throw std::invalid_argument("Observation matrix size mismatch");
    }
    observationMatrix = H;
}

// Set process noise covariance Q
void kalman::setProcessNoiseCovariance(const std::vector<std::vector<double>>& Q) {
    if (Q.size() != stateDim || Q[0].size() != stateDim) {
        throw std::invalid_argument("Process noise matrix size mismatch");
    }
    processNoise = Q;
}

// Set measurement noise covariance R
void kalman::setMeasurementNoiseCovariance(const std::vector<std::vector<double>>& R) {
    if (R.size() != measurementDim || R[0].size() != measurementDim) {
        throw std::invalid_argument("Measurement noise matrix size mismatch");
    }
    measurementNoise = R;
}

// Set error covariance P
void kalman::setErrorCovariance(const std::vector<std::vector<double>>& P) {
    if (P.size() != stateDim || P[0].size() != stateDim) {
        throw std::invalid_argument("Error covariance matrix size mismatch");
    }
    errorCovariance = P;
    initialized = true;
}

// Prediction step
void kalman::predict(const std::vector<double>& control) {
    if (!initialized) {
        throw std::runtime_error("Kalman filter not initialized");
    }

    // Predict state: x = F * x + B * u
    std::vector<double> result(stateDim);
    matrix_vector_multiply(stateTransition, state, result);
    for (int i = 0; i < stateDim; i++) {
        state[i] = result[i];
    }

    if (!control.empty() && controlDim > 0) {
        if (control.size() != controlDim) {
            throw std::invalid_argument("Control vector size mismatch");
        }
        std::vector<double> controlContribution(stateDim);
        matrix_vector_multiply(controlMatrix, control, controlContribution);
        for (int i = 0; i < stateDim; i++) {
            state[i] += controlContribution[i];
        }
    }

    std::vector<std::vector<double>> FP(stateDim, std::vector<double>(stateDim, 0.0));
    std::vector<std::vector<double>> FT(stateDim, std::vector<double>(stateDim, 0.0));
    std::vector<std::vector<double>> FPFT(stateDim, std::vector<double>(stateDim, 0.0));

    // Predict error covariance: P = F * P * F^T + Q
    matrix_multiply(stateTransition, errorCovariance, FP);
    matrix_transpose(stateTransition, FT);
    matrix_multiply(FP, FT, FPFT);
    matrix_add(FPFT, processNoise, errorCovariance);
}

// Update step (correction)
void kalman::update(const std::vector<double>& measurement) {
    if (!initialized) {
        throw std::runtime_error("Kalman filter not initialized");
    }

    if (measurement.size() != measurementDim) {
        throw std::invalid_argument("Measurement vector size mismatch");
    }

    std::vector<double> predictedMeasurement(measurementDim);

    // Calculate innovation: y = z - H * x
    matrix_vector_multiply(observationMatrix, state, predictedMeasurement);
    std::vector<double> innovation(measurementDim);
    for (int i = 0; i < measurementDim; i++) {
        innovation[i] = measurement[i] - predictedMeasurement[i];
    }

    std::vector<std::vector<double>> HP(measurementDim, std::vector<double>(stateDim, 0.0));
    std::vector<std::vector<double>> HT(stateDim, std::vector<double>(measurementDim, 0.0));
    std::vector<std::vector<double>> HPHT(measurementDim, std::vector<double>(measurementDim, 0.0));
    std::vector<std::vector<double>> S(measurementDim, std::vector<double>(measurementDim, 0.0));

    // Calculate innovation covariance: S = H * P * H^T + R
    matrix_multiply(observationMatrix, errorCovariance, HP);
    matrix_transpose(observationMatrix, HT);
    matrix_multiply(HP, HT, HPHT);
    matrix_add(HPHT, measurementNoise, S);

    std::vector<std::vector<double>> PHT(stateDim, std::vector<double>(measurementDim, 0.0));
    std::vector<std::vector<double>> SInv(measurementDim, std::vector<double>(measurementDim, 0.0));
    std::vector<std::vector<double>> K(stateDim, std::vector<double>(measurementDim, 0.0));

    // Calculate Kalman gain: K = P * H^T * S^(-1)
    matrix_multiply(errorCovariance, HT, PHT);
    matrix_inverse(S, SInv);
    matrix_multiply(PHT, SInv, K);

    std::vector<double> Ky(stateDim);

    // Update state: x = x + K * y
    matrix_vector_multiply(K, innovation, Ky);
    for (int i = 0; i < stateDim; i++) {
        state[i] += Ky[i];
    }

    std::vector<std::vector<double>> KH(stateDim, std::vector<double>(stateDim, 0.0));
    std::vector<std::vector<double>> IKH(stateDim, std::vector<double>(stateDim, 0.0));
    std::vector<std::vector<double>> result(stateDim, std::vector<double>(stateDim, 0.0));

    // Update error covariance: P = (I - K * H) * P
    matrix_multiply(K, observationMatrix, KH);
    matrix_subtract(identity, KH, IKH);
    matrix_multiply(IKH, errorCovariance, result);

    for (int i = 0; i < stateDim; i++) {
        for (int j = 0; j < stateDim; j++) {
            errorCovariance[i][j] = result[i][j];
        }
    }
}

// Getters
std::vector<double>& kalman::getState() {
    return state;
}

std::vector<std::vector<double>>& kalman::getErrorCovariance() {
    return errorCovariance;
}

double kalman::getStateElement(const int index) const {
    if (index < 0 || index >= stateDim) {
        throw std::out_of_range("State index out of range");
    }
    return state[index];
}

// Utility methods
void kalman::reset() {
    for (auto& element : state) {
        element = 0.0;
    }
    for (auto& row : errorCovariance) {
        for (auto& element : row) {
            element = 0.0;
        }
    }
    initialized = false;
}

bool kalman::isInitialized() const {
    return initialized;
}
