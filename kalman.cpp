//
// Created by Ronny Milleo on 08/07/25.
//

#include "kalman.h"
#include <stdexcept>
#include <cmath>

// Constructor with dimensions
kalman::kalman(int stateDimension, int measurementDimension, int controlDimension)
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

    createIdentityMatrix(stateDim);
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
    state = matrixVectorMultiply(stateTransition, state);

    if (!control.empty() && controlDim > 0) {
        if (control.size() != controlDim) {
            throw std::invalid_argument("Control vector size mismatch");
        }
        auto controlContribution = matrixVectorMultiply(controlMatrix, control);
        for (int i = 0; i < stateDim; i++) {
            state[i] += controlContribution[i];
        }
    }

    // Predict error covariance: P = F * P * F^T + Q
    auto FP = matrixMultiply(stateTransition, errorCovariance);
    auto FT = matrixTranspose(stateTransition);
    auto FPFT = matrixMultiply(FP, FT);
    errorCovariance = matrixAdd(FPFT, processNoise);
}

// Update step (correction)
void kalman::update(const std::vector<double>& measurement) {
    if (!initialized) {
        throw std::runtime_error("Kalman filter not initialized");
    }

    if (measurement.size() != measurementDim) {
        throw std::invalid_argument("Measurement vector size mismatch");
    }

    // Calculate innovation: y = z - H * x
    auto predictedMeasurement = matrixVectorMultiply(observationMatrix, state);
    std::vector<double> innovation(measurementDim);
    for (int i = 0; i < measurementDim; i++) {
        innovation[i] = measurement[i] - predictedMeasurement[i];
    }

    // Calculate innovation covariance: S = H * P * H^T + R
    auto HP = matrixMultiply(observationMatrix, errorCovariance);
    auto HT = matrixTranspose(observationMatrix);
    auto HPHT = matrixMultiply(HP, HT);
    auto S = matrixAdd(HPHT, measurementNoise);

    // Calculate Kalman gain: K = P * H^T * S^(-1)
    auto PHT = matrixMultiply(errorCovariance, HT);
    auto SInv = matrixInverse(S);
    auto K = matrixMultiply(PHT, SInv);

    // Update state: x = x + K * y
    auto Ky = matrixVectorMultiply(K, innovation);
    for (int i = 0; i < stateDim; i++) {
        state[i] += Ky[i];
    }

    // Update error covariance: P = (I - K * H) * P
    auto KH = matrixMultiply(K, observationMatrix);
    auto IKH = matrixSubtract(identity, KH);
    errorCovariance = matrixMultiply(IKH, errorCovariance);
}

// Getters
std::vector<double> kalman::getState() const {
    return state;
}

std::vector<std::vector<double>> kalman::getErrorCovariance() const {
    return errorCovariance;
}

double kalman::getStateElement(int index) const {
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

// Matrix operations
std::vector<std::vector<double>> kalman::matrixMultiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) const {

    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();

    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

std::vector<double> kalman::matrixVectorMultiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& v) const {

    int rows = A.size();
    int cols = A[0].size();

    std::vector<double> result(rows, 0.0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += A[i][j] * v[j];
        }
    }

    return result;
}

std::vector<std::vector<double>> kalman::matrixAdd(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) const {

    int rows = A.size();
    int cols = A[0].size();

    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }

    return result;
}

std::vector<std::vector<double>> kalman::matrixSubtract(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) const {

    int rows = A.size();
    int cols = A[0].size();

    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }

    return result;
}

std::vector<std::vector<double>> kalman::matrixTranspose(
    const std::vector<std::vector<double>>& A) const {

    int rows = A.size();
    int cols = A[0].size();

    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = A[i][j];
        }
    }

    return result;
}

std::vector<std::vector<double>> kalman::matrixInverse(
    const std::vector<std::vector<double>>& A) const {

    int n = A.size();
    std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n, 0.0));

    // Create augmented matrix [A|I]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i][j] = A[i][j];
        }
        augmented[i][i + n] = 1.0;
    }

    // Gaussian elimination
    for (int i = 0; i < n; i++) {
        // Find pivot
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (std::abs(augmented[k][i]) > std::abs(augmented[maxRow][i])) {
                maxRow = k;
            }
        }

        // Swap rows
        if (maxRow != i) {
            std::swap(augmented[i], augmented[maxRow]);
        }

        // Make diagonal element 1
        double pivot = augmented[i][i];
        if (std::abs(pivot) < 1e-10) {
            throw std::runtime_error("Matrix is singular");
        }

        for (int j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivot;
        }

        // Eliminate column
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = augmented[k][i];
                for (int j = 0; j < 2 * n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    // Extract inverse matrix
    std::vector<std::vector<double>> result(n, std::vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = augmented[i][j + n];
        }
    }

    return result;
}

void kalman::createIdentityMatrix(int size) {
    identity.resize(size, std::vector<double>(size, 0.0));
    for (int i = 0; i < size; i++) {
        identity[i][i] = 1.0;
    }
}
