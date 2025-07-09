//
// Created by Ronny Milleo on 08/07/25.
//

#ifndef KALMAN_H
#define KALMAN_H

#include <vector>

class kalman {
private:
    // State vector (x)
    std::vector<double> state;

    // Error covariance matrix (P)
    std::vector<std::vector<double>> errorCovariance;

    // State transition matrix (F)
    std::vector<std::vector<double>> stateTransition;

    // Control matrix (B)
    std::vector<std::vector<double>> controlMatrix;

    // Process noise covariance matrix (Q)
    std::vector<std::vector<double>> processNoise;

    // Observation matrix (H)
    std::vector<std::vector<double>> observationMatrix;

    // Measurement noise covariance matrix (R)
    std::vector<std::vector<double>> measurementNoise;

    // Identity matrix for calculations
    std::vector<std::vector<double>> identity;

    // Dimensions
    int stateDim;
    int measurementDim;
    int controlDim;

public:
    // Constructors
    kalman() = default;
    kalman(int stateDimension, int measurementDimension, int controlDimension = 0);
    ~kalman() = default;

    // Initialization methods
    void initializeState(const std::vector<double>& initialState);
    void setStateTransitionMatrix(const std::vector<std::vector<double>>& F);
    void setControlMatrix(const std::vector<std::vector<double>>& B);
    void setObservationMatrix(const std::vector<std::vector<double>>& H);
    void setProcessNoiseCovariance(const std::vector<std::vector<double>>& Q);
    void setMeasurementNoiseCovariance(const std::vector<std::vector<double>>& R);
    void setErrorCovariance(const std::vector<std::vector<double>>& P);

    // Core Kalman filter operations
    void predict(const std::vector<double>& control = {});
    void update(const std::vector<double>& measurement);

    // Getters
    std::vector<double>& getState();
    std::vector<std::vector<double>>& getErrorCovariance();
    double getStateElement(int index) const;

    // Utility methods
    void reset();
    bool isInitialized() const;

private:
    bool initialized;
};

#endif //KALMAN_H
