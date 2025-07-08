#include <iostream>
#include "kalman.h"

int main() {
    std::cout << "Kalman Filter Demo\n";
    std::cout << "==================\n\n";

    try {
        // Example: 1D position tracking with velocity
        // State: [position, velocity]
        // Measurement: [position]

        // Create Kalman filter (2D state, 1D measurement, no control)
        kalman kf(2, 1, 0);

        // Initialize state [position=0, velocity=0]
        std::vector<double> initialState = {0.0, 0.0};
        kf.initializeState(initialState);

        // State transition matrix (constant velocity model)
        // x(k+1) = x(k) + v(k) * dt
        // v(k+1) = v(k)
        double dt = 1.0; // time step
        std::vector<std::vector<double>> F = {
            {1.0, dt},
            {0.0, 1.0}
        };
        kf.setStateTransitionMatrix(F);

        // Observation matrix (we can only measure position)
        std::vector<std::vector<double>> H = {
            {1.0, 0.0}
        };
        kf.setObservationMatrix(H);

        // Process noise covariance
        std::vector<std::vector<double>> Q = {
            {0.1, 0.0},
            {0.0, 0.1}
        };
        kf.setProcessNoiseCovariance(Q);

        // Measurement noise covariance
        std::vector<std::vector<double>> R = {
            {1.0}
        };
        kf.setMeasurementNoiseCovariance(R);

        // Initial error covariance
        std::vector<std::vector<double>> P = {
            {1.0, 0.0},
            {0.0, 1.0}
        };
        kf.setErrorCovariance(P);

        std::cout << "Kalman filter initialized successfully!\n";
        std::cout << "Initial state: [" << kf.getStateElement(0) << ", " << kf.getStateElement(1) << "]\n\n";

        // Simulate some measurements
        std::vector<double> measurements = {1.2, 2.8, 4.1, 5.9, 7.8, 9.2, 11.1, 12.9};

        std::cout << "Running Kalman filter with measurements:\n";
        std::cout << "Step\tMeasurement\tEstimated Pos\tEstimated Vel\n";
        std::cout << "----\t-----------\t-------------\t-------------\n";

        for (size_t i = 0; i < measurements.size(); i++) {
            // Predict step
            kf.predict();

            // Update step with measurement
            std::vector<double> z = {measurements[i]};
            kf.update(z);

            // Display results
            std::cout << i + 1 << "\t" << measurements[i] << "\t\t"
                      << kf.getStateElement(0) << "\t\t"
                      << kf.getStateElement(1) << "\n";
        }

        std::cout << "\nFinal estimated state:\n";
        std::cout << "Position: " << kf.getStateElement(0) << "\n";
        std::cout << "Velocity: " << kf.getStateElement(1) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}