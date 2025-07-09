#include <iostream>
#include <iomanip>
#include <string>
#include "kalman.h"

// Simple color codes
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define GREEN   "\033[32m"
#define BLUE    "\033[34m"
#define YELLOW  "\033[33m"

void printHeader() {
    std::cout << BOLD << "\n=== KALMAN FILTER DEMO ===" << RESET << "\n";
    std::cout << "1D Position Tracking with Velocity\n\n";
}

void printSection(const std::string& title) {
    std::cout << BLUE << title << RESET << "\n";
    std::cout << std::string(title.length(), '-') << "\n";
}

int main() {
    printHeader();

    try {
        printSection("Filter Initialization");

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

        std::cout << GREEN << "Filter configured successfully!" << RESET << "\n";
        std::cout << "State: [position, velocity] - Measurement: [position]\n";
        std::cout << "Initial Position: " << std::fixed << std::setprecision(3)
                  << kf.getStateElement(0) << " units\n";
        std::cout << "Initial Velocity: " << std::fixed << std::setprecision(3)
                  << kf.getStateElement(1) << " units/s\n\n";

        // Simulate some measurements
        std::vector<double> measurements = {1.2, 2.8, 4.1, 5.9, 7.8, 9.2, 11.1, 12.9};

        printSection("Processing Measurements");

        std::cout << BOLD << "Step  Measurement  Position  Velocity" << RESET << "\n";
        std::cout << "----  -----------  --------  --------\n";

        for (size_t i = 0; i < measurements.size(); i++) {
            // Predict step
            kf.predict();

            // Update step with measurement
            std::vector<double> z = {measurements[i]};
            kf.update(z);

            std::cout << std::setw(4) << (i + 1) << "  ";
            std::cout << std::setw(11) << std::fixed << std::setprecision(3) << measurements[i] << "  ";
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << kf.getStateElement(0) << "  ";
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << kf.getStateElement(1) << "\n";
        }

        printSection("Final Results");

        // Calculate and display performance metrics
        double finalPos = kf.getStateElement(0);
        double finalVel = kf.getStateElement(1);
        double lastMeasurement = measurements.back();
        double estimationError = std::abs(finalPos - lastMeasurement);

        std::cout << BOLD << "Final Estimated State:" << RESET << "\n";
        std::cout << "Position: " << std::fixed << std::setprecision(4) << finalPos << " units\n";
        std::cout << "Velocity: " << std::fixed << std::setprecision(4) << finalVel << " units/s\n\n";

        std::cout << "Performance:\n";
        std::cout << "Last measurement: " << lastMeasurement << "\n";
        std::cout << "Position error: " << std::fixed << std::setprecision(4) << estimationError << "\n";

        std::cout << "\n" << GREEN << "Processing completed successfully!" << RESET << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}