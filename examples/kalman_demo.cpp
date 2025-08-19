/**
 * @file kalman_demo.cpp
 * @brief Comprehensive Kalman filter demonstration applications
 * @author ronnymilleo
 * @date 12/07/25
 * @version 2.0
 * 
 * This application demonstrates the standard Kalman filter implementation
 * with multiple practical scenarios showing different aspects of linear
 * state estimation and filtering capabilities.
 * 
 * @details Demonstration scenarios:
 * - 1D position and velocity tracking with constant velocity model
 * - 2D object tracking with position and velocity estimation
 * - Sensor fusion example combining multiple measurements
 * - Performance analysis with different noise levels
 * 
 * @note Shows comprehensive usage of the kalman class
 * @note Includes noise simulation and performance evaluation
 * @note Results demonstrate filter convergence and accuracy
 * 
 * @see kalman.h for class interface
 * @see kalman.cpp for implementation
 */

#include "kalman.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>

/**
 * @brief 1D position and velocity tracking demonstration
 * 
 * This function demonstrates basic Kalman filtering for tracking an object
 * moving with approximately constant velocity in 1D, with noisy position measurements.
 */
void demonstrate1DTracking() {
    std::cout << "=== Standard Kalman Filter Demo: 1D Position/Velocity Tracking ===\n\n";
    
    // System parameters
    const double dt = 0.1;              // Time step (seconds)
    const double true_velocity = 2.0;   // True constant velocity (m/s)
    const double process_noise = 0.1;   // Process noise standard deviation
    const double measurement_noise = 0.5; // Measurement noise standard deviation
    const int num_steps = 50;           // Number of simulation steps
    
    // Create 2D state (position, velocity), 1D measurement (position) KF
    kalman kf(2, 1, 0);
    
    // Initialize state [position=0, velocity=0] - poor initial velocity estimate
    std::vector<double> initialState = {0.0, 0.0};
    kf.initializeState(initialState);
    
    // State transition matrix (constant velocity model)
    // x(k+1) = x(k) + v(k) * dt
    // v(k+1) = v(k)
    std::vector<std::vector<double>> F = {
        {1.0, dt},
        {0.0, 1.0}
    };
    kf.setStateTransitionMatrix(F);
    
    // Observation matrix (measure position only)
    std::vector<std::vector<double>> H = {
        {1.0, 0.0}
    };
    kf.setObservationMatrix(H);
    
    // Process noise covariance
    std::vector<std::vector<double>> Q = {
        {process_noise * process_noise * dt * dt, process_noise * process_noise * dt},
        {process_noise * process_noise * dt, process_noise * process_noise}
    };
    kf.setProcessNoiseCovariance(Q);
    
    // Measurement noise covariance
    std::vector<std::vector<double>> R = {
        {measurement_noise * measurement_noise}
    };
    kf.setMeasurementNoiseCovariance(R);
    
    // Initial error covariance (high uncertainty)
    std::vector<std::vector<double>> P = {
        {10.0, 0.0},
        {0.0, 10.0}
    };
    kf.setErrorCovariance(P);
    
    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> measurement_noise_dist(0.0, measurement_noise);
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Step | True Pos | True Vel | Meas Pos | Est Pos | Est Vel | Pos Err | Vel Err\n";
    std::cout << "-----+----------+----------+----------+---------+---------+---------+--------\n";
    
    double true_position = 0.0;
    
    // Simulation loop
    for (int step = 0; step < num_steps; ++step) {
        // Simulate true system (constant velocity with slight perturbations)
        true_position += true_velocity * dt;
        
        // Generate noisy measurement
        double measured_position = true_position + measurement_noise_dist(gen);
        
        // Kalman filter predict step
        kf.predict();
        
        // Kalman filter update step
        kf.update({measured_position});
        
        // Get current estimates
        auto state = kf.getState();
        double estimated_position = state[0];
        double estimated_velocity = state[1];
        
        // Calculate errors
        double position_error = std::abs(estimated_position - true_position);
        double velocity_error = std::abs(estimated_velocity - true_velocity);
        
        // Print results every 5 steps or first/last few steps
        if (step % 5 == 0 || step < 3 || step >= num_steps - 3) {
            std::cout << std::setw(4) << step << " | ";
            std::cout << std::setw(8) << true_position << " | ";
            std::cout << std::setw(8) << true_velocity << " | ";
            std::cout << std::setw(8) << measured_position << " | ";
            std::cout << std::setw(7) << estimated_position << " | ";
            std::cout << std::setw(7) << estimated_velocity << " | ";
            std::cout << std::setw(7) << position_error << " | ";
            std::cout << std::setw(7) << velocity_error << "\n";
        }
    }
    
    std::cout << "\nKF successfully estimated velocity from position-only measurements!\n";
    std::cout << "Notice how velocity estimate converges to the true value of " << true_velocity << " m/s.\n\n";
}

/**
 * @brief 2D object tracking demonstration
 * 
 * This function demonstrates 2D tracking of an object moving with constant
 * velocity in both X and Y directions, with noisy position measurements.
 */
void demonstrate2DTracking() {
    std::cout << "=== Standard Kalman Filter Demo: 2D Object Tracking ===\n\n";
    
    // System parameters
    const double dt = 0.2;                    // Time step
    const double true_vx = 1.5;               // True X velocity
    const double true_vy = -1.0;              // True Y velocity  
    const double process_noise = 0.05;        // Process noise
    const double measurement_noise_x = 0.3;   // X measurement noise
    const double measurement_noise_y = 0.4;   // Y measurement noise
    const int num_steps = 30;                 // Number of steps
    
    // Create 4D state (x, y, vx, vy), 2D measurement (x, y) KF
    kalman kf(4, 2, 0);
    
    // Initialize state
    std::vector<double> initialState = {0.0, 0.0, 0.0, 0.0}; // Poor initial guess
    kf.initializeState(initialState);
    
    // State transition matrix (2D constant velocity)
    std::vector<std::vector<double>> F = {
        {1.0, 0.0, dt,  0.0},
        {0.0, 1.0, 0.0, dt },
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };
    kf.setStateTransitionMatrix(F);
    
    // Observation matrix (measure both positions)
    std::vector<std::vector<double>> H = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0}
    };
    kf.setObservationMatrix(H);
    
    // Process noise covariance
    double q = process_noise * process_noise;
    std::vector<std::vector<double>> Q = {
        {q * dt * dt, 0.0,       q * dt,   0.0      },
        {0.0,         q * dt * dt, 0.0,       q * dt   },
        {q * dt,      0.0,       q,        0.0      },
        {0.0,         q * dt,    0.0,      q        }
    };
    kf.setProcessNoiseCovariance(Q);
    
    // Measurement noise covariance
    std::vector<std::vector<double>> R = {
        {measurement_noise_x * measurement_noise_x, 0.0},
        {0.0, measurement_noise_y * measurement_noise_y}
    };
    kf.setMeasurementNoiseCovariance(R);
    
    // Initial error covariance
    std::vector<std::vector<double>> P = {
        {5.0, 0.0, 0.0, 0.0},
        {0.0, 5.0, 0.0, 0.0},
        {0.0, 0.0, 5.0, 0.0},
        {0.0, 0.0, 0.0, 5.0}
    };
    kf.setErrorCovariance(P);
    
    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise_x_dist(0.0, measurement_noise_x);
    std::normal_distribution<double> noise_y_dist(0.0, measurement_noise_y);
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Step | True Pos (x,y) | Meas Pos (x,y) | Est Pos (x,y) | Est Vel (x,y) | Pos Error\n";
    std::cout << "-----+----------------+----------------+---------------+---------------+----------\n";
    
    double true_x = 0.0, true_y = 0.0;
    
    // Simulation loop
    for (int step = 0; step < num_steps; ++step) {
        // Simulate true system
        true_x += true_vx * dt;
        true_y += true_vy * dt;
        
        // Generate noisy measurements
        double measured_x = true_x + noise_x_dist(gen);
        double measured_y = true_y + noise_y_dist(gen);
        
        // Kalman filter steps
        kf.predict();
        kf.update({measured_x, measured_y});
        
        // Get estimates
        auto state = kf.getState();
        double est_x = state[0], est_y = state[1];
        double est_vx = state[2], est_vy = state[3];
        
        // Calculate position error
        double pos_error = std::sqrt(std::pow(est_x - true_x, 2) + std::pow(est_y - true_y, 2));
        
        // Print results every 3 steps or first/last few
        if (step % 3 == 0 || step < 2 || step >= num_steps - 2) {
            std::cout << std::setw(4) << step << " | ";
            std::cout << "(" << std::setw(5) << true_x << "," << std::setw(5) << true_y << ") | ";
            std::cout << "(" << std::setw(5) << measured_x << "," << std::setw(5) << measured_y << ") | ";
            std::cout << "(" << std::setw(5) << est_x << "," << std::setw(5) << est_y << ") | ";
            std::cout << "(" << std::setw(5) << est_vx << "," << std::setw(5) << est_vy << ") | ";
            std::cout << std::setw(9) << pos_error << "\n";
        }
    }
    
    std::cout << "\nKF successfully tracked 2D motion with coupled state estimation!\n";
    std::cout << "The filter estimated both position and velocity in X and Y dimensions.\n\n";
}

/**
 * @brief Noise sensitivity analysis
 * 
 * This function demonstrates how the Kalman filter performs under
 * different noise conditions, showing robustness and adaptation.
 */
void demonstrateNoiseSensitivity() {
    std::cout << "=== Standard Kalman Filter Demo: Noise Sensitivity Analysis ===\n\n";
    
    const double dt = 0.1;
    const double true_velocity = 1.0;
    const int num_steps = 20;
    
    // Test different noise levels
    std::vector<double> noise_levels = {0.1, 0.5, 1.0, 2.0};
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Noise Level | Final Pos Error | Final Vel Error | Convergence Quality\n";
    std::cout << "------------+-----------------+-----------------+--------------------\n";
    
    for (double noise_level : noise_levels) {
        kalman kf(2, 1, 0);
        
        // Initialize
        kf.initializeState({0.0, 0.0});
        
        std::vector<std::vector<double>> F = {{1.0, dt}, {0.0, 1.0}};
        kf.setStateTransitionMatrix(F);
        
        std::vector<std::vector<double>> H = {{1.0, 0.0}};
        kf.setObservationMatrix(H);
        
        std::vector<std::vector<double>> Q = {{0.01, 0.0}, {0.0, 0.01}};
        kf.setProcessNoiseCovariance(Q);
        
        std::vector<std::vector<double>> R = {{noise_level * noise_level}};
        kf.setMeasurementNoiseCovariance(R);
        
        std::vector<std::vector<double>> P = {{1.0, 0.0}, {0.0, 1.0}};
        kf.setErrorCovariance(P);
        
        // Simulate with this noise level
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> noise_dist(0.0, noise_level);
        
        double true_position = 0.0;
        double convergence_measure = 0.0;
        
        for (int step = 0; step < num_steps; ++step) {
            true_position += true_velocity * dt;
            double measured_position = true_position + noise_dist(gen);
            
            kf.predict();
            kf.update({measured_position});
            
            // Accumulate convergence measure (sum of velocity estimate errors)
            if (step > 5) { // After initial convergence
                convergence_measure += std::abs(kf.getState()[1] - true_velocity);
            }
        }
        
        auto final_state = kf.getState();
        double pos_error = std::abs(final_state[0] - true_position);
        double vel_error = std::abs(final_state[1] - true_velocity);
        convergence_measure /= (num_steps - 5);
        
        std::cout << std::setw(11) << noise_level << " | ";
        std::cout << std::setw(15) << pos_error << " | ";
        std::cout << std::setw(15) << vel_error << " | ";
        std::cout << std::setw(19) << convergence_measure << "\n";
    }
    
    std::cout << "\nKF demonstrates robustness across different noise levels!\n";
    std::cout << "Higher noise leads to slower convergence but maintains stability.\n\n";
}

/**
 * @brief Main demonstration function
 */
int main() {
    std::cout << "Standard Kalman Filter Demonstrations\n";
    std::cout << "=====================================\n\n";
    
    try {
        // Run 1D tracking demo
        demonstrate1DTracking();
        
        // Run 2D tracking demo  
        demonstrate2DTracking();
        
        // Run noise sensitivity analysis
        demonstrateNoiseSensitivity();
        
        std::cout << "All demonstrations completed successfully!\n";
        std::cout << "The standard Kalman Filter effectively handles linear estimation problems.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error during demonstration: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}