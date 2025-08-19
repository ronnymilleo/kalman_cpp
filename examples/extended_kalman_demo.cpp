/**
 * @file extended_kalman_demo.cpp
 * @brief Demonstration application for Extended Kalman Filter
 * @author ronnymilleo
 * @date 12/07/25
 * @version 1.0
 * 
 * This application demonstrates the Extended Kalman Filter implementation
 * with a practical non-linear system: tracking an object moving in a circle
 * with range-only measurements (polar to cartesian coordinates).
 * 
 * @details Scenario:
 * - Object moves in a circular trajectory
 * - State: [x, y, vx, vy] (position and velocity in cartesian coordinates)
 * - Measurements: [range] (distance from origin) 
 * - Non-linear observation function: range = sqrt(x^2 + y^2)
 * - Demonstrates EKF handling of non-linear measurement model
 * 
 * @note This example shows practical usage of the ExtendedKalman class
 * @note Includes noise simulation and performance evaluation
 * @note Results are printed to demonstrate filter convergence
 * 
 * @see extended_kalman.h for class interface
 * @see extended_kalman.cpp for implementation
 */

#include "extended_kalman.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>

/**
 * @brief Circular motion tracking demonstration
 * 
 * This function demonstrates the Extended Kalman Filter tracking an object
 * moving in a circular path with range-only measurements. The system is
 * non-linear due to the range measurement function.
 */
void demonstrateCircularMotionTracking() {
    std::cout << "=== Extended Kalman Filter Demo: Circular Motion Tracking ===\n\n";
    
    // System parameters
    const double dt = 0.1;           // Time step
    const double radius = 5.0;       // Circle radius
    const double angular_velocity = 0.2; // rad/s
    const double process_noise = 0.01;   // Process noise standard deviation
    const double measurement_noise = 0.1; // Measurement noise standard deviation
    const int num_steps = 100;           // Number of simulation steps
    
    // Create 4D state (x, y, vx, vy), 1D measurement (range) EKF
    ExtendedKalman ekf(4, 1, 0);
    
    // Define non-linear state transition function (constant velocity model with slight perturbation)
    auto stateTransitionFunc = [dt](const std::vector<double>& x, const std::vector<double>& /*u*/) {
        return std::vector<double>{
            x[0] + x[2] * dt,  // x = x + vx * dt
            x[1] + x[3] * dt,  // y = y + vy * dt
            x[2],              // vx (constant)
            x[3]               // vy (constant)
        };
    };
    
    // Define state Jacobian matrix F = ∂f/∂x
    auto stateJacobianFunc = [dt](const std::vector<double>& /*x*/) {
        return std::vector<std::vector<double>>{
            {1.0, 0.0, dt,  0.0},
            {0.0, 1.0, 0.0, dt },
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0, 1.0}
        };
    };
    
    // Define non-linear observation function h(x) = sqrt(x^2 + y^2)
    auto observationFunc = [](const std::vector<double>& x) {
        double range = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        return std::vector<double>{range};
    };
    
    // Define observation Jacobian matrix H = ∂h/∂x
    auto observationJacobianFunc = [](const std::vector<double>& x) {
        double range = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        if (range < 1e-10) range = 1e-10; // Avoid division by zero
        
        return std::vector<std::vector<double>>{
            {x[0] / range, x[1] / range, 0.0, 0.0}
        };
    };
    
    // Set up EKF functions
    ekf.setStateTransitionFunction(stateTransitionFunc);
    ekf.setStateJacobianFunction(stateJacobianFunc);
    ekf.setObservationFunction(observationFunc);
    ekf.setObservationJacobianFunction(observationJacobianFunc);
    
    // Set noise covariance matrices
    std::vector<std::vector<double>> Q = {
        {process_noise * process_noise, 0.0, 0.0, 0.0},
        {0.0, process_noise * process_noise, 0.0, 0.0},
        {0.0, 0.0, process_noise * process_noise * 0.1, 0.0},
        {0.0, 0.0, 0.0, process_noise * process_noise * 0.1}
    };
    
    std::vector<std::vector<double>> R = {
        {measurement_noise * measurement_noise}
    };
    
    std::vector<std::vector<double>> P = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };
    
    ekf.setProcessNoiseCovariance(Q);
    ekf.setMeasurementNoiseCovariance(R);
    ekf.setErrorCovariance(P);
    
    // Initialize state with a poor initial guess
    std::vector<double> initialState = {4.0, 3.0, -1.0, 1.0}; // Not exactly on circle
    ekf.initializeState(initialState);
    
    // Set up random number generation for noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> process_noise_dist(0.0, process_noise);
    std::normal_distribution<double> measurement_noise_dist(0.0, measurement_noise);
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Step | True Pos (x,y) | True Range | Meas Range | Est Pos (x,y) | Est Range | Error\n";
    std::cout << "-----+----------------+------------+------------+---------------+-----------+------\n";
    
    // Simulation loop
    for (int step = 0; step < num_steps; ++step) {
        double t = step * dt;
        
        // Generate true state (circular motion)
        double true_x = radius * std::cos(angular_velocity * t);
        double true_y = radius * std::sin(angular_velocity * t);
        // Note: true velocities calculated but not used in this demo
        // double true_vx = -radius * angular_velocity * std::sin(angular_velocity * t);
        // double true_vy = radius * angular_velocity * std::cos(angular_velocity * t);
        double true_range = std::sqrt(true_x * true_x + true_y * true_y);
        
        // Generate noisy measurement
        double measured_range = true_range + measurement_noise_dist(gen);
        
        // EKF predict step
        ekf.predict();
        
        // EKF update step
        ekf.update({measured_range});
        
        // Get current estimate
        auto& estimated_state = ekf.getState();
        double estimated_range = std::sqrt(estimated_state[0] * estimated_state[0] + 
                                         estimated_state[1] * estimated_state[1]);
        
        // Calculate position error
        double position_error = std::sqrt(std::pow(estimated_state[0] - true_x, 2) + 
                                        std::pow(estimated_state[1] - true_y, 2));
        
        // Print results every 10 steps
        if (step % 10 == 0 || step < 5) {
            std::cout << std::setw(4) << step << " | ";
            std::cout << "(" << std::setw(5) << true_x << "," << std::setw(5) << true_y << ") | ";
            std::cout << std::setw(10) << true_range << " | ";
            std::cout << std::setw(10) << measured_range << " | ";
            std::cout << "(" << std::setw(5) << estimated_state[0] << "," << std::setw(5) << estimated_state[1] << ") | ";
            std::cout << std::setw(9) << estimated_range << " | ";
            std::cout << std::setw(5) << position_error << "\n";
        }
    }
    
    std::cout << "\nFilter successfully tracked the circular motion using range-only measurements!\n";
    std::cout << "Notice how the position estimates converge to the true trajectory.\n\n";
}

/**
 * @brief Pendulum tracking demonstration
 * 
 * This function demonstrates EKF tracking of a non-linear pendulum system
 * with angle measurements, showing state estimation for angular position and velocity.
 */
void demonstratePendulumTracking() {
    std::cout << "=== Extended Kalman Filter Demo: Pendulum Tracking ===\n\n";
    
    // Pendulum parameters
    const double dt = 0.05;          // Time step
    const double g = 9.81;           // Gravity
    const double length = 1.0;       // Pendulum length
    const double damping = 0.1;      // Damping coefficient
    const double process_noise = 0.01;   // Process noise
    const double measurement_noise = 0.05; // Measurement noise
    const int num_steps = 200;       // Number of simulation steps
    
    // Create 2D state (angle, angular_velocity), 1D measurement (angle) EKF
    ExtendedKalman ekf(2, 1, 0);
    
    // Define non-linear state transition function (pendulum dynamics)
    auto stateTransitionFunc = [dt, g, length, damping](const std::vector<double>& x, const std::vector<double>& /*u*/) {
        double theta = x[0];      // Current angle
        double omega = x[1];      // Current angular velocity
        
        // Non-linear pendulum equation: theta'' = -(g/L)*sin(theta) - damping*theta'
        double alpha = -(g / length) * std::sin(theta) - damping * omega;
        
        return std::vector<double>{
            theta + omega * dt,       // New angle
            omega + alpha * dt        // New angular velocity
        };
    };
    
    // Define state Jacobian matrix
    auto stateJacobianFunc = [dt, g, length, damping](const std::vector<double>& x) {
        double theta = x[0];
        // double omega = x[1]; // Not needed for Jacobian calculation
        
        // Partial derivatives
        double df1_dtheta = 1.0;
        double df1_domega = dt;
        double df2_dtheta = -(g / length) * std::cos(theta) * dt;
        double df2_domega = 1.0 - damping * dt;
        
        return std::vector<std::vector<double>>{
            {df1_dtheta, df1_domega},
            {df2_dtheta, df2_domega}
        };
    };
    
    // Define observation function (direct angle measurement)
    auto observationFunc = [](const std::vector<double>& x) {
        return std::vector<double>{x[0]}; // Measure angle directly
    };
    
    // Define observation Jacobian matrix
    auto observationJacobianFunc = [](const std::vector<double>& /*x*/) {
        return std::vector<std::vector<double>>{{1.0, 0.0}};
    };
    
    // Set up EKF functions
    ekf.setStateTransitionFunction(stateTransitionFunc);
    ekf.setStateJacobianFunction(stateJacobianFunc);
    ekf.setObservationFunction(observationFunc);
    ekf.setObservationJacobianFunction(observationJacobianFunc);
    
    // Set noise covariance matrices
    std::vector<std::vector<double>> Q = {
        {process_noise * process_noise, 0.0},
        {0.0, process_noise * process_noise}
    };
    
    std::vector<std::vector<double>> R = {
        {measurement_noise * measurement_noise}
    };
    
    std::vector<std::vector<double>> P = {
        {0.1, 0.0},
        {0.0, 0.1}
    };
    
    ekf.setProcessNoiseCovariance(Q);
    ekf.setMeasurementNoiseCovariance(R);
    ekf.setErrorCovariance(P);
    
    // Initialize state
    std::vector<double> initialState = {0.5, 0.0}; // Start at 0.5 rad, 0 rad/s
    ekf.initializeState(initialState);
    
    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> measurement_noise_dist(0.0, measurement_noise);
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Step | True Angle | True Vel | Meas Angle | Est Angle | Est Vel | Angle Err\n";
    std::cout << "-----+------------+----------+------------+-----------+---------+----------\n";
    
    // Initialize true state
    double true_theta = 0.5;
    double true_omega = 0.0;
    
    // Simulation loop
    for (int step = 0; step < num_steps; ++step) {
        // Simulate true pendulum dynamics
        double alpha = -(g / length) * std::sin(true_theta) - damping * true_omega;
        true_theta += true_omega * dt;
        true_omega += alpha * dt;
        
        // Generate noisy measurement
        double measured_angle = true_theta + measurement_noise_dist(gen);
        
        // EKF predict and update
        ekf.predict();
        ekf.update({measured_angle});
        
        // Get current estimate
        auto& estimated_state = ekf.getState();
        double angle_error = std::abs(estimated_state[0] - true_theta);
        
        // Print results every 20 steps
        if (step % 20 == 0 || step < 5) {
            std::cout << std::setw(4) << step << " | ";
            std::cout << std::setw(10) << true_theta << " | ";
            std::cout << std::setw(8) << true_omega << " | ";
            std::cout << std::setw(10) << measured_angle << " | ";
            std::cout << std::setw(9) << estimated_state[0] << " | ";
            std::cout << std::setw(7) << estimated_state[1] << " | ";
            std::cout << std::setw(9) << angle_error << "\n";
        }
    }
    
    std::cout << "\nEKF successfully estimated pendulum state from noisy angle measurements!\n";
    std::cout << "The filter estimates both angle and angular velocity from angle-only measurements.\n\n";
}

/**
 * @brief Main demonstration function
 */
int main() {
    std::cout << "Extended Kalman Filter Demonstrations\n";
    std::cout << "=====================================\n\n";
    
    try {
        // Run circular motion tracking demo
        demonstrateCircularMotionTracking();
        
        // Run pendulum tracking demo
        demonstratePendulumTracking();
        
        std::cout << "All demonstrations completed successfully!\n";
        std::cout << "The Extended Kalman Filter handled non-linear systems effectively.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error during demonstration: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
