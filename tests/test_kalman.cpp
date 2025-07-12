/**
 * @file test_kalman.cpp
 * @brief Comprehensive unit tests for the Kalman filter class
 * @author ronnymilleo
 * @date 12/07/25
 * @version 1.0
 * 
 * This file contains comprehensive unit tests for the standard Kalman filter
 * implementation, covering all public methods, edge cases, error conditions,
 * and numerical accuracy scenarios.
 * 
 * @details Test categories:
 * - Constructor and initialization tests
 * - Matrix setup and validation tests
 * - Filter operation tests (predict/update)
 * - State and covariance getter tests
 * - Error condition and exception tests
 * - Numerical accuracy and stability tests
 * - Complete filtering scenarios
 * 
 * @note Uses Google Test framework for comprehensive testing
 * @note Includes both unit tests and integration tests
 * @note Tests cover edge cases and error conditions
 * 
 * @see kalman.h for API documentation
 * @see matrix_math.h for matrix operations
 */

#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
#include <cmath>
#include "../kalman.h"

/**
 * @brief Test fixture for Kalman filter tests
 * 
 * Provides common setup and utility methods for Kalman filter testing.
 * Includes helper methods for matrix comparison and test data generation.
 */
class KalmanTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a basic 2D state, 1D measurement filter for most tests
        kf = std::make_unique<kalman>(2, 1, 0);
        
        // Standard test matrices for 2D state (position, velocity)
        F = {{1.0, 1.0}, {0.0, 1.0}};  // Constant velocity model
        H = {{1.0, 0.0}};               // Observe position only
        Q = {{0.1, 0.0}, {0.0, 0.1}};  // Process noise
        R = {{0.5}};                    // Measurement noise
        P = {{1.0, 0.0}, {0.0, 1.0}};  // Initial error covariance
        initialState = {0.0, 0.0};      // Initial state [pos, vel]
    }

    void TearDown() override {
        kf.reset();
    }

    /**
     * @brief Compare two matrices with tolerance
     */
    bool matricesEqual(const std::vector<std::vector<double>>& a,
                      const std::vector<std::vector<double>>& b,
                      double tolerance = 1e-10) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].size() != b[i].size()) return false;
            for (size_t j = 0; j < a[i].size(); ++j) {
                if (std::abs(a[i][j] - b[i][j]) > tolerance) return false;
            }
        }
        return true;
    }

    /**
     * @brief Compare two vectors with tolerance
     */
    bool vectorsEqual(const std::vector<double>& a,
                     const std::vector<double>& b,
                     double tolerance = 1e-10) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > tolerance) return false;
        }
        return true;
    }

    /**
     * @brief Initialize filter with standard test setup
     */
    void initializeStandardFilter() {
        kf->initializeState(initialState);
        kf->setStateTransitionMatrix(F);
        kf->setObservationMatrix(H);
        kf->setProcessNoiseCovariance(Q);
        kf->setMeasurementNoiseCovariance(R);
        kf->setErrorCovariance(P);
    }

    std::unique_ptr<kalman> kf;
    std::vector<std::vector<double>> F, H, Q, R, P;
    std::vector<double> initialState;
};

// =============================================================================
// Constructor and Initialization Tests
// =============================================================================

TEST_F(KalmanTest, DefaultConstructor) {
    kalman default_kf;
    EXPECT_FALSE(default_kf.isInitialized());
}

TEST_F(KalmanTest, ParameterizedConstructor) {
    kalman param_kf(3, 2, 1);
    EXPECT_FALSE(param_kf.isInitialized());
}

TEST_F(KalmanTest, ConstructorWithZeroDimensions) {
    // Constructor should now validate dimensions and throw for zero values
    EXPECT_THROW(kalman zero_kf1(0, 1, 0), std::invalid_argument);
    EXPECT_THROW(kalman zero_kf2(1, 0, 0), std::invalid_argument);
    
    // Both zero should also throw
    EXPECT_THROW(kalman zero_kf3(0, 0, 0), std::invalid_argument);
}

TEST_F(KalmanTest, StateInitialization) {
    std::vector<double> state = {1.5, -2.3};
    kf->initializeState(state);
    
    auto retrieved_state = kf->getState();
    EXPECT_TRUE(vectorsEqual(state, retrieved_state));
}

TEST_F(KalmanTest, StateInitializationWrongSize) {
    std::vector<double> wrong_state = {1.0, 2.0, 3.0}; // 3D state for 2D filter
    EXPECT_THROW(kf->initializeState(wrong_state), std::invalid_argument);
}

TEST_F(KalmanTest, InitializationStatus) {
    EXPECT_FALSE(kf->isInitialized());
    
    initializeStandardFilter();
    EXPECT_TRUE(kf->isInitialized());
}

// =============================================================================
// Matrix Setup and Validation Tests
// =============================================================================

TEST_F(KalmanTest, StateTransitionMatrixSetup) {
    kf->setStateTransitionMatrix(F);
    // No direct getter, but we can test indirectly through prediction
    EXPECT_NO_THROW(kf->setStateTransitionMatrix(F));
}

TEST_F(KalmanTest, StateTransitionMatrixWrongSize) {
    std::vector<std::vector<double>> wrong_F = {{1.0}};  // 1x1 matrix for 2D state
    EXPECT_THROW(kf->setStateTransitionMatrix(wrong_F), std::invalid_argument);
}

TEST_F(KalmanTest, ObservationMatrixSetup) {
    kf->setObservationMatrix(H);
    EXPECT_NO_THROW(kf->setObservationMatrix(H));
}

TEST_F(KalmanTest, ObservationMatrixWrongSize) {
    std::vector<std::vector<double>> wrong_H = {{1.0, 0.0, 0.0}};  // Wrong dimensions
    EXPECT_THROW(kf->setObservationMatrix(wrong_H), std::invalid_argument);
}

TEST_F(KalmanTest, ControlMatrixSetup) {
    // Create filter with control dimension > 0
    kalman control_kf(2, 1, 1);
    std::vector<std::vector<double>> B = {{0.5}, {1.0}};  // 2x1 control matrix
    control_kf.setControlMatrix(B);
    EXPECT_NO_THROW(control_kf.setControlMatrix(B));
}

TEST_F(KalmanTest, ProcessNoiseCovarianceSetup) {
    kf->setProcessNoiseCovariance(Q);
    EXPECT_NO_THROW(kf->setProcessNoiseCovariance(Q));
}

TEST_F(KalmanTest, ProcessNoiseCovarianceWrongSize) {
    std::vector<std::vector<double>> wrong_Q = {{1.0}};  // 1x1 matrix for 2D state
    EXPECT_THROW(kf->setProcessNoiseCovariance(wrong_Q), std::invalid_argument);
}

TEST_F(KalmanTest, MeasurementNoiseCovarianceSetup) {
    kf->setMeasurementNoiseCovariance(R);
    EXPECT_NO_THROW(kf->setMeasurementNoiseCovariance(R));
}

TEST_F(KalmanTest, MeasurementNoiseCovarianceWrongSize) {
    std::vector<std::vector<double>> wrong_R = {{1.0, 0.0}, {0.0, 1.0}};  // 2x2 for 1D measurement
    EXPECT_THROW(kf->setMeasurementNoiseCovariance(wrong_R), std::invalid_argument);
}

TEST_F(KalmanTest, ErrorCovarianceSetup) {
    kf->setErrorCovariance(P);
    auto retrieved_P = kf->getErrorCovariance();
    EXPECT_TRUE(matricesEqual(P, retrieved_P));
}

TEST_F(KalmanTest, ErrorCovarianceWrongSize) {
    std::vector<std::vector<double>> wrong_P = {{1.0}};  // 1x1 matrix for 2D state
    EXPECT_THROW(kf->setErrorCovariance(wrong_P), std::invalid_argument);
}

// =============================================================================
// Filter Operation Tests
// =============================================================================

TEST_F(KalmanTest, PredictBeforeInitialization) {
    EXPECT_THROW(kf->predict(), std::runtime_error);
}

TEST_F(KalmanTest, UpdateBeforeInitialization) {
    std::vector<double> measurement = {1.0};
    EXPECT_THROW(kf->update(measurement), std::runtime_error);
}

TEST_F(KalmanTest, BasicPredictStep) {
    initializeStandardFilter();
    
    auto initial_state = kf->getState();
    kf->predict();
    auto predicted_state = kf->getState();
    
    // For constant velocity model: new_pos = old_pos + velocity * dt
    // With dt = 1.0 and initial velocity = 0, position should remain 0
    EXPECT_DOUBLE_EQ(predicted_state[0], 0.0);  // position
    EXPECT_DOUBLE_EQ(predicted_state[1], 0.0);  // velocity
}

TEST_F(KalmanTest, BasicUpdateStep) {
    initializeStandardFilter();
    
    std::vector<double> measurement = {2.5};
    kf->update(measurement);
    
    auto updated_state = kf->getState();
    // State should move towards measurement
    EXPECT_GT(updated_state[0], 0.0);  // position should increase towards measurement
}

TEST_F(KalmanTest, PredictUpdateCycle) {
    initializeStandardFilter();
    
    // Initial state: [0, 0]
    auto initial_state = kf->getState();
    EXPECT_TRUE(vectorsEqual(initial_state, {0.0, 0.0}));
    
    // Predict step
    kf->predict();
    auto predicted_state = kf->getState();
    
    // Update with measurement
    std::vector<double> measurement = {1.0};
    kf->update(measurement);
    auto updated_state = kf->getState();
    
    // Position should be between 0 and 1 (due to Kalman gain)
    EXPECT_GT(updated_state[0], 0.0);
    EXPECT_LT(updated_state[0], 1.0);
}

TEST_F(KalmanTest, UpdateWithWrongMeasurementSize) {
    initializeStandardFilter();
    
    std::vector<double> wrong_measurement = {1.0, 2.0};  // 2D measurement for 1D observation
    EXPECT_THROW(kf->update(wrong_measurement), std::invalid_argument);
}

TEST_F(KalmanTest, MultipleFilterCycles) {
    initializeStandardFilter();
    
    std::vector<double> measurements = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    for (const auto& measurement : measurements) {
        kf->predict();
        kf->update({measurement});
    }
    
    auto final_state = kf->getState();
    
    // After filtering, position should be close to last measurement
    // and velocity should be estimated as positive
    EXPECT_GT(final_state[0], 0.0);
    EXPECT_GT(final_state[1], 0.0);  // Should estimate positive velocity
}

// =============================================================================
// State and Covariance Getter Tests
// =============================================================================

TEST_F(KalmanTest, GetStateBeforeInitialization) {
    // Note: Current implementation allows getState() without initialization
    // This test documents the current behavior
    EXPECT_NO_THROW(kf->getState());
}

TEST_F(KalmanTest, GetStateElementBeforeInitialization) {
    // Note: Current implementation allows getStateElement() without initialization
    // This test documents the current behavior
    EXPECT_NO_THROW(kf->getStateElement(0));
}

TEST_F(KalmanTest, GetStateElement) {
    initializeStandardFilter();
    
    // Set a specific state
    std::vector<double> test_state = {3.14, -2.71};
    kf->initializeState(test_state);
    
    EXPECT_DOUBLE_EQ(kf->getStateElement(0), 3.14);
    EXPECT_DOUBLE_EQ(kf->getStateElement(1), -2.71);
}

TEST_F(KalmanTest, GetStateElementOutOfBounds) {
    initializeStandardFilter();
    
    EXPECT_THROW(kf->getStateElement(2), std::out_of_range);
    // Test large index instead of negative to avoid signedness warning
    EXPECT_THROW(kf->getStateElement(999), std::out_of_range);
}

TEST_F(KalmanTest, GetErrorCovariance) {
    initializeStandardFilter();
    
    auto retrieved_P = kf->getErrorCovariance();
    EXPECT_TRUE(matricesEqual(P, retrieved_P));
}

TEST_F(KalmanTest, ErrorCovarianceEvolution) {
    initializeStandardFilter();
    
    auto initial_P = kf->getErrorCovariance();
    
    // Predict step should increase uncertainty
    kf->predict();
    auto predicted_P = kf->getErrorCovariance();
    EXPECT_GT(predicted_P[0][0], initial_P[0][0]);  // Position uncertainty increases
    
    // Update step should decrease uncertainty
    kf->update({1.0});
    auto updated_P = kf->getErrorCovariance();
    EXPECT_LT(updated_P[0][0], predicted_P[0][0]);  // Position uncertainty decreases
}

// =============================================================================
// Reset and State Management Tests
// =============================================================================

TEST_F(KalmanTest, ResetFilter) {
    initializeStandardFilter();
    EXPECT_TRUE(kf->isInitialized());
    
    kf->reset();
    EXPECT_FALSE(kf->isInitialized());
    
    // Operations should still work after reset (current behavior)
    EXPECT_NO_THROW(kf->getState());
}

TEST_F(KalmanTest, ResetAndReinitialize) {
    initializeStandardFilter();
    
    // Perform some operations
    kf->predict();
    kf->update({2.0});
    
    // Reset and reinitialize
    kf->reset();
    EXPECT_FALSE(kf->isInitialized());
    
    initializeStandardFilter();
    EXPECT_TRUE(kf->isInitialized());
    
    // Should work normally after reinitialization
    EXPECT_NO_THROW(kf->predict());
    EXPECT_NO_THROW(kf->update({1.0}));
}

// =============================================================================
// Numerical Accuracy and Stability Tests
// =============================================================================

TEST_F(KalmanTest, NumericalStabilityWithSmallValues) {
    kalman small_kf(2, 1, 0);
    
    // Very small but valid matrices
    std::vector<std::vector<double>> small_F = {{1.0, 1e-6}, {0.0, 1.0}};
    std::vector<std::vector<double>> small_H = {{1.0, 0.0}};
    std::vector<std::vector<double>> small_Q = {{1e-10, 0.0}, {0.0, 1e-10}};
    std::vector<std::vector<double>> small_R = {{1e-8}};
    std::vector<std::vector<double>> small_P = {{1e-6, 0.0}, {0.0, 1e-6}};
    
    small_kf.initializeState({0.0, 0.0});
    small_kf.setStateTransitionMatrix(small_F);
    small_kf.setObservationMatrix(small_H);
    small_kf.setProcessNoiseCovariance(small_Q);
    small_kf.setMeasurementNoiseCovariance(small_R);
    small_kf.setErrorCovariance(small_P);
    
    // Should handle small values without issues
    EXPECT_NO_THROW(small_kf.predict());
    EXPECT_NO_THROW(small_kf.update({1e-6}));
}

TEST_F(KalmanTest, HighPrecisionFiltering) {
    initializeStandardFilter();
    
    // Test with high precision values
    double precise_measurement = 3.141592653589793;
    kf->update({precise_measurement});
    
    auto state = kf->getState();
    // With the default covariance values, the Kalman gain will balance
    // between the prior (0.0) and measurement, so expect value between them
    EXPECT_GT(state[0], 0.0);  // Should be positive (towards measurement)
    EXPECT_LT(state[0], precise_measurement);  // But less than full measurement due to uncertainty
}

// =============================================================================
// Integration Tests - Complete Scenarios
// =============================================================================

TEST_F(KalmanTest, ConstantVelocityTracking) {
    initializeStandardFilter();
    
    // Simulate object moving with constant velocity
    double true_velocity = 2.0;
    double dt = 0.1;
    std::vector<double> true_positions;
    std::vector<double> measurements;
    
    // Generate truth and noisy measurements
    for (int i = 0; i < 20; ++i) {
        double true_pos = i * dt * true_velocity;
        double noisy_measurement = true_pos + 0.1 * (rand() / double(RAND_MAX) - 0.5);
        true_positions.push_back(true_pos);
        measurements.push_back(noisy_measurement);
    }
    
    // Update time step in state transition matrix
    F[0][1] = dt;
    kf->setStateTransitionMatrix(F);
    
    // Filter the measurements
    for (const auto& measurement : measurements) {
        kf->predict();
        kf->update({measurement});
    }
    
    auto final_state = kf->getState();
    
    // Should estimate velocity close to true value
    EXPECT_NEAR(final_state[1], true_velocity, 0.5);
    // Position should be close to last true position
    EXPECT_NEAR(final_state[0], true_positions.back(), 1.0);
}

TEST_F(KalmanTest, ZeroMeasurementNoise) {
    initializeStandardFilter();
    
    // Set measurement noise to near zero
    std::vector<std::vector<double>> zero_R = {{1e-10}};
    kf->setMeasurementNoiseCovariance(zero_R);
    
    double measurement = 5.0;
    kf->update({measurement});
    
    auto state = kf->getState();
    
    // With very low measurement noise, state should be very close to measurement
    EXPECT_NEAR(state[0], measurement, 1e-6);
}

TEST_F(KalmanTest, HighMeasurementNoise) {
    initializeStandardFilter();
    
    // Set very high measurement noise
    std::vector<std::vector<double>> high_R = {{100.0}};
    kf->setMeasurementNoiseCovariance(high_R);
    
    double measurement = 10.0;
    kf->update({measurement});
    
    auto state = kf->getState();
    
    // With high measurement noise, state should remain close to initial value
    EXPECT_LT(std::abs(state[0]), 2.0);  // Should not move much from initial 0
}

// =============================================================================
// Edge Cases and Error Conditions
// =============================================================================

TEST_F(KalmanTest, IdentityMatrices) {
    kalman identity_kf(2, 2, 0);
    
    // Set up with identity matrices
    std::vector<std::vector<double>> I = {{1.0, 0.0}, {0.0, 1.0}};
    
    identity_kf.initializeState({1.0, 2.0});
    identity_kf.setStateTransitionMatrix(I);
    identity_kf.setObservationMatrix(I);
    identity_kf.setProcessNoiseCovariance(I);
    identity_kf.setMeasurementNoiseCovariance(I);
    identity_kf.setErrorCovariance(I);
    
    // Should work with identity matrices
    EXPECT_NO_THROW(identity_kf.predict());
    EXPECT_NO_THROW(identity_kf.update({1.5, 2.5}));
}

TEST_F(KalmanTest, LargeStateVector) {
    // Test with larger state vector
    kalman large_kf(10, 3, 0);
    
    std::vector<double> large_state(10, 1.0);
    large_kf.initializeState(large_state);
    
    auto retrieved_state = large_kf.getState();
    EXPECT_EQ(retrieved_state.size(), 10);
    EXPECT_TRUE(vectorsEqual(large_state, retrieved_state));
}

/**
 * @brief Main test runner
 */
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
