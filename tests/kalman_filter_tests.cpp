/**
 * @file    kalman_filter_tests.cpp
 * @brief   Unit tests for the linear Kalman filter, from setup to tracking a moving object.
 */

#include "filters/kalman_filter.h"

#include "test_support.h"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {

constexpr double Tolerance = 1e-10;
constexpr double TimeStep = 0.1;

// Position and velocity tracker with the state (1.0, 0.5), the measurement noise 0.1 and an identity covariance
Core::KalmanFilter PositionVelocityFilter() {
    Tests::Model model = Tests::PositionVelocityModel(TimeStep, 0.1, 1.0);
    model.InitialState = {1.0, 0.5};
    return Tests::StartedFilter(model);
}

// What is left of a tracking run: how far the final estimate is from the truth
struct TrackingError {
    double Position;
    double Velocity;
};

// Follows an object that moves at a constant velocity along one axis, measured with uniform noise
TrackingError Track1d(Core::KalmanFilter &filter, const double true_velocity, const int steps,
                      const double noise_amplitude, const std::uint32_t seed) {
    Tests::NoiseSource noise(seed);
    double true_position = 0.0;
    for (int step = 0; step < steps; step++) {
        true_position += true_velocity * TimeStep;
        EXPECT_TRUE(Tests::Succeeded(filter.Predict()));
        EXPECT_TRUE(Tests::Succeeded(filter.Update({true_position + noise.Symmetric(noise_amplitude)})));
    }
    const Core::Vector &state = filter.GetState();
    return {std::abs(state[0] - true_position), std::abs(state[1] - true_velocity)};
}

// Follows an object that moves at a constant velocity on a plane, measured with uniform noise on each axis
TrackingError Track2d(Core::KalmanFilter &filter, const double true_vx, const double true_vy, const int steps,
                      const double noise_amplitude, const std::uint32_t seed) {
    Tests::NoiseSource noise(seed);
    double true_x = 0.0;
    double true_y = 0.0;
    for (int step = 0; step < steps; step++) {
        true_x += true_vx * TimeStep;
        true_y += true_vy * TimeStep;
        EXPECT_TRUE(Tests::Succeeded(filter.Predict()));
        EXPECT_TRUE(Tests::Succeeded(
            filter.Update({true_x + noise.Symmetric(noise_amplitude), true_y + noise.Symmetric(noise_amplitude)})));
    }
    const Core::Vector &state = filter.GetState();
    return {std::hypot(state[0] - true_x, state[1] - true_y), std::hypot(state[2] - true_vx, state[3] - true_vy)};
}

} // namespace

// Setup

TEST(KalmanFilter, AFilterNeedsAtLeastOneStateAndOneMeasurement) {
    EXPECT_FALSE(Core::KalmanFilter::Create(0, 1));
    EXPECT_FALSE(Core::KalmanFilter::Create(1, 0));
    EXPECT_TRUE(Tests::Succeeded(Core::KalmanFilter::Create(1, 1)));
}

TEST(KalmanFilter, ANewFilterHasAZeroStateAndIsNotReady) {
    const Core::Result<Core::KalmanFilter> filter = Core::KalmanFilter::Create(3, 2, 1);

    ASSERT_TRUE(Tests::Succeeded(filter));
    EXPECT_TRUE(Tests::VectorNear({0.0, 0.0, 0.0}, filter->GetState(), 0.0));
    EXPECT_FALSE(filter->IsInitialized());
}

TEST(KalmanFilter, TheFilterIsReadyOnlyOnceTheErrorCovarianceIsSet) {
    Core::Result<Core::KalmanFilter> filter = Core::KalmanFilter::Create(2, 1);
    ASSERT_TRUE(Tests::Succeeded(filter));

    EXPECT_FALSE(filter->Predict());
    EXPECT_FALSE(filter->Update({1.0}));
    ASSERT_TRUE(Tests::Succeeded(filter->SetErrorCovariance(Core::Matrix::Identity(2))));
    EXPECT_TRUE(filter->IsInitialized());
    EXPECT_TRUE(Tests::Succeeded(filter->Predict()));
}

TEST(KalmanFilter, TheInitialStateMustHaveOneValuePerState) {
    Core::Result<Core::KalmanFilter> filter = Core::KalmanFilter::Create(2, 1);
    ASSERT_TRUE(Tests::Succeeded(filter));

    EXPECT_FALSE(filter->InitializeState({1.0, 2.0, 3.0}));
    EXPECT_TRUE(Tests::Succeeded(filter->InitializeState({1.0, 2.0})));
    EXPECT_TRUE(Tests::VectorNear({1.0, 2.0}, filter->GetState(), 0.0));
}

TEST(KalmanFilter, MatricesOfTheWrongShapeAreRejectedAndLeaveTheFilterNotReady) {
    Core::Result<Core::KalmanFilter> filter = Core::KalmanFilter::Create(2, 1);
    ASSERT_TRUE(Tests::Succeeded(filter));
    const Core::Matrix wrong = Core::Matrix::Identity(3);

    EXPECT_FALSE(filter->SetStateTransitionMatrix(wrong));
    EXPECT_FALSE(filter->SetObservationMatrix(wrong));
    EXPECT_FALSE(filter->SetProcessNoiseCovariance(wrong));
    EXPECT_FALSE(filter->SetMeasurementNoiseCovariance(wrong));
    EXPECT_FALSE(filter->SetErrorCovariance(wrong));
    EXPECT_FALSE(filter->IsInitialized());
}

TEST(KalmanFilter, TheErrorMessageSaysWhichShapeWasExpected) {
    Core::Result<Core::KalmanFilter> filter = Core::KalmanFilter::Create(2, 1);
    ASSERT_TRUE(Tests::Succeeded(filter));

    const Core::Result<void> outcome = filter->SetObservationMatrix(Core::Matrix(2, 2));

    ASSERT_FALSE(outcome);
    EXPECT_EQ(outcome.Error(), "The observation matrix must be 1x2, not 2x2");
}

TEST(KalmanFilter, AControlMatrixNeedsAFilterWithControlInputs) {
    Core::Result<Core::KalmanFilter> without_control = Core::KalmanFilter::Create(2, 1);
    Core::Result<Core::KalmanFilter> with_control = Core::KalmanFilter::Create(2, 1, 1);
    ASSERT_TRUE(Tests::Succeeded(without_control));
    ASSERT_TRUE(Tests::Succeeded(with_control));
    const Core::Matrix control_matrix{{0.5}, {1.0}};

    EXPECT_FALSE(without_control->SetControlMatrix(control_matrix));
    EXPECT_TRUE(Tests::Succeeded(with_control->SetControlMatrix(control_matrix)));
}

TEST(KalmanFilter, AStateElementPastTheEndCannotBeRead) {
    Core::KalmanFilter filter = PositionVelocityFilter();

    EXPECT_EQ(filter.GetStateElement(1), 0.5);
    EXPECT_FALSE(filter.GetStateElement(2).has_value());
}

// Prediction

TEST(KalmanFilter, PredictingMovesThePositionByTheVelocityTimesTheTimeStep) {
    Core::KalmanFilter filter = PositionVelocityFilter();

    ASSERT_TRUE(Tests::Succeeded(filter.Predict()));

    EXPECT_TRUE(Tests::VectorNear({1.0 + 0.5 * TimeStep, 0.5}, filter.GetState(), Tolerance));
}

TEST(KalmanFilter, PredictingGrowsTheUncertaintyByFPFTransposedPlusQ) {
    Core::KalmanFilter filter = PositionVelocityFilter();

    ASSERT_TRUE(Tests::Succeeded(filter.Predict()));

    // With P = I, Q = 0.01 I and F = [[1, dt], [0, 1]], F P F^T + Q = [[1 + dt^2 + 0.01, dt], [dt, 1.01]]
    const Core::Matrix expected{{1.0 + TimeStep * TimeStep + 0.01, TimeStep}, {TimeStep, 1.01}};
    EXPECT_TRUE(Tests::MatrixNear(expected, filter.GetErrorCovariance(), Tolerance));
}

TEST(KalmanFilter, PredictingAddsTheEffectOfTheControlInput) {
    Core::Result<Core::KalmanFilter> filter = Core::KalmanFilter::Create(2, 1, 1);
    ASSERT_TRUE(Tests::Succeeded(filter));
    ASSERT_TRUE(Tests::Succeeded(filter->InitializeState({0.0, 1.0})));
    ASSERT_TRUE(Tests::Succeeded(filter->SetStateTransitionMatrix({{1.0, TimeStep}, {0.0, 1.0}})));
    ASSERT_TRUE(Tests::Succeeded(filter->SetControlMatrix({{0.5 * TimeStep * TimeStep}, {TimeStep}})));
    ASSERT_TRUE(Tests::Succeeded(filter->SetErrorCovariance(Core::Matrix::Identity(2))));

    ASSERT_TRUE(Tests::Succeeded(filter->Predict({2.0})));

    // Constant acceleration of 2 for dt: the velocity grows by 2 dt and the position by v dt + dt^2
    EXPECT_TRUE(
        Tests::VectorNear({TimeStep + TimeStep * TimeStep, 1.0 + 2.0 * TimeStep}, filter->GetState(), Tolerance));
}

TEST(KalmanFilter, AControlInputOfTheWrongSizeIsRejectedAndLeavesTheFilterUnchanged) {
    Core::KalmanFilter filter = PositionVelocityFilter();

    EXPECT_FALSE(filter.Predict({1.0}));

    EXPECT_TRUE(Tests::VectorNear({1.0, 0.5}, filter.GetState(), 0.0));
}

// Update

TEST(KalmanFilter, UpdatingLandsBetweenThePredictionAndTheMeasurement) {
    Core::KalmanFilter filter = PositionVelocityFilter();
    ASSERT_TRUE(Tests::Succeeded(filter.Predict()));

    ASSERT_TRUE(Tests::Succeeded(filter.Update({1.06})));

    EXPECT_GT(filter.GetState()[0], 1.0 + 0.5 * TimeStep);
    EXPECT_LT(filter.GetState()[0], 1.06);
}

TEST(KalmanFilter, UpdatingWithEqualTrustLandsHalfwayToTheMeasurement) {
    Core::Result<Core::KalmanFilter> filter = Core::KalmanFilter::Create(1, 1);
    ASSERT_TRUE(Tests::Succeeded(filter));
    ASSERT_TRUE(Tests::Succeeded(filter->SetObservationMatrix({{1.0}})));
    ASSERT_TRUE(Tests::Succeeded(filter->SetMeasurementNoiseCovariance({{1.0}})));
    ASSERT_TRUE(Tests::Succeeded(filter->SetErrorCovariance({{1.0}})));

    ASSERT_TRUE(Tests::Succeeded(filter->Update({2.0})));

    EXPECT_NEAR(filter->GetState()[0], 1.0, Tolerance);
    EXPECT_NEAR(filter->GetErrorCovariance()(0, 0), 0.5, Tolerance);
}

TEST(KalmanFilter, ANearlyNoiselessSensorMakesTheEstimateFollowTheMeasurement) {
    Core::KalmanFilter filter = PositionVelocityFilter();
    ASSERT_TRUE(Tests::Succeeded(filter.SetMeasurementNoiseCovariance({{1e-10}})));

    ASSERT_TRUE(Tests::Succeeded(filter.Update({5.0})));

    EXPECT_NEAR(filter.GetState()[0], 5.0, 1e-6);
}

TEST(KalmanFilter, AVeryNoisySensorBarelyMovesTheEstimate) {
    Core::KalmanFilter filter = PositionVelocityFilter();
    ASSERT_TRUE(Tests::Succeeded(filter.SetMeasurementNoiseCovariance({{100.0}})));

    ASSERT_TRUE(Tests::Succeeded(filter.Update({10.0})));

    EXPECT_LT(std::abs(filter.GetState()[0] - 1.0), 0.2);
}

TEST(KalmanFilter, AMeasurementOfTheWrongSizeIsRejected) {
    Core::KalmanFilter filter = PositionVelocityFilter();

    EXPECT_FALSE(filter.Update({1.0, 2.0}));
}

TEST(KalmanFilter, AnUpdateWithASingularInnovationFailsAndChangesNothing) {
    Core::Result<Core::KalmanFilter> filter = Core::KalmanFilter::Create(1, 1);
    ASSERT_TRUE(Tests::Succeeded(filter));
    ASSERT_TRUE(Tests::Succeeded(filter->InitializeState({5.0})));
    // H = 0 and R = 0 leave S = 0, which cannot be inverted
    ASSERT_TRUE(Tests::Succeeded(filter->SetErrorCovariance({{1.0}})));

    EXPECT_FALSE(filter->Update({9.0}));

    EXPECT_EQ(filter->GetState()[0], 5.0);
    EXPECT_EQ(filter->GetErrorCovariance()(0, 0), 1.0);
}

TEST(KalmanFilter, UncertaintyShrinksWithEveryUpdateAndGrowsWithEveryPrediction) {
    Core::KalmanFilter filter = PositionVelocityFilter();

    ASSERT_TRUE(Tests::Succeeded(filter.Update({1.0})));
    const double after_update = filter.GetErrorCovariance()(0, 0);
    ASSERT_TRUE(Tests::Succeeded(filter.Predict()));

    EXPECT_LT(after_update, 1.0);
    EXPECT_GT(filter.GetErrorCovariance()(0, 0), after_update);
}

// Lifecycle

TEST(KalmanFilter, ResettingClearsTheEstimateButKeepsTheModel) {
    Core::KalmanFilter filter = PositionVelocityFilter();

    filter.Reset();

    EXPECT_FALSE(filter.IsInitialized());
    EXPECT_TRUE(Tests::VectorNear({0.0, 0.0}, filter.GetState(), 0.0));
    EXPECT_TRUE(Tests::MatrixNear(Core::Matrix(2, 2), filter.GetErrorCovariance(), 0.0));

    // The same filter starts again once its covariance is set
    ASSERT_TRUE(Tests::Succeeded(filter.SetErrorCovariance(Core::Matrix::Identity(2))));
    ASSERT_TRUE(Tests::Succeeded(filter.Predict()));
}

TEST(KalmanFilter, AFilterWithManyStatesKeepsAllOfThem) {
    Core::Result<Core::KalmanFilter> filter = Core::KalmanFilter::Create(10, 3);
    ASSERT_TRUE(Tests::Succeeded(filter));
    const Core::Vector state(10, 1.0);

    ASSERT_TRUE(Tests::Succeeded(filter->InitializeState(state)));

    EXPECT_TRUE(Tests::VectorNear(state, filter->GetState(), 0.0));
}

// Tracking

TEST(KalmanFilter, AFilterWithAPoorInitialGuessFindsTheVelocityOfAnObjectOnALine) {
    const double measurement_noise = 0.5;
    Tests::Model model = Tests::PositionVelocityModel(TimeStep, measurement_noise * measurement_noise, 10.0);
    model.Q = Tests::PositionVelocityNoise(TimeStep, 0.1);
    Core::KalmanFilter filter = Tests::StartedFilter(model);

    const TrackingError error = Track1d(filter, 2.0, 50, measurement_noise, 42);

    EXPECT_LT(error.Velocity, 0.2);
    EXPECT_LT(error.Position, 0.5);
}

TEST(KalmanFilter, TheFilterFindsTheVelocityFromEveryInitialVelocityGuess) {
    for (const double initial_velocity : {0.0, 0.5, 1.0, 2.0, 3.0}) {
        Tests::Model model = Tests::PositionVelocityModel(TimeStep, 0.25, 1.0);
        model.InitialState = {0.0, initial_velocity};
        Core::KalmanFilter filter = Tests::StartedFilter(model);

        const TrackingError error = Track1d(filter, 1.5, 30, 0.5, 100);

        EXPECT_LT(error.Velocity, 0.3) << "initial velocity " << initial_velocity;
    }
}

TEST(KalmanFilter, TheFilterFollowsAnObjectOnAPlane) {
    const double measurement_noise = 0.3;
    Core::KalmanFilter filter = Tests::StartedFilter(
        Tests::PlanarPositionVelocityModel(TimeStep, 0.05 * 0.05, measurement_noise * measurement_noise, 5.0));

    const TrackingError error = Track2d(filter, 1.5, -1.0, 30, measurement_noise, 123);

    EXPECT_LT(error.Position, 0.2);
}

TEST(KalmanFilter, TheFilterFollowsEveryDirectionOfMotion) {
    const std::vector<std::pair<double, double>> motions{{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {-1.0, 0.5}};

    for (const auto &[true_vx, true_vy] : motions) {
        Core::KalmanFilter filter = Tests::StartedFilter(Tests::PlanarPositionVelocityModel(TimeStep, 0.01, 0.1, 1.0));

        const TrackingError error = Track2d(filter, true_vx, true_vy, 25, 0.3, 200);

        EXPECT_LT(error.Position, 0.2) << "motion " << true_vx << ", " << true_vy;
        EXPECT_LT(error.Velocity, 0.1) << "motion " << true_vx << ", " << true_vy;
    }
}

TEST(KalmanFilter, TheFilterStaysCloseToTheTruthAtEveryMeasurementNoiseLevel) {
    for (const double noise_level : {0.1, 0.5, 1.0, 2.0}) {
        Core::KalmanFilter filter =
            Tests::StartedFilter(Tests::PositionVelocityModel(TimeStep, noise_level * noise_level, 1.0));

        const TrackingError error = Track1d(filter, 1.0, 20, noise_level, 200);

        EXPECT_LT(error.Velocity, 0.5) << "noise level " << noise_level;
    }
}

TEST(KalmanFilter, TheFilterStaysCloseToTheTruthAtEveryProcessNoiseLevel) {
    for (const double process_noise : {0.01, 0.05, 0.1, 0.2, 0.5}) {
        Tests::Model model = Tests::PositionVelocityModel(TimeStep, 0.25, 1.0);
        model.Q = Tests::PositionVelocityNoise(TimeStep, process_noise);
        Core::KalmanFilter filter = Tests::StartedFilter(model);

        const TrackingError error = Track1d(filter, 1.0, 25, 0.5, 300);

        EXPECT_LT(error.Velocity, 0.6) << "process noise " << process_noise;
    }
}

TEST(KalmanFilter, TheFilterBalancesTrustInTheModelAndInTheSensor) {
    const std::vector<std::pair<double, double>> combinations{{0.1, 0.01}, {0.1, 0.1}, {1.0, 0.01}, {1.0, 0.1}};

    for (const auto &[measurement_noise, process_noise] : combinations) {
        Tests::Model model = Tests::PositionVelocityModel(TimeStep, measurement_noise * measurement_noise, 1.0);
        model.Q = Tests::PositionVelocityNoise(TimeStep, process_noise);
        Core::KalmanFilter filter = Tests::StartedFilter(model);

        const TrackingError error = Track1d(filter, 1.0, 30, measurement_noise, 400);

        EXPECT_LT(error.Velocity, 0.3) << "measurement noise " << measurement_noise << ", process noise "
                                       << process_noise;
        EXPECT_LT(error.Position, 0.5) << "measurement noise " << measurement_noise << ", process noise "
                                       << process_noise;
    }
}

TEST(KalmanFilter, FilteringBeatsTrustingTheRawMeasurements) {
    constexpr int Steps = 200;
    constexpr int SettlingSteps = 50;
    constexpr double NoiseAmplitude = 1.0;
    // Uniform noise of amplitude 1 has a variance of 1/3
    Tests::Model model = Tests::PositionVelocityModel(TimeStep, NoiseAmplitude * NoiseAmplitude / 3.0, 1.0);
    model.Q = Tests::PositionVelocityNoise(TimeStep, 0.01);
    Core::KalmanFilter filter = Tests::StartedFilter(model);

    // Compare the squared error of the filter with that of the raw measurements, once the filter has settled
    Tests::NoiseSource noise(7);
    double true_position = 0.0;
    double filtered_squared_error = 0.0;
    double raw_squared_error = 0.0;
    for (int step = 0; step < Steps; step++) {
        true_position += 1.0 * TimeStep;
        const double measurement_error = noise.Symmetric(NoiseAmplitude);
        ASSERT_TRUE(Tests::Succeeded(filter.Predict()));
        ASSERT_TRUE(Tests::Succeeded(filter.Update({true_position + measurement_error})));
        if (step >= SettlingSteps) {
            const double filter_error = filter.GetState()[0] - true_position;
            filtered_squared_error += filter_error * filter_error;
            raw_squared_error += measurement_error * measurement_error;
        }
    }

    EXPECT_LT(filtered_squared_error, 0.5 * raw_squared_error);
}
