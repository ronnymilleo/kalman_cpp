/**
 * @file    extended_kalman_filter_tests.cpp
 * @brief   Unit tests for the Extended Kalman filter.
 */

#include "filters/extended_kalman_filter.h"

#include "test_support.h"

#include <cmath>
#include <gtest/gtest.h>

namespace {

constexpr double Tolerance = 1e-10;

// f(x) = (x0^2, x1)
Core::Vector SquareFirstState(const Core::Vector &state, const Core::Vector & /*control*/) {
    return {state[0] * state[0], state[1]};
}

Core::Matrix SquareFirstStateJacobian(const Core::Vector &state) {
    return {{2.0 * state[0], 0.0}, {0.0, 1.0}};
}

// f(x, u) = (x1 + u, x0)
Core::Vector SwapStatesWithControl(const Core::Vector &state, const Core::Vector &control) {
    const double push = control.empty() ? 0.0 : control[0];
    return {state[1] + push, state[0]};
}

Core::Matrix SwapStatesJacobian(const Core::Vector & /*state*/) {
    return {{0.0, 1.0}, {1.0, 0.0}};
}

// h(x) = distance from the origin
Core::Vector DistanceFromOrigin(const Core::Vector &state) {
    return {std::hypot(state[0], state[1])};
}

Core::Matrix DistanceFromOriginJacobian(const Core::Vector &state) {
    const double distance = std::hypot(state[0], state[1]);
    return {{state[0] / distance, state[1] / distance}};
}

// Two states and one measurement, identity covariance, and process and measurement noise as given
Core::ExtendedKalmanFilter StartedFilter(const Core::Vector &initial_state, const double process_noise,
                                         const double measurement_noise) {
    Core::ExtendedKalmanFilter filter = Core::ExtendedKalmanFilter::Create(2, 1, 1).value();
    filter.InitializeState(initial_state).value();
    filter.SetProcessNoiseCovariance({{process_noise, 0.0}, {0.0, process_noise}}).value();
    filter.SetMeasurementNoiseCovariance({{measurement_noise}}).value();
    filter.SetErrorCovariance(Core::Matrix::Identity(2)).value();
    return filter;
}

void SetSquareFirstState(Core::ExtendedKalmanFilter &filter) {
    EXPECT_TRUE(Tests::Succeeded(filter.SetStateTransitionFunction(SquareFirstState)));
    EXPECT_TRUE(Tests::Succeeded(filter.SetStateJacobianFunction(SquareFirstStateJacobian)));
}

void SetDistanceFromOrigin(Core::ExtendedKalmanFilter &filter) {
    EXPECT_TRUE(Tests::Succeeded(filter.SetObservationFunction(DistanceFromOrigin)));
    EXPECT_TRUE(Tests::Succeeded(filter.SetObservationJacobianFunction(DistanceFromOriginJacobian)));
}

} // namespace

TEST(ExtendedKalmanFilter, AFilterNeedsAtLeastOneStateAndOneMeasurement) {
    EXPECT_FALSE(Core::ExtendedKalmanFilter::Create(0, 1));
    EXPECT_FALSE(Core::ExtendedKalmanFilter::Create(1, 0));
    EXPECT_TRUE(Tests::Succeeded(Core::ExtendedKalmanFilter::Create(1, 1)));
}

TEST(ExtendedKalmanFilter, ModelFunctionsCannotBeEmpty) {
    Core::ExtendedKalmanFilter filter = Core::ExtendedKalmanFilter::Create(2, 1).value();

    EXPECT_FALSE(filter.SetStateTransitionFunction(nullptr));
    EXPECT_FALSE(filter.SetObservationFunction(nullptr));
    EXPECT_FALSE(filter.SetStateJacobianFunction(nullptr));
    EXPECT_FALSE(filter.SetObservationJacobianFunction(nullptr));
}

TEST(ExtendedKalmanFilter, TheModelIsCompleteOnlyOnceAllFourFunctionsAreSet) {
    Core::ExtendedKalmanFilter filter = Core::ExtendedKalmanFilter::Create(2, 1).value();
    EXPECT_FALSE(filter.AreAllFunctionsSet());

    SetSquareFirstState(filter);
    EXPECT_FALSE(filter.AreAllFunctionsSet());
    SetDistanceFromOrigin(filter);
    EXPECT_TRUE(filter.AreAllFunctionsSet());
}

TEST(ExtendedKalmanFilter, MatricesOfTheWrongShapeAreRejected) {
    Core::ExtendedKalmanFilter filter = Core::ExtendedKalmanFilter::Create(2, 1).value();
    const Core::Matrix wrong = Core::Matrix::Identity(3);

    EXPECT_FALSE(filter.SetProcessNoiseCovariance(wrong));
    EXPECT_FALSE(filter.SetMeasurementNoiseCovariance(wrong));
    EXPECT_FALSE(filter.SetErrorCovariance(wrong));
    EXPECT_FALSE(filter.IsInitialized());
}

TEST(ExtendedKalmanFilter, TheFilterCannotPredictOrUpdateBeforeItIsReady) {
    Core::ExtendedKalmanFilter filter = Core::ExtendedKalmanFilter::Create(2, 1).value();
    SetSquareFirstState(filter);
    SetDistanceFromOrigin(filter);

    EXPECT_FALSE(filter.Predict());
    EXPECT_FALSE(filter.Update({1.0}));
}

TEST(ExtendedKalmanFilter, TheFilterCannotPredictOrUpdateWithoutTheFourFunctions) {
    Core::ExtendedKalmanFilter filter = StartedFilter({1.0, 1.0}, 0.1, 0.2);

    EXPECT_FALSE(filter.Predict());
    EXPECT_FALSE(filter.Update({1.0}));
}

TEST(ExtendedKalmanFilter, PredictingAppliesTheNonLinearStateTransition) {
    Core::ExtendedKalmanFilter filter = StartedFilter({1.5, 1.0}, 0.1, 0.2);
    SetSquareFirstState(filter);
    SetDistanceFromOrigin(filter);

    ASSERT_TRUE(Tests::Succeeded(filter.Predict()));

    EXPECT_TRUE(Tests::VectorNear({2.25, 1.0}, filter.GetState(), Tolerance));
}

TEST(ExtendedKalmanFilter, PredictingLinearizesAroundTheStateTheStepStartsFrom) {
    Core::ExtendedKalmanFilter filter = StartedFilter({2.0, 1.0}, 0.1, 0.2);
    SetSquareFirstState(filter);
    SetDistanceFromOrigin(filter);

    ASSERT_TRUE(Tests::Succeeded(filter.Predict()));

    // The Jacobian at x0 = 2 is 4, so P00 = 4 * 1 * 4 + 0.1; evaluating it at the predicted x0 = 4 would give 64.1
    EXPECT_TRUE(Tests::MatrixNear(Core::Matrix({{16.1, 0.0}, {0.0, 1.1}}), filter.GetErrorCovariance(), Tolerance));
}

TEST(ExtendedKalmanFilter, PredictingPassesTheControlInputToTheStateTransition) {
    Core::ExtendedKalmanFilter filter = StartedFilter({1.0, 2.0}, 0.0, 1.0);
    ASSERT_TRUE(Tests::Succeeded(filter.SetStateTransitionFunction(SwapStatesWithControl)));
    ASSERT_TRUE(Tests::Succeeded(filter.SetStateJacobianFunction(SwapStatesJacobian)));
    SetDistanceFromOrigin(filter);

    ASSERT_TRUE(Tests::Succeeded(filter.Predict({10.0})));

    EXPECT_TRUE(Tests::VectorNear({12.0, 1.0}, filter.GetState(), Tolerance));
}

TEST(ExtendedKalmanFilter, AControlInputOfTheWrongSizeIsRejected) {
    Core::ExtendedKalmanFilter filter = StartedFilter({1.0, 2.0}, 0.0, 1.0);
    SetSquareFirstState(filter);
    SetDistanceFromOrigin(filter);

    EXPECT_FALSE(filter.Predict({1.0, 2.0}));
}

TEST(ExtendedKalmanFilter, AFunctionThatReturnsTheWrongSizeIsRejectedAndLeavesTheFilterUnchanged) {
    Core::ExtendedKalmanFilter filter = StartedFilter({1.0, 2.0}, 0.1, 1.0);
    SetSquareFirstState(filter);
    SetDistanceFromOrigin(filter);
    ASSERT_TRUE(Tests::Succeeded(filter.SetStateTransitionFunction(
        [](const Core::Vector &state, const Core::Vector & /*control*/) { return Core::Vector{state[0]}; })));

    EXPECT_FALSE(filter.Predict());

    EXPECT_TRUE(Tests::VectorNear({1.0, 2.0}, filter.GetState(), 0.0));
}

TEST(ExtendedKalmanFilter, AJacobianOfTheWrongShapeIsRejected) {
    Core::ExtendedKalmanFilter filter = StartedFilter({3.0, 4.0}, 0.1, 1.0);
    SetSquareFirstState(filter);
    ASSERT_TRUE(Tests::Succeeded(filter.SetObservationFunction(DistanceFromOrigin)));
    ASSERT_TRUE(Tests::Succeeded(filter.SetObservationJacobianFunction(
        [](const Core::Vector & /*state*/) { return Core::Matrix({{1.0, 0.0, 0.0}}); })));

    EXPECT_FALSE(filter.Update({5.0}));
}

TEST(ExtendedKalmanFilter, UpdatingMovesTheStateAlongTheGradientOfTheObservation) {
    Core::ExtendedKalmanFilter filter = StartedFilter({3.0, 4.0}, 0.0, 1.0);
    SetSquareFirstState(filter);
    SetDistanceFromOrigin(filter);

    ASSERT_TRUE(Tests::Succeeded(filter.Update({5.5})));

    // The true distance is 5, so the innovation is 0.5. With H = (0.6, 0.8), P = I and R = 1, S = 2 and the gain is
    // (0.3, 0.4); the state moves by gain * innovation
    EXPECT_TRUE(Tests::VectorNear({3.15, 4.2}, filter.GetState(), Tolerance));
}

TEST(ExtendedKalmanFilter, AnUpdateWithASingularInnovationFailsAndChangesNothing) {
    Core::ExtendedKalmanFilter filter = StartedFilter({3.0, 4.0}, 0.0, 0.0);
    SetSquareFirstState(filter);
    SetDistanceFromOrigin(filter);
    // With no uncertainty at all the innovation covariance is zero
    ASSERT_TRUE(Tests::Succeeded(filter.SetErrorCovariance(Core::Matrix(2, 2))));

    EXPECT_FALSE(filter.Update({6.0}));

    EXPECT_TRUE(Tests::VectorNear({3.0, 4.0}, filter.GetState(), 0.0));
}

TEST(ExtendedKalmanFilter, AStateElementPastTheEndCannotBeRead) {
    const Core::ExtendedKalmanFilter filter = StartedFilter({3.0, 4.0}, 0.0, 1.0);

    EXPECT_EQ(filter.GetStateElement(1), 4.0);
    EXPECT_FALSE(filter.GetStateElement(2).has_value());
}

TEST(ExtendedKalmanFilter, ResettingClearsTheEstimateButKeepsTheFunctions) {
    Core::ExtendedKalmanFilter filter = StartedFilter({3.0, 4.0}, 0.0, 1.0);
    SetSquareFirstState(filter);
    SetDistanceFromOrigin(filter);

    filter.Reset();

    EXPECT_FALSE(filter.IsInitialized());
    EXPECT_TRUE(filter.AreAllFunctionsSet());
    EXPECT_TRUE(Tests::VectorNear({0.0, 0.0}, filter.GetState(), 0.0));
}

TEST(ExtendedKalmanFilter, TheFilterTracksAnObjectFromMeasurementsOfItsSquaredPosition) {
    // The object moves along a line at 1 unit per step, and the sensor reports the square of its position
    Core::ExtendedKalmanFilter filter = Core::ExtendedKalmanFilter::Create(2, 1).value();
    ASSERT_TRUE(Tests::Succeeded(
        filter.SetStateTransitionFunction([](const Core::Vector &state, const Core::Vector & /*control*/) {
            return Core::Vector{state[0] + state[1], state[1]};
        })));
    ASSERT_TRUE(Tests::Succeeded(filter.SetStateJacobianFunction(
        [](const Core::Vector & /*state*/) { return Core::Matrix({{1.0, 1.0}, {0.0, 1.0}}); })));
    ASSERT_TRUE(Tests::Succeeded(
        filter.SetObservationFunction([](const Core::Vector &state) { return Core::Vector{state[0] * state[0]}; })));
    ASSERT_TRUE(Tests::Succeeded(filter.SetObservationJacobianFunction(
        [](const Core::Vector &state) { return Core::Matrix({{2.0 * state[0], 0.0}}); })));
    ASSERT_TRUE(Tests::Succeeded(filter.InitializeState({8.0, 0.0})));
    ASSERT_TRUE(Tests::Succeeded(filter.SetProcessNoiseCovariance({{1e-4, 0.0}, {0.0, 1e-4}})));
    ASSERT_TRUE(Tests::Succeeded(filter.SetMeasurementNoiseCovariance({{1.0}})));
    ASSERT_TRUE(Tests::Succeeded(filter.SetErrorCovariance({{4.0, 0.0}, {0.0, 1.0}})));

    Tests::NoiseSource noise(5);
    double true_position = 10.0;
    for (int step = 0; step < 40; step++) {
        true_position += 1.0;
        ASSERT_TRUE(Tests::Succeeded(filter.Predict()));
        ASSERT_TRUE(Tests::Succeeded(filter.Update({true_position * true_position + noise.Symmetric(1.0)})));
    }

    EXPECT_NEAR(filter.GetState()[0], true_position, 0.5);
    EXPECT_NEAR(filter.GetState()[1], 1.0, 0.2);
}
