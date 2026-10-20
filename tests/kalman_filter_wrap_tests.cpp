/**
 * @file    kalman_filter_wrap_tests.cpp
 * @brief   Unit tests for filters whose measurements or states are angles: the residual and state normalizer hooks.
 */

#include "core/angle.h"
#include "filters/extended_kalman_filter.h"
#include "filters/kalman_filter.h"

#include "test_support.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

namespace {

constexpr double Tolerance = 1e-9;
constexpr double TimeStep = 0.1;

Core::Vector WrappedResidual(const Core::Vector &measurement, const Core::Vector &expected_measurement) {
    return {Core::AngleDifference(measurement[0], expected_measurement[0])};
}

void WrapHeading(Core::Vector &state) {
    state[0] = Core::WrapAngle(state[0]);
}

// A heading and its turn rate, with the heading measured: the constant turn rate model
Core::KalmanFilter HeadingFilter(const double dt, const double measurement_variance, const double initial_heading,
                                 const double initial_rate) {
    Tests::Model model = Tests::PositionVelocityModel(dt, measurement_variance, 1.0);
    model.Q = {{1e-4, 0.0}, {0.0, 1e-4}};
    model.InitialState = {initial_heading, initial_rate};
    return Tests::StartedFilter(model);
}

// Tracks a heading that turns at a constant rate through the boundary at pi; returns the largest heading error after
// the first few steps, so a glitch at the crossing counts even if the filter recovers from it
double TrackHeading(Core::KalmanFilter &filter, const double turn_rate, const double start_heading, const int steps,
                    const double noise_amplitude) {
    constexpr int SettlingSteps = 3;
    Tests::NoiseSource noise(11);
    double true_heading = start_heading;
    double largest_error = 0.0;
    for (int step = 0; step < steps; step++) {
        true_heading += turn_rate * TimeStep;
        EXPECT_TRUE(Tests::Succeeded(filter.Predict()));
        EXPECT_TRUE(
            Tests::Succeeded(filter.Update({Core::WrapAngle(true_heading) + noise.Symmetric(noise_amplitude)})));
        if (step >= SettlingSteps) {
            largest_error =
                std::max(largest_error, std::abs(Core::AngleDifference(filter.GetState()[0], true_heading)));
        }
    }
    return largest_error;
}

} // namespace

TEST(KalmanFilterWrap, TheResidualFunctionReplacesTheSubtractionOfTheInnovation) {
    Core::KalmanFilter filter = HeadingFilter(TimeStep, 0.01, Core::Pi - 0.01, 0.0);
    filter.SetResidualFunction(WrappedResidual);

    ASSERT_TRUE(Tests::Succeeded(filter.Update({-Core::Pi + 0.01})));

    // The two headings are 0.02 apart across the boundary, so the estimate moves a little, not by a whole turn
    EXPECT_LT(std::abs(Core::AngleDifference(filter.GetState()[0], Core::Pi)), 0.05);
}

TEST(KalmanFilterWrap, WithoutTheResidualFunctionAHeadingAcrossTheBoundaryThrowsTheEstimateOff) {
    Core::KalmanFilter filter = HeadingFilter(TimeStep, 0.01, Core::Pi - 0.01, 0.0);

    ASSERT_TRUE(Tests::Succeeded(filter.Update({-Core::Pi + 0.01})));

    // The innovation is almost a whole turn, and the estimate follows most of it instead of moving 0.02
    EXPECT_GT(std::abs(filter.GetState()[0] - (Core::Pi - 0.01)), 3.0);
}

TEST(KalmanFilterWrap, TheStateNormalizerAdjustsTheStateAfterAPrediction) {
    Core::KalmanFilter filter = HeadingFilter(1.0, 0.01, Core::Pi - 0.1, 0.3);
    filter.SetStateNormalizer(WrapHeading);

    ASSERT_TRUE(Tests::Succeeded(filter.Predict()));

    // The heading would be pi + 0.2, which is the same direction as -pi + 0.2
    EXPECT_NEAR(filter.GetState()[0], -Core::Pi + 0.2, Tolerance);
}

TEST(KalmanFilterWrap, TheStateNormalizerAdjustsTheStateAfterAnUpdate) {
    Core::KalmanFilter filter = HeadingFilter(TimeStep, 1e-6, Core::Pi - 0.001, 0.0);
    filter.SetStateNormalizer(WrapHeading);
    filter.SetResidualFunction(WrappedResidual);

    ASSERT_TRUE(Tests::Succeeded(filter.Update({-Core::Pi + 0.001})));

    EXPECT_LT(filter.GetState()[0], 0.0);
    EXPECT_GE(filter.GetState()[0], -Core::Pi);
}

TEST(KalmanFilterWrap, AFilterWithBothHooksFollowsAHeadingThroughTheBoundary) {
    Core::KalmanFilter filter = HeadingFilter(TimeStep, 0.01, 2.5, 0.0);
    filter.SetResidualFunction(WrappedResidual);
    filter.SetStateNormalizer(WrapHeading);

    const double error = TrackHeading(filter, 1.0, 2.5, 100, 0.1);

    EXPECT_LT(error, 0.3);
    EXPECT_NEAR(filter.GetState()[1], 1.0, 0.1);
}

TEST(KalmanFilterWrap, AFilterWithoutTheHooksLosesAHeadingThatCrossesTheBoundary) {
    Core::KalmanFilter filter = HeadingFilter(TimeStep, 0.01, 2.5, 0.0);

    const double error = TrackHeading(filter, 1.0, 2.5, 100, 0.1);

    EXPECT_GT(error, 0.5);
}

TEST(KalmanFilterWrap, TheHooksCanBeCleared) {
    Core::KalmanFilter filter = HeadingFilter(TimeStep, 0.01, Core::Pi - 0.01, 0.0);
    filter.SetResidualFunction(WrappedResidual);
    filter.SetResidualFunction(nullptr);

    ASSERT_TRUE(Tests::Succeeded(filter.Update({-Core::Pi + 0.01})));

    EXPECT_GT(std::abs(filter.GetState()[0] - (Core::Pi - 0.01)), 3.0);
}

TEST(KalmanFilterWrap, AResidualFunctionThatReturnsTheWrongSizeIsRejectedAndLeavesTheFilterUnchanged) {
    Core::KalmanFilter filter = HeadingFilter(TimeStep, 0.01, 1.0, 0.0);
    filter.SetResidualFunction([](const Core::Vector &, const Core::Vector &) { return Core::Vector{0.0, 0.0}; });

    EXPECT_FALSE(filter.Update({2.0}));

    EXPECT_TRUE(Tests::VectorNear({1.0, 0.0}, filter.GetState(), 0.0));
}

TEST(ExtendedKalmanFilterWrap, TheHooksWorkOnTheExtendedFilterToo) {
    // The bearing of an object at (x, y) seen from the origin, with the state the position of the object
    Core::ExtendedKalmanFilter filter = Tests::Must(Core::ExtendedKalmanFilter::Create(2, 1));
    Tests::Must(
        filter.SetStateTransitionFunction([](const Core::Vector &state, const Core::Vector &) { return state; }));
    Tests::Must(filter.SetStateJacobianFunction([](const Core::Vector &) { return Core::Matrix::Identity(2); }));
    Tests::Must(filter.SetObservationFunction(
        [](const Core::Vector &state) { return Core::Vector{std::atan2(state[1], state[0])}; }));
    Tests::Must(filter.SetObservationJacobianFunction([](const Core::Vector &state) {
        const double range_squared = state[0] * state[0] + state[1] * state[1];
        return Core::Matrix{{-state[1] / range_squared, state[0] / range_squared}};
    }));
    Tests::Must(filter.InitializeState({-10.0, 0.5}));
    Tests::Must(filter.SetProcessNoiseCovariance({{0.0, 0.0}, {0.0, 0.0}}));
    Tests::Must(filter.SetMeasurementNoiseCovariance({{1e-4}}));
    Tests::Must(filter.SetErrorCovariance({{1.0, 0.0}, {0.0, 1.0}}));
    filter.SetResidualFunction(WrappedResidual);

    // The object is really at (-10, -0.5): the bearing is about -pi + 0.05 where the estimate says pi - 0.05
    ASSERT_TRUE(Tests::Succeeded(filter.Update({std::atan2(-0.5, -10.0)})));

    // Without the wrap, the innovation would be about -2 pi and the estimate would be thrown far away
    EXPECT_LT(std::abs(filter.GetState()[0] + 10.0), 1.0);
    EXPECT_LT(std::abs(filter.GetState()[1]), 1.0);
}

TEST(ExtendedKalmanFilterWrap, TheStateNormalizerAdjustsTheStateAfterAPrediction) {
    Core::ExtendedKalmanFilter filter = Tests::Must(Core::ExtendedKalmanFilter::Create(1, 1));
    Tests::Must(filter.SetStateTransitionFunction(
        [](const Core::Vector &state, const Core::Vector &) { return Core::Vector{state[0] + 1.0}; }));
    Tests::Must(filter.SetStateJacobianFunction([](const Core::Vector &) { return Core::Matrix{{1.0}}; }));
    Tests::Must(filter.SetObservationFunction([](const Core::Vector &state) { return state; }));
    Tests::Must(filter.SetObservationJacobianFunction([](const Core::Vector &) { return Core::Matrix{{1.0}}; }));
    Tests::Must(filter.InitializeState({Core::Pi - 0.5}));
    Tests::Must(filter.SetProcessNoiseCovariance({{0.0}}));
    Tests::Must(filter.SetMeasurementNoiseCovariance({{1.0}}));
    Tests::Must(filter.SetErrorCovariance({{1.0}}));
    filter.SetStateNormalizer([](Core::Vector &state) { state[0] = Core::WrapAngle(state[0]); });

    ASSERT_TRUE(Tests::Succeeded(filter.Predict()));

    EXPECT_NEAR(filter.GetState()[0], -Core::Pi + 0.5, Tolerance);
}
