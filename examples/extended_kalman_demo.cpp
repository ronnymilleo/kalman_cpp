/**
 * @file    extended_kalman_demo.cpp
 * @brief   Demonstrations of the Extended Kalman filter: radar tracking and a damped pendulum, each checking itself.
 * @details Both demonstrations know the truth they simulate, so they compare the estimate with it and with the raw
 *          measurements, and fail with a message (and exit code 1) when the filter does no better than the sensor.
 */

#include "demo_support.h"
#include "filters/extended_kalman_filter.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

// The filter must cut the error of the raw measurements at least to this share to pass its check
constexpr double RequiredErrorShare = 0.5;

// Reported by main when a demonstration fails
constexpr int ExitSuccess = 0;
constexpr int ExitFailure = 1;

// Every setter reports a problem as a message; chaining them with AndThen stops at the first one
Core::Result<void> Configure(Core::ExtendedKalmanFilter &filter, const Core::Vector &initial_state,
                             const Core::ExtendedKalmanFilter::StateTransitionFunction &f,
                             const Core::ExtendedKalmanFilter::StateJacobianFunction &jf,
                             const Core::ExtendedKalmanFilter::ObservationFunction &h,
                             const Core::ExtendedKalmanFilter::ObservationJacobianFunction &jh, const Core::Matrix &q,
                             const Core::Matrix &r, const Core::Matrix &p) {
    return filter.InitializeState(initial_state)
        .AndThen([&] { return filter.SetStateTransitionFunction(f); })
        .AndThen([&] { return filter.SetStateJacobianFunction(jf); })
        .AndThen([&] { return filter.SetObservationFunction(h); })
        .AndThen([&] { return filter.SetObservationJacobianFunction(jh); })
        .AndThen([&] { return filter.SetProcessNoiseCovariance(q); })
        .AndThen([&] { return filter.SetMeasurementNoiseCovariance(r); })
        .AndThen([&] { return filter.SetErrorCovariance(p); });
}

Core::Result<void> DemonstrateRadarTracking() {
    std::cout << "=== Extended Kalman filter: radar tracking an aircraft ===\n\n";

    static constexpr double TimeStep = 1.0;
    constexpr double RangeNoise = 0.5;
    constexpr double BearingNoise = 0.01;
    // Variance of the small accelerations the constant velocity model ignores; the aircraft does not maneuver
    constexpr double AccelerationVariance = 1e-4;
    constexpr int Steps = 60;

    // The aircraft flies in a straight line at constant velocity past a radar at the origin. The radar reports range
    // and bearing, which are non-linear functions of the position the filter estimates. The track stays within
    // +-pi of bearing, so the bearing never wraps around
    constexpr double StartX = 40.0;
    constexpr double StartY = 20.0;
    constexpr double TrueVx = -1.0;
    constexpr double TrueVy = 0.5;

    Core::Result<Core::ExtendedKalmanFilter> created = Core::ExtendedKalmanFilter::Create(4, 2);
    if (!created) {
        return Core::Failure{created.Error()};
    }
    Core::ExtendedKalmanFilter &filter = *created;

    // The states are x, y, vx and vy. The first guess is off by several units in position and does not know the
    // velocity
    constexpr double Q = AccelerationVariance;
    constexpr double Dt2 = TimeStep * TimeStep;
    constexpr double Dt3 = Dt2 * TimeStep;
    constexpr double Dt4 = Dt2 * Dt2;
    // clang-format off
    const Core::Result<void> configured = Configure(
        filter, {StartX - 5.0, StartY + 5.0, 0.0, 0.0},
        [](const Core::Vector &state, const Core::Vector & /*control*/) {
            return Core::Vector{state[0] + state[2] * TimeStep, state[1] + state[3] * TimeStep, state[2], state[3]};
        },
        [](const Core::Vector & /*state*/) {
            return Core::Matrix{{1.0, 0.0, TimeStep, 0.0},
                                {0.0, 1.0, 0.0, TimeStep},
                                {0.0, 0.0, 1.0, 0.0},
                                {0.0, 0.0, 0.0, 1.0}};
        },
        [](const Core::Vector &state) { return Core::Vector{std::hypot(state[0], state[1]), std::atan2(state[1], state[0])}; },
        [](const Core::Vector &state) {
            const double range_squared = state[0] * state[0] + state[1] * state[1];
            const double range = std::sqrt(range_squared);
            return Core::Matrix{{state[0] / range, state[1] / range, 0.0, 0.0},
                                {-state[1] / range_squared, state[0] / range_squared, 0.0, 0.0}};
        },
        {{Q * Dt4 / 4.0, 0.0, Q * Dt3 / 2.0, 0.0},
         {0.0, Q * Dt4 / 4.0, 0.0, Q * Dt3 / 2.0},
         {Q * Dt3 / 2.0, 0.0, Q * Dt2, 0.0},
         {0.0, Q * Dt3 / 2.0, 0.0, Q * Dt2}},
        {{RangeNoise * RangeNoise, 0.0}, {0.0, BearingNoise * BearingNoise}},
        {{100.0, 0.0, 0.0, 0.0}, {0.0, 100.0, 0.0, 0.0}, {0.0, 0.0, 10.0, 0.0}, {0.0, 0.0, 0.0, 10.0}});
    // clang-format on
    if (!configured) {
        return configured;
    }

    Demo::NoiseSource noise(Demo::NoiseSeed);

    std::cout << "Step | True Pos (x,y)  | Meas Pos (x,y)  | Est Pos (x,y)   | Est Vel (x,y) | Est Err | Raw Err\n";
    std::cout << "-----+-----------------+-----------------+-----------------+---------------+---------+--------\n";

    std::vector<double> estimate_errors;
    std::vector<double> raw_errors;
    double true_x = StartX;
    double true_y = StartY;
    for (int step = 0; step < Steps; step++) {
        true_x += TrueVx * TimeStep;
        true_y += TrueVy * TimeStep;

        // What the radar reports, and the position those numbers point to without any filtering
        const double measured_range = std::hypot(true_x, true_y) + noise.Gaussian(RangeNoise);
        const double measured_bearing = std::atan2(true_y, true_x) + noise.Gaussian(BearingNoise);
        const double raw_x = measured_range * std::cos(measured_bearing);
        const double raw_y = measured_range * std::sin(measured_bearing);

        if (const auto predicted = filter.Predict(); !predicted) {
            return predicted;
        }
        if (const auto updated = filter.Update({measured_range, measured_bearing}); !updated) {
            return updated;
        }

        const Core::Vector &state = filter.GetState();
        const double estimate_error = std::hypot(state[0] - true_x, state[1] - true_y);
        const double raw_error = std::hypot(raw_x - true_x, raw_y - true_y);
        if (step >= Demo::SettlingSteps) {
            estimate_errors.push_back(estimate_error);
            raw_errors.push_back(raw_error);
        }
        if (step % 10 == 0 || step < 3) {
            std::cout << Demo::Format("{:4} | ({:6.2f},{:6.2f}) | ({:6.2f},{:6.2f}) | ({:6.2f},{:6.2f}) | "
                                      "({:5.2f},{:5.2f}) | {:7.3f} | {:7.3f}\n",
                                      step, true_x, true_y, raw_x, raw_y, state[0], state[1], state[2], state[3],
                                      estimate_error, raw_error);
        }
    }

    std::cout << "\nChecks over the steps after the first " << Demo::SettlingSteps << ":\n";
    if (const auto better = Demo::CheckImprovement("position error", Demo::RootMeanSquare(estimate_errors),
                                                   Demo::RootMeanSquare(raw_errors), RequiredErrorShare);
        !better) {
        return better;
    }
    const Core::Vector &state = filter.GetState();
    const double velocity_error = std::hypot(state[2] - TrueVx, state[3] - TrueVy);
    std::cout << Demo::Format("final velocity error: {:.4f} (the filter was never told the velocity)\n",
                              velocity_error);
    constexpr double MaxVelocityError = 0.1;
    if (velocity_error > MaxVelocityError) {
        return Core::Failure{
            Demo::Format("The final velocity error ({:.4f}) exceeds {}", velocity_error, MaxVelocityError)};
    }

    std::cout << "\nThe filter turns noisy range and bearing readings into a precise position and a velocity.\n\n";
    return {};
}

Core::Result<void> DemonstratePendulumTracking() {
    std::cout << "=== Extended Kalman filter: damped pendulum ===\n\n";

    static constexpr double TimeStep = 0.05;
    static constexpr double Gravity = 9.81;
    static constexpr double Length = 1.0;
    static constexpr double Damping = 0.1;
    constexpr double ProcessNoise = 0.01;
    constexpr double MeasurementNoise = 0.05;
    constexpr int Steps = 200;

    Core::Result<Core::ExtendedKalmanFilter> created = Core::ExtendedKalmanFilter::Create(2, 1);
    if (!created) {
        return Core::Failure{created.Error()};
    }
    Core::ExtendedKalmanFilter &filter = *created;

    // One step of the pendulum, shared by the simulated truth and by the model of the filter. The velocity is updated
    // first and the angle moves with the new velocity (semi-implicit Euler): the plain explicit scheme adds energy at
    // every step and makes a damped pendulum swing higher and higher
    const auto step_pendulum = [](const Core::Vector &state) {
        const double acceleration = -(Gravity / Length) * std::sin(state[0]) - Damping * state[1];
        const double velocity = state[1] + acceleration * TimeStep;
        return Core::Vector{state[0] + velocity * TimeStep, velocity};
    };

    // The states are the angle and the angular velocity; the sensor measures the angle only
    // clang-format off
    const Core::Result<void> configured = Configure(
        filter, {0.5, 0.0},
        [&step_pendulum](const Core::Vector &state, const Core::Vector & /*control*/) { return step_pendulum(state); },
        [](const Core::Vector &state) {
            const double velocity_by_angle = -(Gravity / Length) * std::cos(state[0]) * TimeStep;
            const double velocity_by_velocity = 1.0 - Damping * TimeStep;
            return Core::Matrix{{1.0 + TimeStep * velocity_by_angle, TimeStep * velocity_by_velocity},
                                {velocity_by_angle, velocity_by_velocity}};
        },
        [](const Core::Vector &state) { return Core::Vector{state[0]}; },
        [](const Core::Vector & /*state*/) { return Core::Matrix{{1.0, 0.0}}; },
        {{ProcessNoise * ProcessNoise, 0.0}, {0.0, ProcessNoise * ProcessNoise}},
        {{MeasurementNoise * MeasurementNoise}},
        {{0.1, 0.0}, {0.0, 0.1}});
    // clang-format on
    if (!configured) {
        return configured;
    }

    Demo::NoiseSource noise(Demo::NoiseSeed);

    std::cout << "Step | True Angle | True Vel | Meas Angle | Est Angle | Est Vel | Angle Err\n";
    std::cout << "-----+------------+----------+------------+-----------+---------+----------\n";

    std::vector<double> estimate_errors;
    std::vector<double> raw_errors;
    std::vector<double> velocity_errors;
    double true_angle = 0.5;
    double true_velocity = 0.0;
    for (int step = 0; step < Steps; step++) {
        const Core::Vector truth = step_pendulum({true_angle, true_velocity});
        true_angle = truth[0];
        true_velocity = truth[1];
        const double measured_angle = true_angle + noise.Gaussian(MeasurementNoise);

        if (const auto predicted = filter.Predict(); !predicted) {
            return predicted;
        }
        if (const auto updated = filter.Update({measured_angle}); !updated) {
            return updated;
        }

        const Core::Vector &state = filter.GetState();
        if (step >= Demo::SettlingSteps) {
            estimate_errors.push_back(state[0] - true_angle);
            raw_errors.push_back(measured_angle - true_angle);
            velocity_errors.push_back(state[1] - true_velocity);
        }
        if (step % 20 == 0 || step < 5) {
            std::cout << Demo::Format("{:4} | {:10.4f} | {:8.4f} | {:10.4f} | {:9.4f} | {:7.4f} | {:9.4f}\n", step,
                                      true_angle, true_velocity, measured_angle, state[0], state[1],
                                      std::abs(state[0] - true_angle));
        }
    }

    std::cout << "\nChecks over the steps after the first " << Demo::SettlingSteps << ":\n";
    if (const auto better = Demo::CheckImprovement("angle error", Demo::RootMeanSquare(estimate_errors),
                                                   Demo::RootMeanSquare(raw_errors), RequiredErrorShare);
        !better) {
        return better;
    }
    // The sensor reads no velocity, so there are no raw measurements to compare with; the error must simply be small
    // next to the velocities involved, which reach about 1.5 rad/s
    constexpr double MaxVelocityError = 0.1;
    const double velocity_error = Demo::RootMeanSquare(velocity_errors);
    std::cout << Demo::Format("velocity error: {:.4f} rad/s (the sensor never reads the velocity)\n", velocity_error);
    if (velocity_error > MaxVelocityError) {
        return Core::Failure{
            Demo::Format("The velocity error ({:.4f}) exceeds {} rad/s", velocity_error, MaxVelocityError)};
    }

    std::cout << "\nThe angular velocity is estimated from angle measurements alone.\n\n";
    return {};
}

} // namespace

int main() {
    std::cout << "Extended Kalman filter demonstrations\n";
    std::cout << "=====================================\n\n";

    for (const auto &demonstration : {DemonstrateRadarTracking, DemonstratePendulumTracking}) {
        if (const Core::Result<void> outcome = demonstration(); !outcome) {
            std::cerr << "Demonstration failed: " << outcome.Error() << "\n";
            return ExitFailure;
        }
    }

    std::cout << "All demonstrations completed.\n";
    return ExitSuccess;
}
