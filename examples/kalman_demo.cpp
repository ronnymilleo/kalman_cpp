/**
 * @file    kalman_demo.cpp
 * @brief   Demonstrations of the linear Kalman filter: tracking on a line and on a plane, and noise sensitivity.
 * @details Each demonstration knows the truth it simulates, so it compares the estimate with it and with the raw
 *          measurements, and fails with a message (and exit code 1) when the filter does no better than the sensor.
 */

#include "demo_support.h"
#include "filters/kalman_filter.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Reported by main when a demonstration fails
constexpr int ExitSuccess = 0;
constexpr int ExitFailure = 1;

// The filter must cut the error of the raw measurements at least to this share to pass its check
constexpr double RequiredErrorShare = 0.5;

// The velocity is never measured, so the check is an absolute limit, small next to the velocities in the scenarios
constexpr double MaxVelocityError = 0.3;

// Every setter reports a problem as a message; chaining them with AndThen stops at the first one
Core::Result<void> Configure(Core::KalmanFilter &filter, const Core::Vector &initial_state, const Core::Matrix &f,
                             const Core::Matrix &h, const Core::Matrix &q, const Core::Matrix &r,
                             const Core::Matrix &p) {
    return filter.InitializeState(initial_state)
        .AndThen([&] { return filter.SetStateTransitionMatrix(f); })
        .AndThen([&] { return filter.SetObservationMatrix(h); })
        .AndThen([&] { return filter.SetProcessNoiseCovariance(q); })
        .AndThen([&] { return filter.SetMeasurementNoiseCovariance(r); })
        .AndThen([&] { return filter.SetErrorCovariance(p); });
}

Core::Result<void> DemonstrateTracking1d() {
    std::cout << "=== Kalman filter: position and velocity on a line ===\n\n";

    constexpr double TimeStep = 0.1;
    constexpr double TrueVelocity = 2.0;
    constexpr double ProcessNoise = 0.02;
    constexpr double MeasurementNoise = 0.5;
    constexpr int Steps = 100;

    Core::Result<Core::KalmanFilter> created = Core::KalmanFilter::Create(2, 1);
    if (!created) {
        return Core::Failure{created.Error()};
    }
    Core::KalmanFilter &filter = *created;

    // The velocity starts as a poor guess, with a high uncertainty to match
    const Core::Result<void> configured =
        Configure(filter, {0.0, 0.0},
                  // Constant velocity: the position grows by velocity * dt at every step
                  {{1.0, TimeStep}, {0.0, 1.0}},
                  // Only the position is measured
                  {{1.0, 0.0}},
                  {{ProcessNoise * ProcessNoise * TimeStep * TimeStep, ProcessNoise * ProcessNoise * TimeStep},
                   {ProcessNoise * ProcessNoise * TimeStep, ProcessNoise * ProcessNoise}},
                  {{MeasurementNoise * MeasurementNoise}}, {{10.0, 0.0}, {0.0, 10.0}});
    if (!configured) {
        return configured;
    }

    Demo::NoiseSource noise(Demo::NoiseSeed);

    std::cout << "Step | True Pos | Meas Pos | Est Pos | Est Vel | Pos Err | Vel Err\n";
    std::cout << "-----+----------+----------+---------+---------+---------+--------\n";

    std::vector<double> estimate_errors;
    std::vector<double> raw_errors;
    double true_position = 0.0;
    for (int step = 0; step < Steps; step++) {
        true_position += TrueVelocity * TimeStep;
        const double measured_position = true_position + noise.Gaussian(MeasurementNoise);

        if (const auto predicted = filter.Predict(); !predicted) {
            return predicted;
        }
        if (const auto updated = filter.Update({measured_position}); !updated) {
            return updated;
        }

        const Core::Vector &state = filter.GetState();
        if (step >= Demo::SettlingSteps) {
            estimate_errors.push_back(state[0] - true_position);
            raw_errors.push_back(measured_position - true_position);
        }
        if (step % 5 == 0 || step < 3 || step >= Steps - 3) {
            std::cout << Demo::Format("{:4} | {:8.3f} | {:8.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f}\n", step,
                                      true_position, measured_position, state[0], state[1],
                                      std::abs(state[0] - true_position), std::abs(state[1] - TrueVelocity));
        }
    }

    std::cout << "\nChecks over the steps after the first " << Demo::SettlingSteps << ":\n";
    if (const auto better = Demo::CheckImprovement("position error", Demo::RootMeanSquare(estimate_errors),
                                                   Demo::RootMeanSquare(raw_errors), RequiredErrorShare);
        !better) {
        return better;
    }
    const double velocity_error = std::abs(filter.GetState()[1] - TrueVelocity);
    std::cout << Demo::Format("final velocity error: {:.4f} (the true velocity is {})\n", velocity_error, TrueVelocity);
    if (velocity_error > MaxVelocityError) {
        return Core::Failure{
            Demo::Format("The final velocity error ({:.4f}) exceeds {}", velocity_error, MaxVelocityError)};
    }

    std::cout << "\nThe velocity is estimated from position measurements alone.\n\n";
    return {};
}

Core::Result<void> DemonstrateTracking2d() {
    std::cout << "=== Kalman filter: position and velocity on a plane ===\n\n";

    constexpr double TimeStep = 0.2;
    constexpr double TrueVx = 1.5;
    constexpr double TrueVy = -1.0;
    constexpr double ProcessNoise = 0.05;
    constexpr double MeasurementNoiseX = 0.3;
    constexpr double MeasurementNoiseY = 0.4;
    constexpr int Steps = 80;

    Core::Result<Core::KalmanFilter> created = Core::KalmanFilter::Create(4, 2);
    if (!created) {
        return Core::Failure{created.Error()};
    }
    Core::KalmanFilter &filter = *created;

    constexpr double Q = ProcessNoise * ProcessNoise;
    // clang-format off
    const Core::Result<void> configured = Configure(
        filter, {0.0, 0.0, 0.0, 0.0},
        // The states are x, y, vx and vy
        {{1.0, 0.0, TimeStep, 0.0},
         {0.0, 1.0, 0.0, TimeStep},
         {0.0, 0.0, 1.0, 0.0},
         {0.0, 0.0, 0.0, 1.0}},
        // Both positions are measured
        {{1.0, 0.0, 0.0, 0.0},
         {0.0, 1.0, 0.0, 0.0}},
        {{Q * TimeStep * TimeStep, 0.0, Q * TimeStep, 0.0},
         {0.0, Q * TimeStep * TimeStep, 0.0, Q * TimeStep},
         {Q * TimeStep, 0.0, Q, 0.0},
         {0.0, Q * TimeStep, 0.0, Q}},
        {{MeasurementNoiseX * MeasurementNoiseX, 0.0},
         {0.0, MeasurementNoiseY * MeasurementNoiseY}},
        {{5.0, 0.0, 0.0, 0.0},
         {0.0, 5.0, 0.0, 0.0},
         {0.0, 0.0, 5.0, 0.0},
         {0.0, 0.0, 0.0, 5.0}});
    // clang-format on
    if (!configured) {
        return configured;
    }

    Demo::NoiseSource noise(Demo::NoiseSeed);

    std::cout << "Step | True Pos (x,y) | Est Pos (x,y)  | Est Vel (x,y)  | Pos Error\n";
    std::cout << "-----+----------------+----------------+----------------+----------\n";

    std::vector<double> estimate_errors;
    std::vector<double> raw_errors;
    double true_x = 0.0;
    double true_y = 0.0;
    for (int step = 0; step < Steps; step++) {
        true_x += TrueVx * TimeStep;
        true_y += TrueVy * TimeStep;
        const double measured_x = true_x + noise.Gaussian(MeasurementNoiseX);
        const double measured_y = true_y + noise.Gaussian(MeasurementNoiseY);

        if (const auto predicted = filter.Predict(); !predicted) {
            return predicted;
        }
        if (const auto updated = filter.Update({measured_x, measured_y}); !updated) {
            return updated;
        }

        const Core::Vector &state = filter.GetState();
        if (step >= Demo::SettlingSteps) {
            estimate_errors.push_back(std::hypot(state[0] - true_x, state[1] - true_y));
            raw_errors.push_back(std::hypot(measured_x - true_x, measured_y - true_y));
        }
        if (step % 3 == 0 || step < 2 || step >= Steps - 2) {
            std::cout << Demo::Format("{:4} | ({:5.2f},{:6.2f}) | ({:5.2f},{:6.2f}) | ({:5.2f},{:6.2f}) | {:9.3f}\n",
                                      step, true_x, true_y, state[0], state[1], state[2], state[3],
                                      std::hypot(state[0] - true_x, state[1] - true_y));
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
    std::cout << Demo::Format("final velocity error: {:.4f}\n", velocity_error);
    if (velocity_error > MaxVelocityError) {
        return Core::Failure{
            Demo::Format("The final velocity error ({:.4f}) exceeds {}", velocity_error, MaxVelocityError)};
    }

    std::cout << "\nBoth positions and both velocities are estimated from the positions alone.\n\n";
    return {};
}

Core::Result<void> DemonstrateNoiseSensitivity() {
    std::cout << "=== Kalman filter: noise sensitivity ===\n\n";

    constexpr double TimeStep = 0.1;
    constexpr double TrueVelocity = 1.0;
    constexpr int Steps = 60;

    std::cout << "Noise Level | Raw Pos Error | Filter Pos Error | Mean Vel Error\n";
    std::cout << "------------+---------------+------------------+---------------\n";

    // A noisier sensor leaves less to gain, so these checks are looser than in the other demonstrations: the filter
    // only has to beat the raw measurements, and its velocity error has to stay below the noise of the sensor
    for (const double noise_level : {0.1, 0.5, 1.0, 2.0}) {
        Core::Result<Core::KalmanFilter> created = Core::KalmanFilter::Create(2, 1);
        if (!created) {
            return Core::Failure{created.Error()};
        }
        Core::KalmanFilter &filter = *created;

        const Core::Result<void> configured =
            Configure(filter, {0.0, 0.0}, {{1.0, TimeStep}, {0.0, 1.0}}, {{1.0, 0.0}}, {{0.01, 0.0}, {0.0, 0.01}},
                      {{noise_level * noise_level}}, {{1.0, 0.0}, {0.0, 1.0}});
        if (!configured) {
            return configured;
        }

        Demo::NoiseSource noise(Demo::NoiseSeed);

        std::vector<double> estimate_errors;
        std::vector<double> raw_errors;
        std::vector<double> velocity_errors;
        double true_position = 0.0;
        for (int step = 0; step < Steps; step++) {
            true_position += TrueVelocity * TimeStep;
            const double measured_position = true_position + noise.Gaussian(noise_level);
            if (const auto predicted = filter.Predict(); !predicted) {
                return predicted;
            }
            if (const auto updated = filter.Update({measured_position}); !updated) {
                return updated;
            }
            if (step >= Demo::SettlingSteps) {
                estimate_errors.push_back(filter.GetState()[0] - true_position);
                raw_errors.push_back(measured_position - true_position);
                velocity_errors.push_back(filter.GetState()[1] - TrueVelocity);
            }
        }

        const double estimate_error = Demo::RootMeanSquare(estimate_errors);
        const double raw_error = Demo::RootMeanSquare(raw_errors);
        const double velocity_error = Demo::RootMeanSquare(velocity_errors);
        std::cout << Demo::Format("{:11.1f} | {:13.4f} | {:16.4f} | {:14.4f}\n", noise_level, raw_error, estimate_error,
                                  velocity_error);

        if (estimate_error > raw_error) {
            return Core::Failure{
                Demo::Format("At noise level {} the filter ({:.4f}) is worse than the raw measurements ({:.4f})",
                             noise_level, estimate_error, raw_error)};
        }
        if (velocity_error > noise_level) {
            return Core::Failure{Demo::Format("At noise level {} the velocity error ({:.4f}) is larger than the noise",
                                              noise_level, velocity_error)};
        }
    }

    std::cout << "\nThe filter beats the raw measurements at every noise level and keeps the velocity error below the "
                 "sensor noise.\n\n";
    return {};
}

} // namespace

int main() {
    std::cout << "Kalman filter demonstrations\n";
    std::cout << "============================\n\n";

    for (const auto &demonstration : {DemonstrateTracking1d, DemonstrateTracking2d, DemonstrateNoiseSensitivity}) {
        if (const Core::Result<void> outcome = demonstration(); !outcome) {
            std::cerr << "Demonstration failed: " << outcome.Error() << "\n";
            return ExitFailure;
        }
    }

    std::cout << "All demonstrations completed.\n";
    return ExitSuccess;
}
