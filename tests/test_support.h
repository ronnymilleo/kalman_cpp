/**
 * @file    test_support.h
 * @brief   Helpers shared by the unit tests: assertions, a repeatable noise source and ready-made filter models.
 */

#ifndef KALMAN_CPP_TEST_SUPPORT_H
#define KALMAN_CPP_TEST_SUPPORT_H

#include "filters/kalman_filter.h"
#include "matrix/matrix.h"

#include <cstdint>
#include <expected>
#include <gtest/gtest.h>
#include <ostream>
#include <random>
#include <string>

namespace Core {

// Lets GoogleTest print matrices in failure messages
void PrintTo(const Matrix &matrix, std::ostream *stream);

} // namespace Core

namespace Tests {

/**
 * @class   NoiseSource
 * @brief   Uniform noise that is the same on every platform.
 * @details The distributions of the standard library give different numbers on different implementations, so a
 *          tracking test that depends on them could pass on one platform and fail on another. The raw output of
 *          std::mt19937 is fixed by the standard, so the noise is built from it directly.
 */
class NoiseSource {
public:
    explicit NoiseSource(std::uint32_t seed);

    double Symmetric(double amplitude);

private:
    std::mt19937 m_Generator;
};

/**
 * @struct  Model
 * @brief   Matrices and initial state needed to configure a linear filter.
 */
struct Model {
    Core::Matrix F;
    Core::Matrix H;
    Core::Matrix Q;
    Core::Matrix R;
    Core::Matrix P;
    Core::Vector InitialState;
};

// Assertions
::testing::AssertionResult MatrixNear(const Core::Matrix &expected, const Core::Matrix &actual, double tolerance);
::testing::AssertionResult VectorNear(const Core::Vector &expected, const Core::Vector &actual, double tolerance);

/**
 * @brief   Turns the outcome of an operation into a GoogleTest assertion result that shows the error message.
 * @param[in] outcome  Result of an operation of the library.
 * @return  Success when the operation succeeded, otherwise a failure carrying the message.
 */
template <typename T>::testing::AssertionResult Succeeded(const std::expected<T, std::string> &outcome) {
    if (outcome) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << outcome.error();
}

// Models
Model PositionVelocityModel(double dt, double measurement_variance, double initial_variance);
Model PlanarPositionVelocityModel(double dt, double process_variance, double measurement_variance,
                                  double initial_variance);
Core::Matrix PositionVelocityNoise(double dt, double process_noise_sigma);
Core::KalmanFilter StartedFilter(const Model &model);

} // namespace Tests

#endif // KALMAN_CPP_TEST_SUPPORT_H
