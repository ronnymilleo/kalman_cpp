/**
 * @file    kalman_math.h
 * @brief   Equations and checks shared by the Kalman filter and the Extended Kalman filter.
 */

#ifndef KALMAN_CPP_KALMAN_MATH_H
#define KALMAN_CPP_KALMAN_MATH_H

#include "core/result.h"
#include "matrix/matrix.h"

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

namespace Core {

/**
 * @brief   Computes the innovation `y`, the difference between a measurement and the one the estimate predicts.
 * @details A plain subtraction is wrong for angles: a bearing of 179 degrees measured against an expected -179 is a 2
 *          degree error, not 358. A residual function wraps those components, see AngleDifference().
 * @param   measurement           The measurement.
 * @param   expected_measurement  What the estimate predicts, with the same size.
 * @return  The difference, with the same size.
 */
using ResidualFunction = std::function<Vector(const Vector &measurement, const Vector &expected_measurement)>;

/**
 * @brief   Brings a state back to its valid set after the filter moved it.
 * @details Angles must stay within one turn, and a quaternion must keep unit length, but a prediction or a correction
 *          knows nothing about that. The filter calls this on the new state before storing it.
 * @param   state  State to adjust in place.
 */
using StateNormalizer = std::function<void(Vector &state)>;

/**
 * @struct  Estimate
 * @brief   State of a system together with the uncertainty of that state.
 */
struct Estimate {
    Vector State;
    Matrix Covariance;
};

// Validation
[[nodiscard]] Result<void> CheckShape(std::string_view name, const Matrix &matrix, std::size_t rows, std::size_t cols);

// Innovation
[[nodiscard]] Result<Vector> ComputeInnovation(const Vector &measurement, const Vector &expected_measurement,
                                               const ResidualFunction &residual);

// Prediction
Matrix PredictCovariance(const Matrix &transition, const Matrix &covariance, const Matrix &process_noise);

// Correction
[[nodiscard]] Result<Estimate> CorrectEstimate(const Estimate &prior, const Matrix &observation,
                                               const Matrix &measurement_noise, const Vector &innovation);

} // namespace Core

#endif // KALMAN_CPP_KALMAN_MATH_H
