/**
 * @file    kalman_math.h
 * @brief   Equations and checks shared by the Kalman filter and the Extended Kalman filter.
 */

#ifndef KALMAN_CPP_KALMAN_MATH_H
#define KALMAN_CPP_KALMAN_MATH_H

#include "core/result.h"
#include "matrix/matrix.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace Core {

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

// Prediction
Matrix PredictCovariance(const Matrix &transition, const Matrix &covariance, const Matrix &process_noise);

// Correction
[[nodiscard]] Result<Estimate> CorrectEstimate(const Estimate &prior, const Matrix &observation,
                                               const Matrix &measurement_noise, const Vector &innovation);

} // namespace Core

#endif // KALMAN_CPP_KALMAN_MATH_H
