/**
 * @file    kalman_math.cpp
 * @brief   Equations and checks shared by the Kalman filter and the Extended Kalman filter.
 */

#include "kalman_math.h"

#include "core/text.h"

namespace Core {

/**
 * @brief   Checks that a matrix has the shape the filter expects.
 * @param[in] name    What the matrix is, used in the message, such as "State transition matrix".
 * @param[in] matrix  Matrix to check.
 * @param[in] rows    Expected number of rows.
 * @param[in] cols    Expected number of columns.
 * @return  Nothing when the shape fits, otherwise a message with the expected and the actual shape.
 */
Result<void> CheckShape(const std::string_view name, const Matrix &matrix, const std::size_t rows,
                        const std::size_t cols) {
    if (matrix.GetRows() != rows || matrix.GetCols() != cols) {
        return Failure{Concat(name, " must be ", rows, "x", cols, ", not ", matrix.GetRows(), "x", matrix.GetCols())};
    }
    return {};
}

/**
 * @brief   Propagates the uncertainty of the state through one step of the model.
 * @param[in] transition     State transition matrix `F`, or its Jacobian for a non-linear model.
 * @param[in] covariance     Error covariance `P` before the step.
 * @param[in] process_noise  Process noise covariance `Q`.
 * @return  The error covariance after the step, `F P F^T + Q`.
 * @note    The shapes must fit: `F` and `P` square and of the same size, `Q` of that size too.
 */
Matrix PredictCovariance(const Matrix &transition, const Matrix &covariance, const Matrix &process_noise) {
    return transition * covariance * transition.Transposed() + process_noise;
}

/**
 * @brief   Corrects a state estimate with the innovation of a measurement.
 * @details Computes the innovation covariance `S = H P H^T + R`, the gain `K = P H^T S^-1`, the corrected state
 *          `x + K y` and the corrected covariance `(I - K H) P`.
 * @param[in] prior              Estimate before the measurement.
 * @param[in] observation        Observation matrix `H`, or its Jacobian for a non-linear model.
 * @param[in] measurement_noise  Measurement noise covariance `R`.
 * @param[in] innovation         Difference `y` between the measurement and the one the estimate predicts.
 * @return  The corrected estimate, or a message saying that the innovation covariance cannot be inverted.
 * @note    The shapes must fit: `H` is `measurements x states`, `P` is `states x states`, `R` is
 *          `measurements x measurements` and `y` has one element per measurement.
 */
Result<Estimate> CorrectEstimate(const Estimate &prior, const Matrix &observation, const Matrix &measurement_noise,
                                 const Vector &innovation) {
    const Matrix observation_transposed = observation.Transposed();
    const Matrix innovation_covariance = observation * prior.Covariance * observation_transposed + measurement_noise;

    const Result<Matrix> innovation_covariance_inverse = innovation_covariance.Inverse();
    if (!innovation_covariance_inverse) {
        return Failure{"The innovation covariance cannot be inverted: " + innovation_covariance_inverse.Error()};
    }

    const Matrix gain = prior.Covariance * observation_transposed * *innovation_covariance_inverse;
    const Vector correction = gain * innovation;

    Estimate corrected{prior.State, Matrix()};
    for (std::size_t index = 0; index < corrected.State.size(); index++) {
        corrected.State[index] += correction[index];
    }
    corrected.Covariance = (Matrix::Identity(prior.Covariance.GetRows()) - gain * observation) * prior.Covariance;
    return corrected;
}

} // namespace Core
