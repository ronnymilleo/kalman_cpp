/**
 * @file    test_support.cpp
 * @brief   Helpers shared by the unit tests: assertions, a repeatable noise source and ready-made filter models.
 */

#include "test_support.h"

#include <cmath>

namespace Core {

/**
 * @brief   Prints a matrix row by row, for GoogleTest failure messages.
 * @param[in]  matrix  Matrix to print.
 * @param[out] stream  Stream to print to.
 */
void PrintTo(const Matrix &matrix, std::ostream *stream) {
    *stream << matrix.GetRows() << "x" << matrix.GetCols() << " {";
    for (std::size_t row = 0; row < matrix.GetRows(); row++) {
        *stream << (row == 0 ? "{" : ", {");
        for (std::size_t col = 0; col < matrix.GetCols(); col++) {
            *stream << (col == 0 ? "" : ", ") << matrix(row, col);
        }
        *stream << "}";
    }
    *stream << "}";
}

} // namespace Core

namespace Tests {

namespace {

// 2^32, the number of values std::mt19937 can produce
constexpr double GeneratorRange = 4294967296.0;

} // namespace

/**
 * @brief   Creates a noise source.
 * @param[in] seed  Starting point; the same seed always gives the same sequence.
 */
NoiseSource::NoiseSource(const std::uint32_t seed) : m_Generator(seed) {
}

/**
 * @brief   Draws a number uniformly distributed around zero.
 * @param[in] amplitude  Half the width of the interval.
 * @return  A value in `[-amplitude, amplitude)`.
 */
double NoiseSource::Symmetric(const double amplitude) {
    const double unit = static_cast<double>(m_Generator()) / GeneratorRange;
    return (2.0 * unit - 1.0) * amplitude;
}

/**
 * @brief   Compares two matrices element by element.
 * @param[in] expected   Matrix the test expects.
 * @param[in] actual     Matrix the code produced.
 * @param[in] tolerance  Largest accepted difference between two elements.
 * @return  Success when the shapes match and every element is within the tolerance, otherwise a failure that says
 *          where they differ.
 */
::testing::AssertionResult MatrixNear(const Core::Matrix &expected, const Core::Matrix &actual,
                                      const double tolerance) {
    if (!expected.HasSameShape(actual)) {
        return ::testing::AssertionFailure() << "expected a " << expected.GetRows() << "x" << expected.GetCols()
                                             << " matrix but got " << actual.GetRows() << "x" << actual.GetCols();
    }
    for (std::size_t row = 0; row < expected.GetRows(); row++) {
        for (std::size_t col = 0; col < expected.GetCols(); col++) {
            if (std::abs(expected(row, col) - actual(row, col)) > tolerance) {
                return ::testing::AssertionFailure() << "element (" << row << ", " << col << ") is " << actual(row, col)
                                                     << " but " << expected(row, col) << " was expected";
            }
        }
    }
    return ::testing::AssertionSuccess();
}

/**
 * @brief   Compares two vectors element by element.
 * @param[in] expected   Vector the test expects.
 * @param[in] actual     Vector the code produced.
 * @param[in] tolerance  Largest accepted difference between two elements.
 * @return  Success when the sizes match and every element is within the tolerance, otherwise a failure that says
 *          where they differ.
 */
::testing::AssertionResult VectorNear(const Core::Vector &expected, const Core::Vector &actual,
                                      const double tolerance) {
    if (expected.size() != actual.size()) {
        return ::testing::AssertionFailure() << "expected " << expected.size() << " elements but got " << actual.size();
    }
    for (std::size_t index = 0; index < expected.size(); index++) {
        if (std::abs(expected[index] - actual[index]) > tolerance) {
            return ::testing::AssertionFailure()
                   << "element " << index << " is " << actual[index] << " but " << expected[index] << " was expected";
        }
    }
    return ::testing::AssertionSuccess();
}

/**
 * @brief   Describes a position and velocity tracker that measures the position only.
 * @details The process noise starts as a small diagonal; tests that care about it replace `Q`.
 * @param[in] dt                    Time between steps.
 * @param[in] measurement_variance  Variance of the position measurement.
 * @param[in] initial_variance      Starting variance of both states; the starting state is zero.
 * @return  The model.
 */
Model PositionVelocityModel(const double dt, const double measurement_variance, const double initial_variance) {
    return {{{1.0, dt}, {0.0, 1.0}},
            {{1.0, 0.0}},
            {{0.01, 0.0}, {0.0, 0.01}},
            {{measurement_variance}},
            {{initial_variance, 0.0}, {0.0, initial_variance}},
            {0.0, 0.0}};
}

/**
 * @brief   Describes a tracker of position and velocity on two axes that measures both positions.
 * @details The states are `x`, `y`, `vx` and `vy`, and the starting state is zero.
 * @param[in] dt                    Time between steps.
 * @param[in] process_variance      Variance of the velocity noise on each axis.
 * @param[in] measurement_variance  Variance of each position measurement.
 * @param[in] initial_variance      Starting variance of all four states.
 * @return  The model.
 */
Model PlanarPositionVelocityModel(const double dt, const double process_variance, const double measurement_variance,
                                  const double initial_variance) {
    const double q = process_variance;
    // clang-format off
    return {{{1.0, 0.0, dt,  0.0},
                  {0.0, 1.0, 0.0, dt },
                  {0.0, 0.0, 1.0, 0.0},
                  {0.0, 0.0, 0.0, 1.0}},
            {{1.0, 0.0, 0.0, 0.0},
                  {0.0, 1.0, 0.0, 0.0}},
            {{q * dt * dt, 0.0,          q * dt, 0.0   },
                  {0.0,         q * dt * dt,  0.0,    q * dt},
                  {q * dt,      0.0,          q,      0.0   },
                  {0.0,         q * dt,       0.0,    q     }},
            {{measurement_variance, 0.0},
                  {0.0, measurement_variance}},
            {{initial_variance, 0.0, 0.0, 0.0},
                  {0.0, initial_variance, 0.0, 0.0},
                  {0.0, 0.0, initial_variance, 0.0},
                  {0.0, 0.0, 0.0, initial_variance}},
            {0.0, 0.0, 0.0, 0.0}};
    // clang-format on
}

/**
 * @brief   Builds the process noise of a position and velocity pair driven by random velocity changes.
 * @param[in] dt                   Time between steps.
 * @param[in] process_noise_sigma  Standard deviation of the velocity changes.
 * @return  The 2x2 covariance.
 */
Core::Matrix PositionVelocityNoise(const double dt, const double process_noise_sigma) {
    const double variance = process_noise_sigma * process_noise_sigma;
    return {{variance * dt * dt, variance * dt}, {variance * dt, variance}};
}

/**
 * @brief   Sets a filter up with a model and leaves it ready to predict.
 * @details A failure to configure the filter throws, which GoogleTest reports as a failed test with the message.
 * @param[in] model  Model to apply.
 * @return  The filter.
 */
Core::KalmanFilter StartedFilter(const Model &model) {
    Core::KalmanFilter filter = Must(Core::KalmanFilter::Create(model.F.GetRows(), model.H.GetRows()));
    Must(filter.InitializeState(model.InitialState));
    Must(filter.SetStateTransitionMatrix(model.F));
    Must(filter.SetObservationMatrix(model.H));
    Must(filter.SetProcessNoiseCovariance(model.Q));
    Must(filter.SetMeasurementNoiseCovariance(model.R));
    Must(filter.SetErrorCovariance(model.P));
    return filter;
}

} // namespace Tests
