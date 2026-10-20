/**
 * @file    extended_kalman_filter.h
 * @brief   Discrete-time Extended Kalman filter for non-linear systems.
 */

#ifndef KALMAN_CPP_EXTENDED_KALMAN_FILTER_H
#define KALMAN_CPP_EXTENDED_KALMAN_FILTER_H

#include "matrix/matrix.h"

#include <cstddef>
#include <expected>
#include <functional>
#include <optional>
#include <string>

namespace Core {

/**
 * @class   ExtendedKalmanFilter
 * @brief   Estimates the state of a non-linear system from noisy measurements.
 * @details The model is `x(k+1) = f(x(k), u(k)) + w(k)` and `z(k) = h(x(k)) + v(k)`, with process noise
 *          `w ~ N(0, Q)` and measurement noise `v ~ N(0, R)`. The filter linearizes `f` and `h` at every step with
 *          their Jacobians `F = df/dx` and `H = dh/dx`, which the caller supplies together with the functions.
 *
 *          A filter is created with the sizes of the state, the measurement and the optional control input,
 *          configured through the setters and ready to predict and update once the four functions and the error
 *          covariance `P` have been set. Operations that can fail return `std::expected` with a message, and a
 *          filter is left unchanged when an operation fails. The class is not thread safe.
 */
class ExtendedKalmanFilter {
public:
    // Function types
    using StateTransitionFunction = std::function<Vector(const Vector &state, const Vector &control)>;
    using ObservationFunction = std::function<Vector(const Vector &state)>;
    using StateJacobianFunction = std::function<Matrix(const Vector &state)>;
    using ObservationJacobianFunction = std::function<Matrix(const Vector &state)>;

    // Construction
    [[nodiscard]] static std::expected<ExtendedKalmanFilter, std::string>
    Create(std::size_t state_dim, std::size_t measurement_dim, std::size_t control_dim = 0);

    // Model configuration
    [[nodiscard]] std::expected<void, std::string> InitializeState(const Vector &initial_state);
    [[nodiscard]] std::expected<void, std::string> SetStateTransitionFunction(const StateTransitionFunction &function);
    [[nodiscard]] std::expected<void, std::string> SetObservationFunction(const ObservationFunction &function);
    [[nodiscard]] std::expected<void, std::string> SetStateJacobianFunction(const StateJacobianFunction &function);
    [[nodiscard]] std::expected<void, std::string>
    SetObservationJacobianFunction(const ObservationJacobianFunction &function);
    [[nodiscard]] std::expected<void, std::string> SetProcessNoiseCovariance(const Matrix &q);
    [[nodiscard]] std::expected<void, std::string> SetMeasurementNoiseCovariance(const Matrix &r);
    [[nodiscard]] std::expected<void, std::string> SetErrorCovariance(const Matrix &p);

    // Filtering
    [[nodiscard]] std::expected<void, std::string> Predict(const Vector &control = {});
    [[nodiscard]] std::expected<void, std::string> Update(const Vector &measurement);

    // Estimates
    const Vector &GetState() const;
    const Matrix &GetErrorCovariance() const;
    std::optional<double> GetStateElement(std::size_t index) const;
    bool IsInitialized() const;
    bool AreAllFunctionsSet() const;

    // Lifecycle
    void Reset();

private:
    // Sizes
    std::size_t m_StateDim;
    std::size_t m_MeasurementDim;
    std::size_t m_ControlDim;

    // Estimate
    Vector m_State;
    Matrix m_ErrorCovariance;

    // Model
    StateTransitionFunction m_StateTransitionFunction;
    ObservationFunction m_ObservationFunction;
    StateJacobianFunction m_StateJacobianFunction;
    ObservationJacobianFunction m_ObservationJacobianFunction;
    Matrix m_ProcessNoise;
    Matrix m_MeasurementNoise;

    bool m_IsInitialized = false;

    // Construction
    ExtendedKalmanFilter(std::size_t state_dim, std::size_t measurement_dim, std::size_t control_dim);
};

} // namespace Core

#endif // KALMAN_CPP_EXTENDED_KALMAN_FILTER_H
