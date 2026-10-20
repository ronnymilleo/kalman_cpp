/**
 * @file    kalman_filter.h
 * @brief   Discrete-time Kalman filter for linear systems.
 */

#ifndef KALMAN_CPP_KALMAN_FILTER_H
#define KALMAN_CPP_KALMAN_FILTER_H

#include "matrix/matrix.h"

#include <cstddef>
#include <expected>
#include <optional>
#include <string>

namespace Core {

/**
 * @class   KalmanFilter
 * @brief   Estimates the state of a linear system from noisy measurements.
 * @details The model is `x(k+1) = F x(k) + B u(k) + w(k)` and `z(k) = H x(k) + v(k)`, with process noise
 *          `w ~ N(0, Q)` and measurement noise `v ~ N(0, R)`. A filter is created with the sizes of the state, the
 *          measurement and the optional control input, configured through the setters and ready to predict and update
 *          once the error covariance `P` has been set.
 *
 *          Operations that can fail return `std::expected` with a message. A filter is left unchanged when an
 *          operation fails. The class is not thread safe.
 */
class KalmanFilter {
public:
    // Construction
    [[nodiscard]] static std::expected<KalmanFilter, std::string>
    Create(std::size_t state_dim, std::size_t measurement_dim, std::size_t control_dim = 0);

    // Model configuration
    [[nodiscard]] std::expected<void, std::string> InitializeState(const Vector &initial_state);
    [[nodiscard]] std::expected<void, std::string> SetStateTransitionMatrix(const Matrix &f);
    [[nodiscard]] std::expected<void, std::string> SetControlMatrix(const Matrix &b);
    [[nodiscard]] std::expected<void, std::string> SetObservationMatrix(const Matrix &h);
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
    Matrix m_StateTransition;
    Matrix m_ControlMatrix;
    Matrix m_ProcessNoise;
    Matrix m_ObservationMatrix;
    Matrix m_MeasurementNoise;

    bool m_IsInitialized = false;

    // Construction
    KalmanFilter(std::size_t state_dim, std::size_t measurement_dim, std::size_t control_dim);
};

} // namespace Core

#endif // KALMAN_CPP_KALMAN_FILTER_H
