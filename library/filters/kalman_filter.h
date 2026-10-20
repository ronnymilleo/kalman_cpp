/**
 * @file    kalman_filter.h
 * @brief   Discrete-time Kalman filter for linear systems.
 */

#ifndef KALMAN_CPP_KALMAN_FILTER_H
#define KALMAN_CPP_KALMAN_FILTER_H

#include "core/result.h"
#include "filters/kalman_math.h"
#include "matrix/matrix.h"

#include <cstddef>
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
 *          Operations that can fail return a `Result` with a message. A filter is left unchanged when an
 *          operation fails. The class is not thread safe.
 */
class KalmanFilter {
public:
    // Construction
    [[nodiscard]] static Result<KalmanFilter> Create(std::size_t state_dim, std::size_t measurement_dim,
                                                     std::size_t control_dim = 0);

    // Model configuration
    [[nodiscard]] Result<void> InitializeState(const Vector &initial_state);
    [[nodiscard]] Result<void> SetStateTransitionMatrix(const Matrix &f);
    [[nodiscard]] Result<void> SetControlMatrix(const Matrix &b);
    [[nodiscard]] Result<void> SetObservationMatrix(const Matrix &h);
    [[nodiscard]] Result<void> SetProcessNoiseCovariance(const Matrix &q);
    [[nodiscard]] Result<void> SetMeasurementNoiseCovariance(const Matrix &r);
    [[nodiscard]] Result<void> SetErrorCovariance(const Matrix &p);

    // Angles and other states that do not live on a line
    void SetResidualFunction(const ResidualFunction &residual);
    void SetStateNormalizer(const StateNormalizer &normalizer);

    // Filtering
    [[nodiscard]] Result<void> Predict(const Vector &control = {});
    [[nodiscard]] Result<void> Update(const Vector &measurement);

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

    ResidualFunction m_Residual;
    StateNormalizer m_Normalizer;

    bool m_IsInitialized = false;

    // Construction
    KalmanFilter(std::size_t state_dim, std::size_t measurement_dim, std::size_t control_dim);
};

} // namespace Core

#endif // KALMAN_CPP_KALMAN_FILTER_H
