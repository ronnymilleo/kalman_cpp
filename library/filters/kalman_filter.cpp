/**
 * @file    kalman_filter.cpp
 * @brief   Discrete-time Kalman filter for linear systems.
 */

#include "kalman_filter.h"

#include "filters/kalman_math.h"

#include <format>
#include <utility>

namespace Core {

/**
 * @brief   Creates a filter with the given sizes.
 * @details The state transition matrix starts as the identity and every other matrix as zeros. The filter is not
 *          ready to predict or update until SetErrorCovariance() has been called.
 * @param[in] state_dim        Number of states, at least one.
 * @param[in] measurement_dim  Number of measurements, at least one.
 * @param[in] control_dim      Number of control inputs; zero for a system without control.
 * @return  The filter, or a message saying which size is invalid.
 */
std::expected<KalmanFilter, std::string>
KalmanFilter::Create(const std::size_t state_dim, const std::size_t measurement_dim, const std::size_t control_dim) {
    if (state_dim == 0) {
        return std::unexpected("The state dimension cannot be zero");
    }
    if (measurement_dim == 0) {
        return std::unexpected("The measurement dimension cannot be zero");
    }
    return KalmanFilter(state_dim, measurement_dim, control_dim);
}

/**
 * @brief   Sets the state estimate.
 * @param[in] initial_state  One value per state.
 * @return  Nothing, or a message when the number of values is wrong.
 */
std::expected<void, std::string> KalmanFilter::InitializeState(const Vector &initial_state) {
    if (initial_state.size() != m_StateDim) {
        return std::unexpected(
            std::format("The initial state must have {} elements, not {}", m_StateDim, initial_state.size()));
    }
    m_State = initial_state;
    return {};
}

/**
 * @brief   Sets the state transition matrix `F`, which says how the state evolves from one step to the next.
 * @param[in] f  Matrix of size `state_dim x state_dim`.
 * @return  Nothing, or a message when the shape is wrong.
 */
std::expected<void, std::string> KalmanFilter::SetStateTransitionMatrix(const Matrix &f) {
    if (const auto shape = CheckShape("The state transition matrix", f, m_StateDim, m_StateDim); !shape) {
        return shape;
    }
    m_StateTransition = f;
    return {};
}

/**
 * @brief   Sets the control matrix `B`, which says how the control input moves the state.
 * @param[in] b  Matrix of size `state_dim x control_dim`; the filter must have been created with control inputs.
 * @return  Nothing, or a message when the filter has no control inputs or the shape is wrong.
 */
std::expected<void, std::string> KalmanFilter::SetControlMatrix(const Matrix &b) {
    if (m_ControlDim == 0) {
        return std::unexpected("The filter has no control inputs, so it has no control matrix");
    }
    if (const auto shape = CheckShape("The control matrix", b, m_StateDim, m_ControlDim); !shape) {
        return shape;
    }
    m_ControlMatrix = b;
    return {};
}

/**
 * @brief   Sets the observation matrix `H`, which says which part of the state the sensors measure.
 * @param[in] h  Matrix of size `measurement_dim x state_dim`.
 * @return  Nothing, or a message when the shape is wrong.
 */
std::expected<void, std::string> KalmanFilter::SetObservationMatrix(const Matrix &h) {
    if (const auto shape = CheckShape("The observation matrix", h, m_MeasurementDim, m_StateDim); !shape) {
        return shape;
    }
    m_ObservationMatrix = h;
    return {};
}

/**
 * @brief   Sets the process noise covariance `Q`, the uncertainty the model adds at every step.
 * @param[in] q  Matrix of size `state_dim x state_dim`.
 * @return  Nothing, or a message when the shape is wrong.
 */
std::expected<void, std::string> KalmanFilter::SetProcessNoiseCovariance(const Matrix &q) {
    if (const auto shape = CheckShape("The process noise covariance", q, m_StateDim, m_StateDim); !shape) {
        return shape;
    }
    m_ProcessNoise = q;
    return {};
}

/**
 * @brief   Sets the measurement noise covariance `R`, the uncertainty of the sensors.
 * @param[in] r  Matrix of size `measurement_dim x measurement_dim`.
 * @return  Nothing, or a message when the shape is wrong.
 */
std::expected<void, std::string> KalmanFilter::SetMeasurementNoiseCovariance(const Matrix &r) {
    if (const auto shape = CheckShape("The measurement noise covariance", r, m_MeasurementDim, m_MeasurementDim);
        !shape) {
        return shape;
    }
    m_MeasurementNoise = r;
    return {};
}

/**
 * @brief   Sets the error covariance `P`, the uncertainty of the state estimate, which makes the filter ready.
 * @param[in] p  Matrix of size `state_dim x state_dim`.
 * @return  Nothing, or a message when the shape is wrong.
 */
std::expected<void, std::string> KalmanFilter::SetErrorCovariance(const Matrix &p) {
    if (const auto shape = CheckShape("The error covariance", p, m_StateDim, m_StateDim); !shape) {
        return shape;
    }
    m_ErrorCovariance = p;
    m_IsInitialized = true;
    return {};
}

/**
 * @brief   Moves the estimate one step forward in time: `x = F x + B u` and `P = F P F^T + Q`.
 * @param[in] control  One value per control input, or empty when there is no control input this step.
 * @return  Nothing, or a message when the filter is not ready or the control input has the wrong size.
 */
std::expected<void, std::string> KalmanFilter::Predict(const Vector &control) {
    if (!m_IsInitialized) {
        return std::unexpected("The Kalman filter is not initialized; set the error covariance first");
    }
    if (!control.empty() && control.size() != m_ControlDim) {
        return std::unexpected(
            std::format("The control input must have {} elements, not {}", m_ControlDim, control.size()));
    }

    Vector next_state = m_StateTransition * m_State;
    if (!control.empty()) {
        const Vector control_effect = m_ControlMatrix * control;
        for (std::size_t index = 0; index < m_StateDim; index++) {
            next_state[index] += control_effect[index];
        }
    }

    m_ErrorCovariance = PredictCovariance(m_StateTransition, m_ErrorCovariance, m_ProcessNoise);
    m_State = std::move(next_state);
    return {};
}

/**
 * @brief   Corrects the estimate with a measurement.
 * @details Computes the innovation `y = z - H x`, its covariance `S = H P H^T + R`, the gain `K = P H^T S^-1`, and
 *          then `x = x + K y` and `P = (I - K H) P`.
 * @param[in] measurement  One value per measurement.
 * @return  Nothing, or a message when the filter is not ready, the measurement has the wrong size or the innovation
 *          covariance cannot be inverted.
 */
std::expected<void, std::string> KalmanFilter::Update(const Vector &measurement) {
    if (!m_IsInitialized) {
        return std::unexpected("The Kalman filter is not initialized; set the error covariance first");
    }
    if (measurement.size() != m_MeasurementDim) {
        return std::unexpected(
            std::format("The measurement must have {} elements, not {}", m_MeasurementDim, measurement.size()));
    }

    const Vector expected_measurement = m_ObservationMatrix * m_State;
    Vector innovation(m_MeasurementDim);
    for (std::size_t index = 0; index < m_MeasurementDim; index++) {
        innovation[index] = measurement[index] - expected_measurement[index];
    }

    std::expected<Estimate, std::string> corrected = CorrectEstimate(
        {.State = m_State, .Covariance = m_ErrorCovariance}, m_ObservationMatrix, m_MeasurementNoise, innovation);
    if (!corrected) {
        return std::unexpected(corrected.error());
    }

    m_State = std::move(corrected->State);
    m_ErrorCovariance = std::move(corrected->Covariance);
    return {};
}

/**
 * @brief   Gets the state estimate.
 * @return  One value per state.
 */
const Vector &KalmanFilter::GetState() const {
    return m_State;
}

/**
 * @brief   Gets the uncertainty of the state estimate.
 * @return  The error covariance `P`.
 */
const Matrix &KalmanFilter::GetErrorCovariance() const {
    return m_ErrorCovariance;
}

/**
 * @brief   Gets one element of the state estimate.
 * @param[in] index  Position in the state vector, from zero.
 * @return  The element, or nothing when the index lies past the end.
 */
std::optional<double> KalmanFilter::GetStateElement(const std::size_t index) const {
    if (index >= m_StateDim) {
        return std::nullopt;
    }
    return m_State[index];
}

/**
 * @brief   Tells whether the filter is ready to predict and update.
 * @return  True once the error covariance has been set; false after a reset.
 */
bool KalmanFilter::IsInitialized() const {
    return m_IsInitialized;
}

/**
 * @brief   Zeroes the state and the error covariance and marks the filter as not ready.
 * @details The sizes and the model matrices are kept, so the filter can be started again with new initial
 *          conditions.
 */
void KalmanFilter::Reset() {
    m_State.assign(m_StateDim, 0.0);
    m_ErrorCovariance = Matrix(m_StateDim, m_StateDim);
    m_IsInitialized = false;
}

KalmanFilter::KalmanFilter(const std::size_t state_dim, const std::size_t measurement_dim,
                           const std::size_t control_dim)
    : m_StateDim(state_dim), m_MeasurementDim(measurement_dim), m_ControlDim(control_dim), m_State(state_dim, 0.0),
      m_ErrorCovariance(state_dim, state_dim), m_StateTransition(Matrix::Identity(state_dim)),
      m_ControlMatrix(state_dim, control_dim), m_ProcessNoise(state_dim, state_dim),
      m_ObservationMatrix(measurement_dim, state_dim), m_MeasurementNoise(measurement_dim, measurement_dim) {
}

} // namespace Core
