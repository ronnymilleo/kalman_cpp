/**
 * @file    extended_kalman_filter.cpp
 * @brief   Discrete-time Extended Kalman filter for non-linear systems.
 */

#include "extended_kalman_filter.h"

#include "filters/kalman_math.h"

#include <format>
#include <utility>

namespace Core {

/**
 * @brief   Creates a filter with the given sizes.
 * @details The filter is not ready to predict or update until the four functions and the error covariance have been
 *          set.
 * @param[in] state_dim        Number of states, at least one.
 * @param[in] measurement_dim  Number of measurements, at least one.
 * @param[in] control_dim      Number of control inputs; zero for a system without control.
 * @return  The filter, or a message saying which size is invalid.
 */
std::expected<ExtendedKalmanFilter, std::string> ExtendedKalmanFilter::Create(const std::size_t state_dim,
                                                                              const std::size_t measurement_dim,
                                                                              const std::size_t control_dim) {
    if (state_dim == 0) {
        return std::unexpected("The state dimension cannot be zero");
    }
    if (measurement_dim == 0) {
        return std::unexpected("The measurement dimension cannot be zero");
    }
    return ExtendedKalmanFilter(state_dim, measurement_dim, control_dim);
}

/**
 * @brief   Sets the state estimate.
 * @param[in] initial_state  One value per state.
 * @return  Nothing, or a message when the number of values is wrong.
 */
std::expected<void, std::string> ExtendedKalmanFilter::InitializeState(const Vector &initial_state) {
    if (initial_state.size() != m_StateDim) {
        return std::unexpected(
            std::format("The initial state must have {} elements, not {}", m_StateDim, initial_state.size()));
    }
    m_State = initial_state;
    return {};
}

/**
 * @brief   Sets the state transition function `f(x, u)`.
 * @param[in] function  Function that maps the state and the control input to the next state; it receives an empty
 *                      control vector when the step has no control input.
 * @return  Nothing, or a message when the function is empty.
 */
std::expected<void, std::string>
ExtendedKalmanFilter::SetStateTransitionFunction(const StateTransitionFunction &function) {
    if (!function) {
        return std::unexpected("The state transition function cannot be empty");
    }
    m_StateTransitionFunction = function;
    return {};
}

/**
 * @brief   Sets the observation function `h(x)`.
 * @param[in] function  Function that maps the state to the measurement it should produce.
 * @return  Nothing, or a message when the function is empty.
 */
std::expected<void, std::string> ExtendedKalmanFilter::SetObservationFunction(const ObservationFunction &function) {
    if (!function) {
        return std::unexpected("The observation function cannot be empty");
    }
    m_ObservationFunction = function;
    return {};
}

/**
 * @brief   Sets the Jacobian of the state transition function.
 * @param[in] function  Function that returns `df/dx` as a `state_dim x state_dim` matrix.
 * @return  Nothing, or a message when the function is empty.
 */
std::expected<void, std::string> ExtendedKalmanFilter::SetStateJacobianFunction(const StateJacobianFunction &function) {
    if (!function) {
        return std::unexpected("The state Jacobian function cannot be empty");
    }
    m_StateJacobianFunction = function;
    return {};
}

/**
 * @brief   Sets the Jacobian of the observation function.
 * @param[in] function  Function that returns `dh/dx` as a `measurement_dim x state_dim` matrix.
 * @return  Nothing, or a message when the function is empty.
 */
std::expected<void, std::string>
ExtendedKalmanFilter::SetObservationJacobianFunction(const ObservationJacobianFunction &function) {
    if (!function) {
        return std::unexpected("The observation Jacobian function cannot be empty");
    }
    m_ObservationJacobianFunction = function;
    return {};
}

/**
 * @brief   Sets the process noise covariance `Q`, the uncertainty the model adds at every step.
 * @param[in] q  Matrix of size `state_dim x state_dim`.
 * @return  Nothing, or a message when the shape is wrong.
 */
std::expected<void, std::string> ExtendedKalmanFilter::SetProcessNoiseCovariance(const Matrix &q) {
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
std::expected<void, std::string> ExtendedKalmanFilter::SetMeasurementNoiseCovariance(const Matrix &r) {
    if (const auto shape = CheckShape("The measurement noise covariance", r, m_MeasurementDim, m_MeasurementDim);
        !shape) {
        return shape;
    }
    m_MeasurementNoise = r;
    return {};
}

/**
 * @brief   Sets the error covariance `P`, the uncertainty of the state estimate.
 * @param[in] p  Matrix of size `state_dim x state_dim`.
 * @return  Nothing, or a message when the shape is wrong.
 */
std::expected<void, std::string> ExtendedKalmanFilter::SetErrorCovariance(const Matrix &p) {
    if (const auto shape = CheckShape("The error covariance", p, m_StateDim, m_StateDim); !shape) {
        return shape;
    }
    m_ErrorCovariance = p;
    m_IsInitialized = true;
    return {};
}

/**
 * @brief   Moves the estimate one step forward in time: `x = f(x, u)` and `P = F P F^T + Q`.
 * @details `F` is the Jacobian evaluated at the estimate the step starts from, before `f` moves it.
 * @param[in] control  One value per control input, or empty when there is no control input this step.
 * @return  Nothing, or a message when the filter is not ready, the control input has the wrong size, or a function
 *          returns a result of the wrong size.
 */
std::expected<void, std::string> ExtendedKalmanFilter::Predict(const Vector &control) {
    if (!m_IsInitialized) {
        return std::unexpected("The Extended Kalman filter is not initialized; set the error covariance first");
    }
    if (!AreAllFunctionsSet()) {
        return std::unexpected("The four model functions must be set before predicting");
    }
    if (!control.empty() && control.size() != m_ControlDim) {
        return std::unexpected(
            std::format("The control input must have {} elements, not {}", m_ControlDim, control.size()));
    }

    const Matrix jacobian = m_StateJacobianFunction(m_State);
    if (const auto shape = CheckShape("The state Jacobian", jacobian, m_StateDim, m_StateDim); !shape) {
        return shape;
    }
    Vector next_state = m_StateTransitionFunction(m_State, control);
    if (next_state.size() != m_StateDim) {
        return std::unexpected(std::format("The state transition function must return {} elements, not {}", m_StateDim,
                                           next_state.size()));
    }

    m_ErrorCovariance = PredictCovariance(jacobian, m_ErrorCovariance, m_ProcessNoise);
    m_State = std::move(next_state);
    return {};
}

/**
 * @brief   Corrects the estimate with a measurement.
 * @details Computes the innovation `y = z - h(x)`, then corrects the state and the covariance like the linear
 *          filter does, with `H` the Jacobian of `h` evaluated at the current estimate.
 * @param[in] measurement  One value per measurement.
 * @return  Nothing, or a message when the filter is not ready, the measurement has the wrong size, a function returns
 *          a result of the wrong size, or the innovation covariance cannot be inverted.
 */
std::expected<void, std::string> ExtendedKalmanFilter::Update(const Vector &measurement) {
    if (!m_IsInitialized) {
        return std::unexpected("The Extended Kalman filter is not initialized; set the error covariance first");
    }
    if (!AreAllFunctionsSet()) {
        return std::unexpected("The four model functions must be set before updating");
    }
    if (measurement.size() != m_MeasurementDim) {
        return std::unexpected(
            std::format("The measurement must have {} elements, not {}", m_MeasurementDim, measurement.size()));
    }

    const Vector expected_measurement = m_ObservationFunction(m_State);
    if (expected_measurement.size() != m_MeasurementDim) {
        return std::unexpected(std::format("The observation function must return {} elements, not {}", m_MeasurementDim,
                                           expected_measurement.size()));
    }
    const Matrix jacobian = m_ObservationJacobianFunction(m_State);
    if (const auto shape = CheckShape("The observation Jacobian", jacobian, m_MeasurementDim, m_StateDim); !shape) {
        return shape;
    }

    Vector innovation(m_MeasurementDim);
    for (std::size_t index = 0; index < m_MeasurementDim; index++) {
        innovation[index] = measurement[index] - expected_measurement[index];
    }

    std::expected<Estimate, std::string> corrected =
        CorrectEstimate({.State = m_State, .Covariance = m_ErrorCovariance}, jacobian, m_MeasurementNoise, innovation);
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
const Vector &ExtendedKalmanFilter::GetState() const {
    return m_State;
}

/**
 * @brief   Gets the uncertainty of the state estimate.
 * @return  The error covariance `P`.
 */
const Matrix &ExtendedKalmanFilter::GetErrorCovariance() const {
    return m_ErrorCovariance;
}

/**
 * @brief   Gets one element of the state estimate.
 * @param[in] index  Position in the state vector, from zero.
 * @return  The element, or nothing when the index lies past the end.
 */
std::optional<double> ExtendedKalmanFilter::GetStateElement(const std::size_t index) const {
    if (index >= m_StateDim) {
        return std::nullopt;
    }
    return m_State[index];
}

/**
 * @brief   Tells whether the error covariance has been set.
 * @return  True once the error covariance has been set; false after a reset. Predicting and updating also need the
 *          four functions, see AreAllFunctionsSet().
 */
bool ExtendedKalmanFilter::IsInitialized() const {
    return m_IsInitialized;
}

/**
 * @brief   Tells whether the model is complete.
 * @return  True when the state transition and observation functions and their Jacobians have all been set.
 */
bool ExtendedKalmanFilter::AreAllFunctionsSet() const {
    return m_StateTransitionFunction && m_ObservationFunction && m_StateJacobianFunction &&
           m_ObservationJacobianFunction;
}

/**
 * @brief   Zeroes the state and the error covariance and marks the filter as not ready.
 * @details The sizes, the functions and the noise covariances are kept, so the filter can be started again with new
 *          initial conditions.
 */
void ExtendedKalmanFilter::Reset() {
    m_State.assign(m_StateDim, 0.0);
    m_ErrorCovariance = Matrix(m_StateDim, m_StateDim);
    m_IsInitialized = false;
}

ExtendedKalmanFilter::ExtendedKalmanFilter(const std::size_t state_dim, const std::size_t measurement_dim,
                                           const std::size_t control_dim)
    : m_StateDim(state_dim), m_MeasurementDim(measurement_dim), m_ControlDim(control_dim), m_State(state_dim, 0.0),
      m_ErrorCovariance(state_dim, state_dim), m_ProcessNoise(state_dim, state_dim),
      m_MeasurementNoise(measurement_dim, measurement_dim) {
}

} // namespace Core
