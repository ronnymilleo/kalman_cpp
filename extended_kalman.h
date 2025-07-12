/**
 * @file extended_kalman.h
 * @brief Extended Kalman filter class definition and interface
 * @author ronnymilleo
 * @date 12/07/25
 * @version 1.0
 * 
 * This header defines the Extended Kalman filter class for real-time state 
 * estimation of non-linear systems. The implementation provides a complete 
 * discrete-time Extended Kalman filter suitable for non-linear systems with 
 * Gaussian noise. The filter supports optional control inputs and is designed 
 * for efficient computation with dynamic sizing.
 * 
 * @details Key features:
 * - Dynamic matrix sizing based on state and measurement dimensions
 * - Support for non-linear state transition and observation functions
 * - Automatic Jacobian computation through user-provided functions
 * - Support for control inputs (optional)
 * - Comprehensive error checking and validation
 * - Modern C++ implementation with STL containers and function objects
 * - Exception-safe design with RAII principles
 * 
 * @note This is a C++ implementation using std::vector for matrix storage
 * @note All matrices are stored in row-major order
 * @note Filter requires proper initialization before use
 * @note Non-linear functions are provided via std::function objects
 * 
 * @see matrix_math.h for matrix operation implementations
 * @see kalman.h for standard linear Kalman filter
 */

#ifndef EXTENDED_KALMAN_H
#define EXTENDED_KALMAN_H

#include <vector>
#include <functional>

/**
 * @class ExtendedKalman
 * @brief Discrete-time Extended Kalman filter implementation
 * 
 * This class implements an Extended Kalman filter for non-linear dynamic
 * systems with Gaussian noise. The filter estimates the state of a system
 * based on noisy measurements and a mathematical model of the non-linear
 * system dynamics.
 * 
 * @details Mathematical model:
 * - State equation: x(k+1) = f(x(k), u(k)) + w(k)
 * - Measurement equation: z(k) = h(x(k)) + v(k)
 * - Process noise w(k) ~ N(0,Q)
 * - Measurement noise v(k) ~ N(0,R)
 * - Linearization via Jacobian matrices F = ∂f/∂x and H = ∂h/∂x
 * 
 * @note All matrices are stored as std::vector<std::vector<double>>
 * @note Filter must be properly initialized before use
 * @note Thread safety is not guaranteed - use external synchronization if needed
 * @note Non-linear functions must be provided by the user
 * 
 * @see matrix_math.h for underlying matrix operations
 * 
 * @example
 * @code
 * // Create a 2D state, 1D measurement EKF
 * ExtendedKalman ekf(2, 1, 0);
 * 
 * // Define non-linear functions
 * auto stateFunc = [](const std::vector<double>& x, const std::vector<double>& u) {
 *     // Non-linear state transition
 *     return std::vector<double>{x[0] + x[1], x[1]};
 * };
 * 
 * auto observationFunc = [](const std::vector<double>& x) {
 *     // Non-linear measurement function
 *     return std::vector<double>{x[0] * x[0]};
 * };
 * 
 * // Set functions and run filter
 * ekf.setStateTransitionFunction(stateFunc);
 * ekf.setObservationFunction(observationFunc);
 * // ... set Jacobian functions and matrices ...
 * 
 * // Filter cycle
 * ekf.predict();
 * ekf.update(measurement);
 * @endcode
 */
class ExtendedKalman {
public:
    /** @name Function Type Definitions */
    /** @{ */
    
    /**
     * @brief Non-linear state transition function type f(x, u)
     * @param state Current state vector
     * @param control Control input vector (can be empty)
     * @return Predicted next state vector
     */
    using StateTransitionFunction = std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)>;
    
    /**
     * @brief Non-linear observation function type h(x)
     * @param state Current state vector
     * @return Predicted measurement vector
     */
    using ObservationFunction = std::function<std::vector<double>(const std::vector<double>&)>;
    
    /**
     * @brief State Jacobian function type ∂f/∂x
     * @param state Current state vector
     * @return Jacobian matrix of state transition function
     */
    using StateJacobianFunction = std::function<std::vector<std::vector<double>>(const std::vector<double>&)>;
    
    /**
     * @brief Observation Jacobian function type ∂h/∂x
     * @param state Current state vector
     * @return Jacobian matrix of observation function
     */
    using ObservationJacobianFunction = std::function<std::vector<std::vector<double>>(const std::vector<double>&)>;
    
    /** @} */

private:
    /** @brief Current state vector (x) - estimated system state */
    std::vector<double> state;

    /** @brief Error covariance matrix (P) - uncertainty in state estimate */
    std::vector<std::vector<double>> errorCovariance;

    /** @brief Process noise covariance matrix (Q) - model uncertainty */
    std::vector<std::vector<double>> processNoise;

    /** @brief Measurement noise covariance matrix (R) - sensor uncertainty */
    std::vector<std::vector<double>> measurementNoise;

    /** @brief Identity matrix for calculations - cached for efficiency */
    std::vector<std::vector<double>> identity;

    /** @brief Number of state variables */
    size_t stateDim{};
    /** @brief Number of measurement variables */
    size_t measurementDim{};
    /** @brief Number of control variables */
    size_t controlDim{};

    /** @brief Non-linear state transition function f(x, u) */
    StateTransitionFunction stateTransitionFunc;
    
    /** @brief Non-linear observation function h(x) */
    ObservationFunction observationFunc;
    
    /** @brief State Jacobian function ∂f/∂x */
    StateJacobianFunction stateJacobianFunc;
    
    /** @brief Observation Jacobian function ∂h/∂x */
    ObservationJacobianFunction observationJacobianFunc;

    /** @brief Filter initialization status */
    bool initialized{false};

public:
    /** @name Constructors and Destructor */
    /** @{ */
    
    /**
     * @brief Default constructor
     * @note Creates an uninitialized filter - must call constructor with dimensions
     */
    ExtendedKalman() = default;
    
    /**
     * @brief Parameterized constructor
     * @param stateDimension Number of state variables (n)
     * @param measurementDimension Number of measurement variables (m)
     * @param controlDimension Number of control variables (p), default is 0
     * 
     * Creates an Extended Kalman filter with specified dimensions and initializes
     * all matrices to appropriate sizes filled with zeros.
     * 
     * @throws std::invalid_argument if any dimension is less than 1
     * 
     * @note Filter requires further initialization via setter methods
     * @note Identity matrix is pre-computed for efficiency
     * @note Non-linear functions must be set before filter operation
     */
    ExtendedKalman(size_t stateDimension, size_t measurementDimension, size_t controlDimension = 0);
    
    /**
     * @brief Destructor
     * @note Default destructor - automatic cleanup via RAII
     */
    ~ExtendedKalman() = default;
    
    /** @} */

    /** @name Initialization Methods */
    /** @{ */
    
    /**
     * @brief Initialize the state vector
     * @param initialState Initial state estimate
     * 
     * Sets the initial state vector and marks the filter as initialized.
     * This must be called before running predict/update cycles.
     * 
     * @throws std::invalid_argument if initialState size doesn't match stateDim
     * 
     * @example
     * @code
     * std::vector<double> init = {0.0, 0.0}; // position=0, velocity=0
     * ekf.initializeState(init);
     * @endcode
     */
    void initializeState(const std::vector<double>& initialState);
    
    /**
     * @brief Set the non-linear state transition function f(x, u)
     * @param func State transition function
     * 
     * Defines how the state evolves from one time step to the next using
     * a non-linear function. This function should implement the system dynamics.
     * 
     * @throws std::invalid_argument if func is nullptr (empty function)
     * 
     * @note This function defines the non-linear system dynamics
     * @note The function should handle control inputs appropriately
     * @note For systems without control, the control vector will be empty
     */
    void setStateTransitionFunction(const StateTransitionFunction& func);
    
    /**
     * @brief Set the non-linear observation function h(x)
     * @param func Observation function
     * 
     * Maps the state space to the measurement space using a non-linear function.
     * This function should implement how measurements relate to the state.
     * 
     * @throws std::invalid_argument if func is nullptr (empty function)
     * 
     * @example
     * @code
     * // Non-linear observation: range measurement from 2D position
     * auto obsFunc = [](const std::vector<double>& x) {
     *     return std::vector<double>{sqrt(x[0]*x[0] + x[1]*x[1])};
     * };
     * ekf.setObservationFunction(obsFunc);
     * @endcode
     */
    void setObservationFunction(const ObservationFunction& func);
    
    /**
     * @brief Set the state Jacobian function ∂f/∂x
     * @param func State Jacobian function
     * 
     * Computes the Jacobian matrix of the state transition function with
     * respect to the state. This is used for linearization in the prediction step.
     * 
     * @throws std::invalid_argument if func is nullptr (empty function)
     * 
     * @note The Jacobian should be evaluated at the current state
     * @note Matrix dimensions should be (stateDim x stateDim)
     * @note Partial derivatives should be computed analytically for accuracy
     */
    void setStateJacobianFunction(const StateJacobianFunction& func);
    
    /**
     * @brief Set the observation Jacobian function ∂h/∂x
     * @param func Observation Jacobian function
     * 
     * Computes the Jacobian matrix of the observation function with
     * respect to the state. This is used for linearization in the update step.
     * 
     * @throws std::invalid_argument if func is nullptr (empty function)
     * 
     * @note The Jacobian should be evaluated at the predicted state
     * @note Matrix dimensions should be (measurementDim x stateDim)
     * @note Partial derivatives should be computed analytically for accuracy
     */
    void setObservationJacobianFunction(const ObservationJacobianFunction& func);
    
    /**
     * @brief Set the process noise covariance matrix (Q)
     * @param Q Process noise covariance (stateDim x stateDim)
     * 
     * Represents uncertainty in the system model. Higher values indicate
     * less trust in the model predictions.
     * 
     * @throws std::invalid_argument if Q dimensions don't match (stateDim x stateDim)
     * 
     * @note Should be positive semi-definite
     * @note Diagonal elements represent individual state variable uncertainties
     */
    void setProcessNoiseCovariance(const std::vector<std::vector<double>>& Q);
    
    /**
     * @brief Set the measurement noise covariance matrix (R)
     * @param R Measurement noise covariance (measurementDim x measurementDim)
     * 
     * Represents uncertainty in the measurements. Higher values indicate
     * less trust in the sensor readings.
     * 
     * @throws std::invalid_argument if R dimensions don't match expected size
     * 
     * @note Should be positive definite
     * @note Diagonal elements represent individual sensor uncertainties
     */
    void setMeasurementNoiseCovariance(const std::vector<std::vector<double>>& R);
    
    /**
     * @brief Set the error covariance matrix (P)
     * @param P Initial error covariance (stateDim x stateDim)
     * 
     * Represents initial uncertainty in the state estimate. Higher values
     * indicate less confidence in the initial state.
     * 
     * @throws std::invalid_argument if P dimensions don't match (stateDim x stateDim)
     * 
     * @note Should be positive semi-definite
     * @note Large initial values allow faster convergence to true state
     */
    void setErrorCovariance(const std::vector<std::vector<double>>& P);

    /** @} */

    /** @name Core Extended Kalman Filter Operations */
    /** @{ */
    
    /**
     * @brief Prediction step of the Extended Kalman filter
     * @param control Control input vector (optional)
     * 
     * Predicts the next state and error covariance based on the non-linear system model.
     * This implements the EKF prediction equations:
     * - x_pred = f(x, u)
     * - F = ∂f/∂x (Jacobian)
     * - P_pred = F * P * F^T + Q
     * 
     * @throws std::runtime_error if filter is not initialized
     * @throws std::runtime_error if required functions are not set
     * @throws std::invalid_argument if control size doesn't match controlDim
     * 
     * @note Call this before update() in each filter cycle
     * @note Control input is optional if controlDim = 0
     * @note State transition and Jacobian functions must be set
     */
    void predict(const std::vector<double>& control = {});
    
    /**
     * @brief Update step of the Extended Kalman filter
     * @param measurement Measurement vector
     * 
     * Updates the state estimate based on the measurement using the non-linear
     * observation model. This implements the EKF correction equations:
     * - z_pred = h(x)
     * - H = ∂h/∂x (Jacobian)
     * - K = P * H^T * (H * P * H^T + R)^(-1)
     * - x = x + K * (z - z_pred)
     * - P = (I - K * H) * P
     * 
     * @throws std::runtime_error if filter is not initialized
     * @throws std::runtime_error if required functions are not set
     * @throws std::invalid_argument if measurement size doesn't match measurementDim
     * 
     * @note Call this after predict() in each filter cycle
     * @note Measurement vector must match the configured measurement dimension
     * @note Observation and Jacobian functions must be set
     */
    void update(const std::vector<double>& measurement);

    /** @} */

    /** @name Getters and Access Methods */
    /** @{ */
    
    /**
     * @brief Get the current state vector
     * @return Reference to the current state estimate
     * 
     * @throws std::runtime_error if filter is not initialized
     * 
     * @note Returns a reference for efficiency - do not modify directly
     */
    std::vector<double>& getState();
    
    /**
     * @brief Get the current error covariance matrix
     * @return Reference to the current error covariance estimate
     * 
     * @throws std::runtime_error if filter is not initialized
     * 
     * @note Returns a reference for efficiency - do not modify directly
     * @note Diagonal elements represent uncertainty in each state variable
     */
    std::vector<std::vector<double>>& getErrorCovariance();
    
    /**
     * @brief Get a specific element from the state vector
     * @param index Index of the state element (0-based)
     * @return Value of the specified state element
     * 
     * @throws std::runtime_error if filter is not initialized
     * @throws std::out_of_range if index is invalid
     * 
     * @note More efficient than accessing getState()[index]
     */
    double getStateElement(size_t index) const;

    /** @} */

    /** @name Utility Methods */
    /** @{ */
    
    /**
     * @brief Reset the filter to uninitialized state
     * 
     * Clears the state vector and marks the filter as uninitialized.
     * Matrix dimensions and values are preserved, but functions are cleared.
     * 
     * @note Useful for restarting the filter with different initial conditions
     * @note Requires calling initializeState() and setting functions again before use
     */
    void reset();
    
    /**
     * @brief Check if the filter is properly initialized
     * @return true if filter is initialized and ready for predict/update cycles
     * 
     * @note Returns false if initializeState() has not been called
     * @note Returns false if required functions are not set
     */
    bool isInitialized() const;
    
    /**
     * @brief Check if all required functions are set
     * @return true if all non-linear functions and Jacobian functions are set
     * 
     * @note Required for EKF operation
     * @note All four function types must be provided
     */
    bool areAllFunctionsSet() const;

    /** @} */
};

#endif //EXTENDED_KALMAN_H
