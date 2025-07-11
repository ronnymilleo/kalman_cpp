/**
 * @file kalman.h
 * @brief Kalman filter class definition and interface
 * @author ronnymilleo
 * @date 08/07/25
 * @version 1.0
 * 
 * This header defines the Kalman filter class for real-time state estimation.
 * The implementation provides a complete discrete-time Kalman filter suitable
 * for linear systems with Gaussian noise. The filter supports optional control
 * inputs and is designed for efficient computation with dynamic sizing.
 * 
 * @details Key features:
 * - Dynamic matrix sizing based on state and measurement dimensions
 * - Support for control inputs (optional)
 * - Comprehensive error checking and validation
 * - Standard discrete-time Kalman filter equations
 * - Modern C++ implementation with STL containers
 * - Exception-safe design with RAII principles
 * 
 * @note This is a C++ implementation using std::vector for matrix storage
 * @note All matrices are stored in row-major order
 * @note Filter requires proper initialization before use
 * 
 * @see matrix_math.h for matrix operation implementations
 * @see main.cpp for usage examples
 */

#ifndef KALMAN_H
#define KALMAN_H

#include <vector>

/**
 * @class kalman
 * @brief Discrete-time Kalman filter implementation
 * 
 * This class implements a standard discrete-time Kalman filter for linear
 * dynamic systems with Gaussian noise. The filter estimates the state of
 * a system based on noisy measurements and a mathematical model of the
 * system dynamics.
 * 
 * @details Mathematical model:
 * - State equation: x(k+1) = F*x(k) + B*u(k) + w(k)
 * - Measurement equation: z(k) = H*x(k) + v(k)
 * - Process noise w(k) ~ N(0,Q)
 * - Measurement noise v(k) ~ N(0,R)
 * 
 * @note All matrices are stored as std::vector<std::vector<double>>
 * @note Filter must be properly initialized before use
 * @note Thread safety is not guaranteed - use external synchronization if needed
 * 
 * @see matrix_math.h for underlying matrix operations
 * 
 * @example
 * @code
 * // Create a 2D state, 1D measurement filter
 * kalman kf(2, 1, 0);
 * 
 * // Initialize state
 * std::vector<double> initialState = {0.0, 0.0};
 * kf.initializeState(initialState);
 * 
 * // Set matrices and run filter
 * kf.setStateTransitionMatrix(F);
 * kf.setObservationMatrix(H);
 * // ... set other matrices ...
 * 
 * // Filter cycle
 * kf.predict();
 * kf.update(measurement);
 * @endcode
 */
class kalman {
private:
    /** @brief Current state vector (x) - estimated system state */
    std::vector<double> state;

    /** @brief Error covariance matrix (P) - uncertainty in state estimate */
    std::vector<std::vector<double>> errorCovariance;

    /** @brief State transition matrix (F) - system dynamics model */
    std::vector<std::vector<double>> stateTransition;

    /** @brief Control matrix (B) - maps control inputs to state changes */
    std::vector<std::vector<double>> controlMatrix;

    /** @brief Process noise covariance matrix (Q) - model uncertainty */
    std::vector<std::vector<double>> processNoise;

    /** @brief Observation matrix (H) - maps state to measurements */
    std::vector<std::vector<double>> observationMatrix;

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

public:
    /** @name Constructors and Destructor */
    /** @{ */
    
    /**
     * @brief Default constructor
     * @note Creates an uninitialized filter - must call constructor with dimensions
     */
    kalman() = default;
    
    /**
     * @brief Parameterized constructor
     * @param stateDimension Number of state variables (n)
     * @param measurementDimension Number of measurement variables (m)
     * @param controlDimension Number of control variables (p), default is 0
     * 
     * Creates a Kalman filter with specified dimensions and initializes
     * all matrices to appropriate sizes filled with zeros.
     * 
     * @throws std::invalid_argument if any dimension is less than 1
     * 
     * @note Filter requires further initialization via setter methods
     * @note Identity matrix is pre-computed for efficiency
     */
    kalman(int stateDimension, int measurementDimension, int controlDimension = 0);
    
    /**
     * @brief Destructor
     * @note Default destructor - automatic cleanup via RAII
     */
    ~kalman() = default;
    
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
     * kf.initializeState(init);
     * @endcode
     */
    void initializeState(const std::vector<double>& initialState);
    
    /**
     * @brief Set the state transition matrix (F)
     * @param F State transition matrix (stateDim x stateDim)
     * 
     * Defines how the state evolves from one time step to the next.
     * For a constant velocity model: F = [[1, dt], [0, 1]]
     * 
     * @throws std::invalid_argument if F dimensions don't match (stateDim x stateDim)
     * 
     * @note This matrix encodes the system dynamics model
     * @note For time-varying systems, this should be updated each time step
     */
    void setStateTransitionMatrix(const std::vector<std::vector<double>>& F);
    
    /**
     * @brief Set the control matrix (B)
     * @param B Control matrix (stateDim x controlDim)
     * 
     * Maps control inputs to state changes. Only needed if controlDim > 0.
     * 
     * @throws std::invalid_argument if B dimensions don't match expected size
     * @throws std::logic_error if controlDim is 0
     */
    void setControlMatrix(const std::vector<std::vector<double>>& B);
    /**
     * @brief Set the observation matrix (H)
     * @param H Observation matrix (measurementDim x stateDim)
     * 
     * Maps the state space to the measurement space. Defines which
     * state variables are directly observable.
     * 
     * @throws std::invalid_argument if H dimensions don't match expected size
     * 
     * @example
     * @code
     * // Observe only position from [position, velocity] state
     * std::vector<std::vector<double>> H = {{1.0, 0.0}};
     * kf.setObservationMatrix(H);
     * @endcode
     */
    void setObservationMatrix(const std::vector<std::vector<double>>& H);
    
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

    /** @name Core Kalman Filter Operations */
    /** @{ */
    
    /**
     * @brief Prediction step of the Kalman filter
     * @param control Control input vector (optional)
     * 
     * Predicts the next state and error covariance based on the system model.
     * This implements the prediction equations:
     * - x_pred = F * x + B * u
     * - P_pred = F * P * F^T + Q
     * 
     * @throws std::runtime_error if filter is not initialized
     * @throws std::invalid_argument if control size doesn't match controlDim
     * 
     * @note Call this before update() in each filter cycle
     * @note Control input is optional if controlDim = 0
     */
    void predict(const std::vector<double>& control = {});
    
    /**
     * @brief Update step of the Kalman filter
     * @param measurement Measurement vector
     * 
     * Updates the state estimate based on the measurement. This implements
     * the correction equations:
     * - K = P * H^T * (H * P * H^T + R)^(-1)
     * - x = x + K * (z - H * x)
     * - P = (I - K * H) * P
     * 
     * @throws std::runtime_error if filter is not initialized
     * @throws std::invalid_argument if measurement size doesn't match measurementDim
     * 
     * @note Call this after predict() in each filter cycle
     * @note Measurement vector must match the configured measurement dimension
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
    double getStateElement(int index) const;

    /** @} */

    /** @name Utility Methods */
    /** @{ */
    
    /**
     * @brief Reset the filter to uninitialized state
     * 
     * Clears the state vector and marks the filter as uninitialized.
     * Matrix dimensions and values are preserved.
     * 
     * @note Useful for restarting the filter with different initial conditions
     * @note Requires calling initializeState() again before use
     */
    void reset();
    
    /**
     * @brief Check if the filter is properly initialized
     * @return true if filter is initialized and ready for predict/update cycles
     * 
     * @note Returns false if initializeState() has not been called
     */
    bool isInitialized() const;

    /** @} */

private:
    /** @brief Filter initialization status */
    bool initialized{};
};

#endif //KALMAN_H
