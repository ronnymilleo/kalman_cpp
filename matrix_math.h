/**
 * @file matrix_math.h
 * @brief Matrix mathematics operations for Kalman filter computations
 * @author ronnymilleo
 * @date 08/07/25
 * @version 1.0
 * 
 * This header provides essential matrix operations required for Kalman filter
 * computations. All functions operate on std::vector<std::vector<double>>
 * matrices and provide efficient implementations of standard linear algebra
 * operations with comprehensive error checking.
 * 
 * @details Supported operations:
 * - Basic arithmetic: addition, subtraction, multiplication
 * - Matrix transformations: transpose, inverse
 * - Vector operations: matrix-vector multiplication
 * - Utility functions: identity matrix generation
 * - Robust error handling for dimension mismatches
 * 
 * @note All matrices are stored in row-major order
 * @note Functions assume pre-allocated result matrices with correct dimensions
 * @note Matrix inversion uses Gaussian elimination with partial pivoting
 * @note All operations are performed in-place on the result matrix
 * 
 * @see kalman.h for the main Kalman filter implementation
 * 
 * @warning Matrix inversion may fail for singular matrices - check return values
 * @warning No numerical stability optimizations beyond partial pivoting
 */

#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <vector>

/** @name Basic Matrix Operations */
/** @{ */

/**
 * @brief Matrix addition: result = A + B
 * @param A First input matrix (m x n)
 * @param B Second input matrix (m x n)
 * @param result Output matrix (m x n) - must be pre-allocated
 * 
 * Performs element-wise addition of two matrices. All matrices must have
 * identical dimensions.
 * 
 * @throws std::invalid_argument if matrix dimensions don't match
 * @throws std::invalid_argument if result matrix has wrong dimensions
 * 
 * @note Result matrix must be pre-allocated with correct dimensions
 * @note Operation is performed in-place on the result matrix
 * 
 * @example
 * @code
 * std::vector<std::vector<double>> A = {{1, 2}, {3, 4}};
 * std::vector<std::vector<double>> B = {{5, 6}, {7, 8}};
 * std::vector<std::vector<double>> result(2, std::vector<double>(2));
 * matrix_add(A, B, result); // result = {{6, 8}, {10, 12}}
 * @endcode
 */
void matrix_add(const std::vector<std::vector<double>>& A,
                const std::vector<std::vector<double>>& B,
                std::vector<std::vector<double>>& result);

/**
 * @brief Matrix subtraction: result = A - B
 * @param A First input matrix (m x n)
 * @param B Second input matrix (m x n)  
 * @param result Output matrix (m x n) - must be pre-allocated
 * 
 * Performs element-wise subtraction of two matrices. All matrices must have
 * identical dimensions.
 * 
 * @throws std::invalid_argument if matrix dimensions don't match
 * @throws std::invalid_argument if result matrix has wrong dimensions
 * 
 * @note Result matrix must be pre-allocated with correct dimensions
 * @note Operation is performed in-place on the result matrix
 */
void matrix_subtract(const std::vector<std::vector<double>>& A,
                     const std::vector<std::vector<double>>& B,
                     std::vector<std::vector<double>>& result);

/** @} */

/** @name Matrix Transformations */
/** @{ */

/**
 * @brief Matrix transpose: result = A^T
 * @param A Input matrix (m x n)
 * @param result Output matrix (n x m) - must be pre-allocated
 * 
 * Computes the transpose of a matrix, swapping rows and columns.
 * 
 * @throws std::invalid_argument if result dimensions don't match A^T
 * 
 * @note Result matrix must have dimensions (n x m) if A is (m x n)
 * @note Can handle non-square matrices
 * @note Self-transpose is supported if A is square and result == A
 */
void matrix_transpose(const std::vector<std::vector<double>>& A,
                      std::vector<std::vector<double>>& result);

/** @} */

/** @name Matrix Multiplication */
/** @{ */

/**
 * @brief Matrix multiplication: result = A * B
 * @param A First input matrix (m x k)
 * @param B Second input matrix (k x n)
 * @param result Output matrix (m x n) - must be pre-allocated
 * 
 * Performs standard matrix multiplication using the definition:
 * result[i][j] = Σ(A[i][p] * B[p][j]) for p = 0 to k-1
 * 
 * @throws std::invalid_argument if A columns != B rows
 * @throws std::invalid_argument if result dimensions don't match (m x n)
 * 
 * @note Inner dimensions must match: A(m x k) * B(k x n) = result(m x n)
 * @note Result matrix must be pre-allocated with correct dimensions
 * @note Time complexity: O(m * n * k)
 */
void matrix_multiply(const std::vector<std::vector<double>>& A,
                     const std::vector<std::vector<double>>& B,
                     std::vector<std::vector<double>>& result);

/**
 * @brief Matrix-vector multiplication: result = A * v
 * @param A Input matrix (m x n)
 * @param v Input vector (n elements)
 * @param result Output vector (m elements) - must be pre-allocated
 * 
 * Multiplies a matrix with a column vector, treating the vector as an (n x 1) matrix.
 * 
 * @throws std::invalid_argument if A columns != vector size
 * @throws std::invalid_argument if result size != A rows
 * 
 * @note More efficient than general matrix multiplication for vectors
 * @note Result vector must be pre-allocated with correct size
 */
void matrix_vector_multiply(const std::vector<std::vector<double>>& A,
                            const std::vector<double>& v,
                            std::vector<double>& result);

/** @} */

/** @name Utility Functions */
/** @{ */

/**
 * @brief Create identity matrix: result = I
 * @param result Output matrix (n x n) - must be pre-allocated and square
 * 
 * Fills the provided square matrix with identity values (1.0 on diagonal, 0.0 elsewhere).
 * 
 * @throws std::invalid_argument if result matrix is not square
 * 
 * @note Result matrix must be pre-allocated as square matrix
 * @note Overwrites any existing content in result matrix
 */
void matrix_identity(std::vector<std::vector<double>>& result);

/**
 * @brief Matrix inversion: result = A^(-1)
 * @param A Input matrix (n x n) - must be square and non-singular
 * @param result Output matrix (n x n) - must be pre-allocated
 * 
 * Computes the matrix inverse using Gaussian elimination with partial pivoting.
 * The algorithm transforms [A|I] to [I|A^(-1)] through row operations.
 * 
 * @throws std::invalid_argument if A is not square
 * @throws std::invalid_argument if result dimensions don't match A
 * @throws std::runtime_error if A is singular (non-invertible)
 * 
 * @note Uses partial pivoting for numerical stability
 * @note May fail for ill-conditioned matrices
 * @note Result matrix must be pre-allocated with same dimensions as A
 * 
 * @warning Check for exceptions when inverting potentially singular matrices
 * @warning Numerical precision may be limited for very small determinants
 */
void matrix_inverse(const std::vector<std::vector<double>>& A,
                    std::vector<std::vector<double>>& result);

/** @} */

#endif //MATRIX_MATH_H
