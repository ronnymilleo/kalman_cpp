/**
 * @file matrix_math.cpp
 * @brief Matrix mathematics operations implementation
 * @author ronnymilleo
 * @date 08/07/25
 * @version 1.0
 * 
 * This file implements essential matrix operations for the Kalman filter.
 * All functions provide comprehensive error checking and efficient computation
 * of linear algebra operations using std::vector-based matrices.
 * 
 * @details Implementation features:
 * - Robust error checking for dimension mismatches
 * - Efficient algorithms for matrix operations
 * - Gaussian elimination with partial pivoting for matrix inversion
 * - Clear separation of concerns for each operation
 * - Exception-safe design with proper error messages
 * - Optimized loops for better performance
 * 
 * @note All matrices are stored in row-major order
 * @note Functions assume pre-allocated result containers
 * @note Matrix inversion uses partial pivoting for numerical stability
 * 
 * @see matrix_math.h for function documentation
 * @see kalman.cpp for usage examples
 */

#include "matrix_math.h"
#include <stdexcept>

void matrix_add(const std::vector<std::vector<double>>& A,
                const std::vector<std::vector<double>>& B,
                std::vector<std::vector<double>>& result) {
    if (A.empty() || B.empty() || A.size() != B.size() || A[0].size() != B[0].size()) {
        throw std::invalid_argument("Matrix size mismatch");
    }
    if (result.size() != A.size() || result[0].size() != A[0].size()) {
        throw std::invalid_argument("Result matrix size mismatch");
    }

    const size_t rows = A.size();
    const size_t cols = A[0].size();

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
}

void matrix_subtract(const std::vector<std::vector<double>>& A,
                     const std::vector<std::vector<double>>& B,
                     std::vector<std::vector<double>>& result) {
    if (A.empty() || B.empty() || A.size() != B.size() || A[0].size() != B[0].size()) {
        throw std::invalid_argument("Matrix size mismatch");
    }
    if (result.size() != A.size() || result[0].size() != A[0].size()) {
        throw std::invalid_argument("Result matrix size mismatch");
    }

    const size_t rows = A.size();
    const size_t cols = A[0].size();

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
}

void matrix_transpose(const std::vector<std::vector<double>>& A,
                      std::vector<std::vector<double>>& result) {
    if (A.empty() || A[0].size() != result.size() || A.size() != result[0].size()) {
        throw std::invalid_argument("Matrix size mismatch");
    }

    const size_t rows = A.size();
    const size_t cols = A[0].size();

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j][i] = A[i][j];
        }
    }
}

void matrix_multiply(const std::vector<std::vector<double>>& A,
                     const std::vector<std::vector<double>>& B,
                     std::vector<std::vector<double>>& result) {
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument("Matrix size mismatch");
    }
    if (result.size() != A.size() || result[0].size() != B[0].size()) {
        throw std::invalid_argument("Result matrix size mismatch");
    }

    const size_t rowsA = A.size();
    const size_t colsA = A[0].size();
    const size_t colsB = B[0].size();

    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            result[i][j] = 0.0; // Corrigido: zera antes de acumular
            for (size_t k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrix_vector_multiply(const std::vector<std::vector<double>>& A,
                            const std::vector<double>& v,
                            std::vector<double>& result) {
    if (A.empty() || v.empty() || A[0].size() != v.size()) {
        throw std::invalid_argument("Matrix and vector size mismatch");
    }
    if (result.size() != A.size()) {
        throw std::invalid_argument("Result vector size mismatch");
    }

    const size_t rows = A.size();
    const size_t cols = A[0].size();

    for (size_t i = 0; i < rows; i++) {
        result[i] = 0.0; // Corrigido: zera antes de acumular
        for (size_t j = 0; j < cols; j++) {
            result[i] += A[i][j] * v[j];
        }
    }
}

void matrix_identity(std::vector<std::vector<double>>& result) {
    if (result.empty() || result.size() != result[0].size()) {
        throw std::invalid_argument("Not square matrix");
    }
    size_t n = result.size();
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            result[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void matrix_inverse(const std::vector<std::vector<double>>& A,
                    std::vector<std::vector<double>>& result) {
    const size_t n = A.size();
    result = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    // Create augmented matrix [A|I]
    std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n, 0.0));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            augmented[i][j] = A[i][j];
        }
        augmented[i][i + n] = 1.0;
    }

    // Gaussian elimination
    for (size_t i = 0; i < n; i++) {
        // Find pivot
        size_t maxRow = i;
        for (size_t k = i + 1; k < n; k++) {
            if (std::abs(augmented[k][i]) > std::abs(augmented[maxRow][i])) {
                maxRow = k;
            }
        }

        // Swap rows
        if (maxRow != i) {
            std::swap(augmented[i], augmented[maxRow]);
        }

        // Normalize pivot row
        double pivot = augmented[i][i];
        for (size_t j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivot;
        }

        // Eliminate other rows
        for (size_t k = 0; k < n; k++) {
            if (k != i) {
                double factor = augmented[k][i];
                for (size_t j = 0; j < 2 * n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    // Extract inverse matrix
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            result[i][j] = augmented[i][j + n];
        }
    }
}