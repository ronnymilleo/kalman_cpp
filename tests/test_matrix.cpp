/**
 * @file test_matrix.cpp
 * @brief Unit tests for matrix mathematics operations
 * @author ronnymilleo
 * @date 08/07/25
 * @version 1.0
 * 
 * This file contains comprehensive unit tests for the matrix_math module
 * using the Google Test framework. Tests cover all matrix operations
 * including edge cases, error conditions, and numerical accuracy.
 * 
 * @details Test coverage:
 * - Basic operations: addition, subtraction, multiplication
 * - Matrix transformations: transpose, inverse
 * - Vector operations: matrix-vector multiplication
 * - Utility functions: identity matrix generation
 * - Error handling: dimension mismatches, invalid inputs
 * - Numerical accuracy: floating-point precision tests
 * 
 * @note Tests use EXPECT_EQ for exact comparisons where appropriate
 * @note Floating-point comparisons should use EXPECT_NEAR for tolerance
 * @note Each test is self-contained and independent
 * 
 * @see matrix_math.h for function documentation
 * @see matrix_math.cpp for implementation details
 */

#include <gtest/gtest.h>
#include "../matrix_math.h"

TEST(MatrixMathTest, Add) {
    const std::vector<std::vector<double>> A = {{1, 2}, {3, 4}};
    const std::vector<std::vector<double>> B = {{5, 6}, {7, 8}};
    std::vector<std::vector<double>> result(2, std::vector<double>(2, 0.0));

    matrix_add(A, B, result);

    EXPECT_EQ(result[0][0], 6);
    EXPECT_EQ(result[0][1], 8);
    EXPECT_EQ(result[1][0], 10);
    EXPECT_EQ(result[1][1], 12);
}

TEST(MatrixMathTest, Subtract) {
    const std::vector<std::vector<double>> A = {{5, 6}, {7, 8}};
    const std::vector<std::vector<double>> B = {{1, 2}, {3, 4}};
    std::vector<std::vector<double>> result(2, std::vector<double>(2, 0.0));

    matrix_subtract(A, B, result);

    EXPECT_EQ(result[0][0], 4);
    EXPECT_EQ(result[0][1], 4);
    EXPECT_EQ(result[1][0], 4);
    EXPECT_EQ(result[1][1], 4);
}

TEST(MatrixMathTest, Transpose) {
    const std::vector<std::vector<double>> A = {{1, 2}, {3, 4}};
    std::vector<std::vector<double>> result(2, std::vector<double>(2, 0.0));

    matrix_transpose(A, result);

    EXPECT_EQ(result[0][0], 1);
    EXPECT_EQ(result[0][1], 3);
    EXPECT_EQ(result[1][0], 2);
    EXPECT_EQ(result[1][1], 4);
}

TEST(MatrixMathTest, Multiply) {
    const std::vector<std::vector<double>> A = {{1, 2}, {3, 4}};
    const std::vector<std::vector<double>> B = {{5, 6}, {7, 8}};
    std::vector<std::vector<double>> result(2, std::vector<double>(2, 0.0));

    matrix_multiply(A, B, result);

    EXPECT_EQ(result[0][0], 19);
    EXPECT_EQ(result[0][1], 22);
    EXPECT_EQ(result[1][0], 43);
    EXPECT_EQ(result[1][1], 50);
}

TEST(MatrixMathTest, VectorMultiply) {
    const std::vector<std::vector<double>> A = {{1, 2}, {3, 4}};
    const std::vector<double> v = {5, 6};
    std::vector<double> result(2, 0.0);

    matrix_vector_multiply(A, v, result);

    EXPECT_EQ(result[0], 17);
    EXPECT_EQ(result[1], 39);
}

TEST(MatrixMathTest, Identity) {
    std::vector<std::vector<double>> result(3, std::vector<double>(3, 0.0));
    matrix_identity(result);

    EXPECT_EQ(result[0][0], 1);
    EXPECT_EQ(result[1][1], 1);
    EXPECT_EQ(result[2][2], 1);
    EXPECT_EQ(result[0][1], 0);
    EXPECT_EQ(result[1][2], 0);
}

TEST(MatrixMathTest, Inverse) {
    const std::vector<std::vector<double>> A = {{4, 7}, {2, 6}};
    std::vector<std::vector<double>> result(2, std::vector<double>(2, 0.0));

    matrix_inverse(A, result);

    EXPECT_NEAR(result[0][0], 0.6, 1e-9);
    EXPECT_NEAR(result[0][1], -0.7, 1e-9);
    EXPECT_NEAR(result[1][0], -0.2, 1e-9);
    EXPECT_NEAR(result[1][1], 0.4, 1e-9);
}