//
// Created by Ronny Milleo on 08/07/25.
//

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