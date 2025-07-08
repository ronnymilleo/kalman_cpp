//
// Created by Ronny Milleo on 08/07/25.
//

#include <gtest/gtest.h>
#include "matrix_math.h"

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
