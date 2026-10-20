/**
 * @file    matrix_tests.cpp
 * @brief   Unit tests for the matrix module.
 */

#include "matrix/matrix.h"

#include "test_support.h"

#include <gtest/gtest.h>

namespace {

constexpr double Tolerance = 1e-12;

} // namespace

TEST(Matrix, ANewMatrixHasTheRequestedShapeAndIsFilledWithZeros) {
    const Core::Matrix matrix(2, 3);

    EXPECT_EQ(matrix.GetRows(), 2);
    EXPECT_EQ(matrix.GetCols(), 3);
    EXPECT_TRUE(Tests::MatrixNear(Core::Matrix({{0, 0, 0}, {0, 0, 0}}), matrix, 0.0));
}

TEST(Matrix, ADefaultMatrixIsEmpty) {
    const Core::Matrix matrix;

    EXPECT_TRUE(matrix.IsEmpty());
    EXPECT_EQ(matrix.GetRows(), 0);
    EXPECT_EQ(matrix.GetCols(), 0);
}

TEST(Matrix, NestedListsFillTheMatrixRowByRow) {
    const Core::Matrix matrix{{1, 2, 3}, {4, 5, 6}};

    EXPECT_EQ(matrix.GetRows(), 2);
    EXPECT_EQ(matrix.GetCols(), 3);
    EXPECT_EQ(matrix(0, 2), 3);
    EXPECT_EQ(matrix(1, 0), 4);
}

TEST(Matrix, ElementsCanBeWritten) {
    Core::Matrix matrix(2, 2);

    matrix(1, 0) = 7.5;

    EXPECT_EQ(matrix(1, 0), 7.5);
    EXPECT_EQ(matrix(0, 0), 0.0);
}

TEST(Matrix, OnlyASquareMatrixIsSquare) {
    EXPECT_TRUE(Core::Matrix(3, 3).IsSquare());
    EXPECT_FALSE(Core::Matrix(2, 3).IsSquare());
}

TEST(Matrix, TwoMatricesHaveTheSameShapeWhenRowsAndColumnsMatch) {
    EXPECT_TRUE(Core::Matrix(2, 3).HasSameShape(Core::Matrix(2, 3)));
    EXPECT_FALSE(Core::Matrix(2, 3).HasSameShape(Core::Matrix(3, 2)));
}

TEST(Matrix, TheIdentityHasOnesOnTheDiagonalOnly) {
    EXPECT_TRUE(Tests::MatrixNear(Core::Matrix({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}), Core::Matrix::Identity(3), 0.0));
}

TEST(Matrix, MatricesWithTheSameElementsAreEqual) {
    EXPECT_EQ(Core::Matrix({{1, 2}, {3, 4}}), Core::Matrix({{1, 2}, {3, 4}}));
    EXPECT_NE(Core::Matrix({{1, 2}, {3, 4}}), Core::Matrix({{1, 2}, {3, 5}}));
}

TEST(Matrix, AddingMatricesAddsEachElement) {
    const Core::Matrix sum = Core::Matrix({{1, 2}, {3, 4}}) + Core::Matrix({{5, 6}, {7, 8}});

    EXPECT_TRUE(Tests::MatrixNear(Core::Matrix({{6, 8}, {10, 12}}), sum, Tolerance));
}

TEST(Matrix, SubtractingMatricesSubtractsEachElement) {
    const Core::Matrix difference = Core::Matrix({{5, 6}, {7, 8}}) - Core::Matrix({{1, 2}, {3, 4}});

    EXPECT_TRUE(Tests::MatrixNear(Core::Matrix({{4, 4}, {4, 4}}), difference, Tolerance));
}

TEST(Matrix, TransposingARectangularMatrixSwapsItsShape) {
    const Core::Matrix transposed = Core::Matrix({{1, 2, 3}, {4, 5, 6}}).Transposed();

    EXPECT_TRUE(Tests::MatrixNear(Core::Matrix({{1, 4}, {2, 5}, {3, 6}}), transposed, Tolerance));
}

TEST(Matrix, MultiplyingMatricesFollowsTheRowByColumnRule) {
    const Core::Matrix product = Core::Matrix({{1, 2, 3}, {4, 5, 6}}) * Core::Matrix({{7, 8}, {9, 10}, {11, 12}});

    EXPECT_TRUE(Tests::MatrixNear(Core::Matrix({{58, 64}, {139, 154}}), product, Tolerance));
}

TEST(Matrix, MultiplyingByTheIdentityLeavesAMatrixUnchanged) {
    const Core::Matrix matrix{{1, 2}, {3, 4}};

    EXPECT_TRUE(Tests::MatrixNear(matrix, matrix * Core::Matrix::Identity(2), Tolerance));
}

TEST(Matrix, MultiplyingByAVectorGivesOneValueForEachRow) {
    const Core::Vector product = Core::Matrix({{1, 2, 3}, {4, 5, 6}}) * Core::Vector{1.0, 0.0, -1.0};

    EXPECT_TRUE(Tests::VectorNear({-2.0, -2.0}, product, Tolerance));
}

TEST(Matrix, TheInverseOfA2x2MatrixMatchesTheClosedForm) {
    const std::expected<Core::Matrix, std::string> inverse = Core::Matrix({{4, 7}, {2, 6}}).Inverse();

    ASSERT_TRUE(Tests::Succeeded(inverse));
    EXPECT_TRUE(Tests::MatrixNear(Core::Matrix({{0.6, -0.7}, {-0.2, 0.4}}), *inverse, Tolerance));
}

TEST(Matrix, AMatrixTimesItsInverseIsTheIdentity) {
    const Core::Matrix matrix{{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};

    const std::expected<Core::Matrix, std::string> inverse = matrix.Inverse();

    ASSERT_TRUE(Tests::Succeeded(inverse));
    EXPECT_TRUE(Tests::MatrixNear(Core::Matrix::Identity(3), matrix * *inverse, Tolerance));
}

TEST(Matrix, InversionSwapsRowsWhenTheDiagonalHasAZero) {
    const Core::Matrix matrix{{0, 1}, {1, 0}};

    const std::expected<Core::Matrix, std::string> inverse = matrix.Inverse();

    ASSERT_TRUE(Tests::Succeeded(inverse));
    EXPECT_TRUE(Tests::MatrixNear(matrix, *inverse, Tolerance));
}

TEST(Matrix, AMatrixWithTinyEntriesIsStillInvertible) {
    // A converged covariance can be this small; only a matrix that is singular relative to its scale has no inverse
    const std::expected<Core::Matrix, std::string> inverse = Core::Matrix({{1e-15, 0}, {0, 1e-15}}).Inverse();

    ASSERT_TRUE(Tests::Succeeded(inverse));
    EXPECT_NEAR((*inverse)(0, 0), 1e15, 1e3);
}

TEST(Matrix, ASingularMatrixHasNoInverse) {
    const std::expected<Core::Matrix, std::string> inverse = Core::Matrix({{1, 2}, {2, 4}}).Inverse();

    ASSERT_FALSE(inverse);
    EXPECT_NE(inverse.error().find("singular"), std::string::npos);
}

TEST(Matrix, TheZeroMatrixHasNoInverse) {
    EXPECT_FALSE(Core::Matrix(2, 2).Inverse());
}

TEST(Matrix, OnlyANonEmptySquareMatrixCanBeInverted) {
    EXPECT_FALSE(Core::Matrix({{1, 2, 3}, {4, 5, 6}}).Inverse());
    EXPECT_FALSE(Core::Matrix().Inverse());
}
