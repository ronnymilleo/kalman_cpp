/**
 * @file    kalman_math_tests.cpp
 * @brief   Unit tests for the equations and checks shared by the filters.
 */

#include "filters/kalman_math.h"

#include "test_support.h"

#include <gtest/gtest.h>

namespace {

constexpr double Tolerance = 1e-12;

} // namespace

TEST(KalmanMath, ACheckPassesWhenTheShapeMatches) {
    EXPECT_TRUE(Tests::Succeeded(Core::CheckShape("The matrix", Core::Matrix(2, 3), 2, 3)));
}

TEST(KalmanMath, ACheckNamesTheMatrixAndBothShapesWhenTheyDiffer) {
    const std::expected<void, std::string> outcome = Core::CheckShape("The matrix", Core::Matrix(2, 3), 3, 3);

    ASSERT_FALSE(outcome);
    EXPECT_EQ(outcome.error(), "The matrix must be 3x3, not 2x3");
}

TEST(KalmanMath, ThePredictedCovarianceIsFPFTransposedPlusQ) {
    const Core::Matrix transition{{1.0, 0.1}, {0.0, 1.0}};

    const Core::Matrix covariance =
        Core::PredictCovariance(transition, Core::Matrix::Identity(2), Core::Matrix({{0.01, 0.0}, {0.0, 0.01}}));

    EXPECT_TRUE(Tests::MatrixNear(Core::Matrix({{1.02, 0.1}, {0.1, 1.01}}), covariance, Tolerance));
}

TEST(KalmanMath, CorrectingWithEqualTrustLandsHalfwayToTheMeasurement) {
    const Core::Estimate prior{.State = {0.0}, .Covariance = {{1.0}}};

    const std::expected<Core::Estimate, std::string> corrected =
        Core::CorrectEstimate(prior, Core::Matrix({{1.0}}), Core::Matrix({{1.0}}), {2.0});

    // With P = R = 1 the gain is P / (P + R) = 0.5, so the estimate moves from 0 to 1 and the variance halves
    ASSERT_TRUE(Tests::Succeeded(corrected));
    EXPECT_TRUE(Tests::VectorNear({1.0}, corrected->State, Tolerance));
    EXPECT_TRUE(Tests::MatrixNear(Core::Matrix({{0.5}}), corrected->Covariance, Tolerance));
}

TEST(KalmanMath, CorrectingWithoutAnyUncertaintyFailsBecauseTheInnovationCovarianceIsSingular) {
    const Core::Estimate prior{.State = {0.0}, .Covariance = {{0.0}}};

    const std::expected<Core::Estimate, std::string> corrected =
        Core::CorrectEstimate(prior, Core::Matrix({{1.0}}), Core::Matrix({{0.0}}), {2.0});

    EXPECT_FALSE(corrected);
}
