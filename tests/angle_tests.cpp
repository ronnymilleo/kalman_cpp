/**
 * @file    angle_tests.cpp
 * @brief   Unit tests for the angle module.
 */

#include "core/angle.h"

#include <gtest/gtest.h>

namespace {

constexpr double Tolerance = 1e-12;
constexpr double Degrees = Core::Pi / 180.0;

} // namespace

TEST(Angle, AnAngleInsideOneTurnIsLeftAlone) {
    EXPECT_NEAR(Core::WrapAngle(0.5), 0.5, Tolerance);
    EXPECT_NEAR(Core::WrapAngle(-2.5), -2.5, Tolerance);
    EXPECT_NEAR(Core::WrapAngle(0.0), 0.0, Tolerance);
}

TEST(Angle, AnAnglePastPiComesBackOnTheNegativeSide) {
    EXPECT_NEAR(Core::WrapAngle(Core::Pi + 0.2), -Core::Pi + 0.2, Tolerance);
}

TEST(Angle, AnAngleBelowMinusPiComesBackOnThePositiveSide) {
    EXPECT_NEAR(Core::WrapAngle(-Core::Pi - 0.2), Core::Pi - 0.2, Tolerance);
}

TEST(Angle, PiAndMinusPiAreTheSameDirectionAndWrapToPi) {
    EXPECT_NEAR(Core::WrapAngle(Core::Pi), Core::Pi, Tolerance);
    EXPECT_NEAR(Core::WrapAngle(-Core::Pi), Core::Pi, Tolerance);
}

TEST(Angle, WholeTurnsWrapToTheSameAngle) {
    EXPECT_NEAR(Core::WrapAngle(1.0 + 10.0 * 2.0 * Core::Pi), 1.0, 1e-9);
    EXPECT_NEAR(Core::WrapAngle(1.0 - 7.0 * 2.0 * Core::Pi), 1.0, 1e-9);
}

TEST(Angle, TheDifferenceOfAnglesAcrossTheBoundaryIsTheShortWayAround) {
    // 179 degrees minus -179 degrees is -2 degrees, not 358
    EXPECT_NEAR(Core::AngleDifference(179.0 * Degrees, -179.0 * Degrees), -2.0 * Degrees, 1e-9);
    EXPECT_NEAR(Core::AngleDifference(-179.0 * Degrees, 179.0 * Degrees), 2.0 * Degrees, 1e-9);
}
