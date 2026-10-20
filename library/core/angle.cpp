/**
 * @file    angle.cpp
 * @brief   Angles in radians that wrap around, for filters whose states or measurements are headings and bearings.
 */

#include "angle.h"

#include <cmath>

namespace Core {

namespace {

constexpr double FullTurn = 2.0 * Pi;

} // namespace

/**
 * @brief   Brings an angle into the range from just above -pi up to and including pi.
 * @param[in] angle_radians  Angle of any size.
 * @return  The same direction, within one turn centered on zero.
 */
double WrapAngle(const double angle_radians) {
    double wrapped = std::fmod(angle_radians + Pi, FullTurn);
    if (wrapped <= 0.0) {
        wrapped += FullTurn;
    }
    return wrapped - Pi;
}

/**
 * @brief   Subtracts two angles along the shorter way around the circle.
 * @param[in] minuend_radians     Angle to subtract from.
 * @param[in] subtrahend_radians  Angle to subtract.
 * @return  The signed difference, within one turn centered on zero: 179 degrees minus -179 degrees is -2 degrees.
 */
double AngleDifference(const double minuend_radians, const double subtrahend_radians) {
    return WrapAngle(minuend_radians - subtrahend_radians);
}

} // namespace Core
