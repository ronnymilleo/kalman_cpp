/**
 * @file    angle.h
 * @brief   Angles in radians that wrap around, for filters whose states or measurements are headings and bearings.
 */

#ifndef KALMAN_CPP_ANGLE_H
#define KALMAN_CPP_ANGLE_H

namespace Core {

// Half a turn in radians
constexpr double Pi = 3.14159265358979323846;

// Wrapping
double WrapAngle(double angle_radians);
double AngleDifference(double minuend_radians, double subtrahend_radians);

} // namespace Core

#endif // KALMAN_CPP_ANGLE_H
