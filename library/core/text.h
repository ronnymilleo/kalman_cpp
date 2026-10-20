/**
 * @file    text.h
 * @brief   Builds the messages of failures from text and numbers.
 */

#ifndef KALMAN_CPP_TEXT_H
#define KALMAN_CPP_TEXT_H

#include <sstream>
#include <string>

namespace Core {

/**
 * @brief   Joins text and numbers into one string, such as Concat("expected ", 3, " elements").
 * @param[in] parts  Anything that can be written to a stream.
 * @return  The parts written one after the other.
 */
template <typename... Parts> std::string Concat(const Parts &...parts) {
    std::ostringstream stream;
    (stream << ... << parts);
    return stream.str();
}

} // namespace Core

#endif // KALMAN_CPP_TEXT_H
