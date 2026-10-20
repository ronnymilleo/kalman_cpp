/**
 * @file    demo_support.cpp
 * @brief   What the demonstrations share: a fixed noise seed and the checks that compare a filter with the truth.
 */

#include "demo_support.h"

#include <cmath>
#include <format>
#include <iostream>

namespace Demo {

namespace {

// 2^32, the number of values std::mt19937 can produce
constexpr double GeneratorRange = 4294967296.0;
constexpr double TwoPi = 6.283185307179586;

} // namespace

/**
 * @brief   Creates a noise source.
 * @param[in] seed  Starting point; the same seed always gives the same sequence.
 */
NoiseSource::NoiseSource(const std::uint32_t seed) : m_Generator(seed) {
}

/**
 * @brief   Draws a number from a Gaussian distribution centered on zero.
 * @param[in] standard_deviation  Spread of the distribution.
 * @return  The number drawn.
 */
double NoiseSource::Gaussian(const double standard_deviation) {
    // The first uniform number is shifted to lie in (0, 1], because the logarithm of zero is not finite
    const double first = (static_cast<double>(m_Generator()) + 1.0) / (GeneratorRange + 1.0);
    const double second = static_cast<double>(m_Generator()) / GeneratorRange;
    return standard_deviation * std::sqrt(-2.0 * std::log(first)) * std::cos(TwoPi * second);
}

/**
 * @brief   Summarizes a series of errors in a single number.
 * @param[in] errors  Differences between estimates, or measurements, and the truth; not empty.
 * @return  The square root of the mean of the squares.
 */
double RootMeanSquare(const std::vector<double> &errors) {
    double sum = 0.0;
    for (const double error : errors) {
        sum += error * error;
    }
    return std::sqrt(sum / static_cast<double>(errors.size()));
}

/**
 * @brief   Prints how much better the filter is than the raw measurements and fails when it is not good enough.
 * @param[in] what            What is compared, such as "position error".
 * @param[in] filtered_error  Error of the filter's estimates.
 * @param[in] raw_error       Error of the raw measurements.
 * @param[in] required_share  Largest accepted ratio between the two, such as 0.5 for at least twice as good.
 * @return  Nothing when the filter passes, otherwise a message with the three numbers.
 */
std::expected<void, std::string> CheckImprovement(const std::string_view what, const double filtered_error,
                                                  const double raw_error, const double required_share) {
    std::cout << std::format("{}: raw measurements {:.4f}, filter {:.4f} ({:.1f}x smaller)\n", what, raw_error,
                             filtered_error, raw_error / filtered_error);
    if (filtered_error > required_share * raw_error) {
        return std::unexpected(
            std::format("The filter's {} ({:.4f}) is not at most {:.0f}% of the raw measurements' ({:.4f})", what,
                        filtered_error, required_share * 100.0, raw_error));
    }
    return {};
}

} // namespace Demo
