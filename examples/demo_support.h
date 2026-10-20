/**
 * @file    demo_support.h
 * @brief   What the demonstrations share: a fixed noise seed and the checks that compare a filter with the truth.
 */

#ifndef KALMAN_CPP_DEMO_SUPPORT_H
#define KALMAN_CPP_DEMO_SUPPORT_H

#include <cstdint>
#include <expected>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace Demo {

// Seeds the simulated noise, so every run of a demonstration prints the same numbers
constexpr unsigned int NoiseSeed = 2024;

// Measurements and estimates are compared once the filter has had this many steps to settle
constexpr int SettlingSteps = 10;

/**
 * @class   NoiseSource
 * @brief   Gaussian noise that is the same on every platform.
 * @details The distributions of the standard library produce different numbers on different implementations, so a
 *          demonstration that checks itself against them could pass on one platform and fail on another. The raw
 *          output of std::mt19937 is fixed by the standard, and the Box-Muller transform below turns it into
 *          Gaussian noise.
 */
class NoiseSource {
public:
    explicit NoiseSource(std::uint32_t seed);

    double Gaussian(double standard_deviation);

private:
    std::mt19937 m_Generator;
};

// Checks
double RootMeanSquare(const std::vector<double> &errors);
[[nodiscard]] std::expected<void, std::string> CheckImprovement(std::string_view what, double filtered_error,
                                                                double raw_error, double required_share);

} // namespace Demo

#endif // KALMAN_CPP_DEMO_SUPPORT_H
