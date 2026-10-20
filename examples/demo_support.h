/**
 * @file    demo_support.h
 * @brief   What the demonstrations share: a fixed noise seed and the checks that compare a filter with the truth.
 */

#ifndef KALMAN_CPP_DEMO_SUPPORT_H
#define KALMAN_CPP_DEMO_SUPPORT_H

#include "core/result.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <sstream>
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

// Text
void ApplyFormatSpec(std::ostream &stream, const std::string &spec);

/**
 * @brief   Writes values into a template, like std::format does in C++20, which this C++17 project does not have.
 * @details Each `{}` takes the next argument. A `{:W}` pads it to a width of W characters, `{:.Pf}` writes a number
 * with P decimals and `{:W.Pf}` does both.
 * @param[in] format  Template text with one `{...}` for each argument.
 * @param[in] args    Values to write, anything a stream can write.
 * @return  The text with the values in place.
 */
template <typename... Args> std::string Format(const std::string &format, const Args &...args) {
    std::ostringstream stream;
    std::size_t position = 0;
    const auto write_next = [&](const auto &value) {
        const std::size_t open = format.find('{', position);
        const std::size_t close = format.find('}', open);
        stream << format.substr(position, open - position);
        std::ostringstream field;
        ApplyFormatSpec(field, format.substr(open + 1, close - open - 1));
        field << value;
        stream << field.str();
        position = close + 1;
    };
    (write_next(args), ...);
    stream << format.substr(position);
    return stream.str();
}

// Checks
double RootMeanSquare(const std::vector<double> &errors);
[[nodiscard]] Core::Result<void> CheckImprovement(std::string_view what, double filtered_error, double raw_error,
                                                  double required_share);

} // namespace Demo

#endif // KALMAN_CPP_DEMO_SUPPORT_H
