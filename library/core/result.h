/**
 * @file    result.h
 * @brief   The outcome of an operation that can fail: a value, or a message saying why there is none.
 */

#ifndef KALMAN_CPP_RESULT_H
#define KALMAN_CPP_RESULT_H

#include <cassert>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace Core {

/**
 * @struct  Failure
 * @brief   Why an operation failed. Returning one from a function that returns a Result reports the failure.
 */
struct Failure {
    std::string Message;
};

/**
 * @class   Result
 * @brief   Holds the value of an operation that succeeded, or the Failure of one that did not.
 * @details This is what std::expected<T, std::string> gives in C++23, in the part the project uses, so the code
 *          builds with C++17. Value(), operator* and operator-> require a value, and Error() requires a failure;
 *          they are checked with `assert` in debug builds, so test HasValue() first. Discarding a Result is a
 *          warning, because it hides an error.
 */
template <typename T> class [[nodiscard]] Result {
public:
    Result(T value) : m_Outcome(std::move(value)) {}
    Result(Failure failure) : m_Outcome(std::move(failure)) {}

    // Inspection
    bool HasValue() const { return std::holds_alternative<T>(m_Outcome); }
    explicit operator bool() const { return HasValue(); }
    const std::string &Error() const {
        assert(!HasValue());
        return std::get_if<Failure>(&m_Outcome)->Message;
    }

    // Access
    T &Value() {
        assert(HasValue());
        return *std::get_if<T>(&m_Outcome);
    }
    const T &Value() const {
        assert(HasValue());
        return *std::get_if<T>(&m_Outcome);
    }
    T &operator*() { return Value(); }
    const T &operator*() const { return Value(); }
    T *operator->() { return &Value(); }
    const T *operator->() const { return &Value(); }

private:
    std::variant<T, Failure> m_Outcome;
};

/**
 * @class   Result<void>
 * @brief   The outcome of an operation that returns nothing: success, or a Failure.
 */
template <> class [[nodiscard]] Result<void> {
public:
    Result() = default;
    Result(Failure failure) : m_Error(std::move(failure.Message)) {}

    // Inspection
    bool HasValue() const { return !m_Error.has_value(); }
    explicit operator bool() const { return HasValue(); }
    const std::string &Error() const {
        assert(!HasValue());
        return *m_Error;
    }

    // Chaining
    /**
     * @brief   Runs the next step only when this one succeeded, so a sequence stops at its first failure.
     * @param[in] next  Function without arguments that returns a Result<void>.
     * @return  This failure, or the outcome of the next step.
     */
    template <typename Function> Result AndThen(Function next) const { return HasValue() ? next() : *this; }

private:
    std::optional<std::string> m_Error;
};

} // namespace Core

#endif // KALMAN_CPP_RESULT_H
