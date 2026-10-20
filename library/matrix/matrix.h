/**
 * @file    matrix.h
 * @brief   Dense matrices with the linear algebra the Kalman filters need.
 */

#ifndef KALMAN_CPP_MATRIX_H
#define KALMAN_CPP_MATRIX_H

#include <cstddef>
#include <expected>
#include <initializer_list>
#include <string>
#include <vector>

namespace Core {

// Column vector, such as a state or a measurement
using Vector = std::vector<double>;

/**
 * @class   Matrix
 * @brief   Dense matrix of doubles stored in row-major order.
 * @details A default-constructed matrix is empty. Element access and the arithmetic operators have preconditions
 *          instead of error returns, so the expressions of the filter equations stay readable: indexes lie inside the
 *          matrix and the shapes of the operands fit. They are checked with `assert` in debug builds; code that
 *          takes shapes from outside must validate them first.
 */
class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::initializer_list<std::initializer_list<double>> values);

    // Construction
    static Matrix Identity(std::size_t size);

    // Shape
    std::size_t GetRows() const;
    std::size_t GetCols() const;
    bool IsEmpty() const;
    bool IsSquare() const;
    bool HasSameShape(const Matrix &other) const;

    // Elements
    double &operator()(std::size_t row, std::size_t col);
    double operator()(std::size_t row, std::size_t col) const;

    // Transforms
    Matrix Transposed() const;
    [[nodiscard]] std::expected<Matrix, std::string> Inverse() const;

    bool operator==(const Matrix &) const = default;

private:
    std::size_t m_Rows = 0;
    std::size_t m_Cols = 0;
    std::vector<double> m_Data;
};

// Arithmetic
Matrix operator+(const Matrix &lhs, const Matrix &rhs);
Matrix operator-(const Matrix &lhs, const Matrix &rhs);
Matrix operator*(const Matrix &lhs, const Matrix &rhs);
Vector operator*(const Matrix &matrix, const Vector &vector);

} // namespace Core

#endif // KALMAN_CPP_MATRIX_H
