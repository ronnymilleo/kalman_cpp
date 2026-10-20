/**
 * @file    matrix.cpp
 * @brief   Dense matrices with the linear algebra the Kalman filters need.
 */

#include "matrix.h"

#include "core/text.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

namespace Core {

namespace {

// A pivot smaller than this share of the largest entry is treated as zero; a fixed absolute limit would reject
// well-conditioned matrices whose entries are all tiny, such as a converged covariance
constexpr double SingularRelativeTolerance = 1e-12;

double LargestMagnitude(const Matrix &matrix) {
    double largest = 0.0;
    for (std::size_t row = 0; row < matrix.GetRows(); row++) {
        for (std::size_t col = 0; col < matrix.GetCols(); col++) {
            largest = std::max(largest, std::abs(matrix(row, col)));
        }
    }
    return largest;
}

} // namespace

/**
 * @brief   Creates a matrix filled with zeros.
 * @param[in] rows  Number of rows.
 * @param[in] cols  Number of columns.
 */
Matrix::Matrix(const std::size_t rows, const std::size_t cols) : m_Rows(rows), m_Cols(cols), m_Data(rows * cols, 0.0) {
}

/**
 * @brief   Creates a matrix from nested lists of values, one inner list per row.
 * @param[in] values  Rows of the matrix; all of them must have the same length.
 */
Matrix::Matrix(const std::initializer_list<std::initializer_list<double>> values)
    : m_Rows(values.size()), m_Cols(values.size() == 0 ? 0 : values.begin()->size()) {
    m_Data.reserve(m_Rows * m_Cols);
    for (const std::initializer_list<double> &row : values) {
        assert(row.size() == m_Cols && "all rows of a matrix must have the same length");
        m_Data.insert(m_Data.end(), row.begin(), row.end());
    }
}

/**
 * @brief   Creates the identity matrix of a given size.
 * @param[in] size  Number of rows and columns.
 * @return  Square matrix with ones on the diagonal and zeros elsewhere.
 */
Matrix Matrix::Identity(const std::size_t size) {
    Matrix identity(size, size);
    for (std::size_t index = 0; index < size; index++) {
        identity(index, index) = 1.0;
    }
    return identity;
}

/**
 * @brief   Gets the number of rows.
 * @return  Number of rows.
 */
std::size_t Matrix::GetRows() const {
    return m_Rows;
}

/**
 * @brief   Gets the number of columns.
 * @return  Number of columns.
 */
std::size_t Matrix::GetCols() const {
    return m_Cols;
}

/**
 * @brief   Tells whether the matrix has no elements.
 * @return  True for a matrix without rows or columns.
 */
bool Matrix::IsEmpty() const {
    return m_Data.empty();
}

/**
 * @brief   Tells whether the matrix has as many rows as columns.
 * @return  True for a square matrix; an empty matrix counts as square.
 */
bool Matrix::IsSquare() const {
    return m_Rows == m_Cols;
}

/**
 * @brief   Compares the shape of two matrices.
 * @param[in] other  Matrix to compare with.
 * @return  True when both have the same number of rows and columns.
 */
bool Matrix::HasSameShape(const Matrix &other) const {
    return m_Rows == other.m_Rows && m_Cols == other.m_Cols;
}

/**
 * @brief   Accesses an element for reading and writing.
 * @param[in] row  Row index, from zero.
 * @param[in] col  Column index, from zero.
 * @return  Reference to the element.
 * @note    Both indexes must lie inside the matrix.
 */
double &Matrix::operator()(const std::size_t row, const std::size_t col) {
    assert(row < m_Rows && col < m_Cols);
    return m_Data[row * m_Cols + col];
}

/**
 * @brief   Reads an element.
 * @param[in] row  Row index, from zero.
 * @param[in] col  Column index, from zero.
 * @return  Value of the element.
 * @note    Both indexes must lie inside the matrix.
 */
double Matrix::operator()(const std::size_t row, const std::size_t col) const {
    assert(row < m_Rows && col < m_Cols);
    return m_Data[row * m_Cols + col];
}

/**
 * @brief   Swaps the rows and columns.
 * @return  The transpose, with as many rows as this matrix has columns.
 */
Matrix Matrix::Transposed() const {
    Matrix transposed(m_Cols, m_Rows);
    for (std::size_t row = 0; row < m_Rows; row++) {
        for (std::size_t col = 0; col < m_Cols; col++) {
            transposed(col, row) = (*this)(row, col);
        }
    }
    return transposed;
}

/**
 * @brief   Inverts the matrix by Gauss-Jordan elimination with partial pivoting.
 * @return  The inverse, or a message saying why there is none: the matrix is not square, or it is singular.
 */
Result<Matrix> Matrix::Inverse() const {
    if (!IsSquare() || IsEmpty()) {
        return Failure{Concat("Only a non-empty square matrix can be inverted, not ", m_Rows, "x", m_Cols)};
    }

    const std::size_t size = m_Rows;
    const double tolerance = SingularRelativeTolerance * LargestMagnitude(*this);

    // Work on [A | I]; once A has been reduced to the identity, the right half holds its inverse
    Matrix augmented(size, 2 * size);
    for (std::size_t row = 0; row < size; row++) {
        for (std::size_t col = 0; col < size; col++) {
            augmented(row, col) = (*this)(row, col);
        }
        augmented(row, size + row) = 1.0;
    }

    for (std::size_t pivot_index = 0; pivot_index < size; pivot_index++) {
        std::size_t pivot_row = pivot_index;
        for (std::size_t row = pivot_index + 1; row < size; row++) {
            if (std::abs(augmented(row, pivot_index)) > std::abs(augmented(pivot_row, pivot_index))) {
                pivot_row = row;
            }
        }
        if (std::abs(augmented(pivot_row, pivot_index)) <= tolerance) {
            return Failure{"The matrix is singular and has no inverse"};
        }
        if (pivot_row != pivot_index) {
            for (std::size_t col = 0; col < 2 * size; col++) {
                std::swap(augmented(pivot_index, col), augmented(pivot_row, col));
            }
        }

        const double pivot = augmented(pivot_index, pivot_index);
        for (std::size_t col = 0; col < 2 * size; col++) {
            augmented(pivot_index, col) /= pivot;
        }
        for (std::size_t row = 0; row < size; row++) {
            if (row == pivot_index) {
                continue;
            }
            const double factor = augmented(row, pivot_index);
            for (std::size_t col = 0; col < 2 * size; col++) {
                augmented(row, col) -= factor * augmented(pivot_index, col);
            }
        }
    }

    Matrix inverse(size, size);
    for (std::size_t row = 0; row < size; row++) {
        for (std::size_t col = 0; col < size; col++) {
            inverse(row, col) = augmented(row, size + col);
        }
    }
    return inverse;
}

/**
 * @brief   Compares two matrices element by element.
 * @param[in] other  Matrix to compare with.
 * @return  True when both have the same shape and exactly the same elements.
 */
bool Matrix::operator==(const Matrix &other) const {
    return m_Rows == other.m_Rows && m_Cols == other.m_Cols && m_Data == other.m_Data;
}

/**
 * @brief   Tells whether two matrices differ in shape or in any element.
 * @param[in] other  Matrix to compare with.
 * @return  True when the matrices are not equal.
 */
bool Matrix::operator!=(const Matrix &other) const {
    return !(*this == other);
}

/**
 * @brief   Adds two matrices element by element.
 * @param[in] lhs  First matrix.
 * @param[in] rhs  Second matrix, with the same shape as the first.
 * @return  The sum.
 */
Matrix operator+(const Matrix &lhs, const Matrix &rhs) {
    assert(lhs.HasSameShape(rhs));
    Matrix sum(lhs.GetRows(), lhs.GetCols());
    for (std::size_t row = 0; row < lhs.GetRows(); row++) {
        for (std::size_t col = 0; col < lhs.GetCols(); col++) {
            sum(row, col) = lhs(row, col) + rhs(row, col);
        }
    }
    return sum;
}

/**
 * @brief   Subtracts a matrix from another element by element.
 * @param[in] lhs  Matrix to subtract from.
 * @param[in] rhs  Matrix to subtract, with the same shape as the first.
 * @return  The difference.
 */
Matrix operator-(const Matrix &lhs, const Matrix &rhs) {
    assert(lhs.HasSameShape(rhs));
    Matrix difference(lhs.GetRows(), lhs.GetCols());
    for (std::size_t row = 0; row < lhs.GetRows(); row++) {
        for (std::size_t col = 0; col < lhs.GetCols(); col++) {
            difference(row, col) = lhs(row, col) - rhs(row, col);
        }
    }
    return difference;
}

/**
 * @brief   Multiplies two matrices.
 * @param[in] lhs  Left matrix, `m x k`.
 * @param[in] rhs  Right matrix, `k x n`.
 * @return  The product, `m x n`.
 */
Matrix operator*(const Matrix &lhs, const Matrix &rhs) {
    assert(lhs.GetCols() == rhs.GetRows());
    Matrix product(lhs.GetRows(), rhs.GetCols());
    for (std::size_t row = 0; row < lhs.GetRows(); row++) {
        for (std::size_t col = 0; col < rhs.GetCols(); col++) {
            for (std::size_t inner = 0; inner < lhs.GetCols(); inner++) {
                product(row, col) += lhs(row, inner) * rhs(inner, col);
            }
        }
    }
    return product;
}

/**
 * @brief   Multiplies a matrix by a column vector.
 * @param[in] matrix  Matrix, `m x n`.
 * @param[in] vector  Vector with `n` elements.
 * @return  Vector with `m` elements.
 */
Vector operator*(const Matrix &matrix, const Vector &vector) {
    assert(matrix.GetCols() == vector.size());
    Vector product(matrix.GetRows(), 0.0);
    for (std::size_t row = 0; row < matrix.GetRows(); row++) {
        for (std::size_t col = 0; col < matrix.GetCols(); col++) {
            product[row] += matrix(row, col) * vector[col];
        }
    }
    return product;
}

} // namespace Core
