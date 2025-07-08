//
// Created by Ronny Milleo on 08/07/25.
//

#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <vector>

void matrix_add(const std::vector<std::vector<double>>& A,
                const std::vector<std::vector<double>>& B,
                std::vector<std::vector<double>>& result);

void matrix_subtract(const std::vector<std::vector<double>>& A,
                     const std::vector<std::vector<double>>& B,
                     std::vector<std::vector<double>>& result);

void matrix_transpose(const std::vector<std::vector<double>>& A,
                      std::vector<std::vector<double>>& result);

void matrix_multiply(const std::vector<std::vector<double>>& A,
                     const std::vector<std::vector<double>>& B,
                     std::vector<std::vector<double>>& result);

void matrix_vector_multiply(const std::vector<std::vector<double>>& A,
                            const std::vector<double>& v,
                            std::vector<double>& result);

void matrix_identity(std::vector<std::vector<double>>& result);

void matrix_inverse(const std::vector<std::vector<double>>& A,
                    std::vector<std::vector<double>>& result);

#endif //MATRIX_MATH_H
