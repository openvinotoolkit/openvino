// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/shape.hpp>
#include <cstdlib>
#include <vector>

namespace ov {
namespace reference {
namespace inverse {

template <typename T>
void lu_decomposition(const T* input,
                      std::vector<std::vector<T>>& L,
                      std::vector<std::vector<T>>& U,
                      std::vector<T>& P,
                      bool& sign,
                      size_t b,
                      size_t n) {
    // Make L identity, U a copy of input and P a range(0, n)
    size_t batch_idx = b * n * n;
    for (size_t i = 0; i < n; ++i) {
        P[i] = static_cast<T>(i);
        L[i][i] = static_cast<T>(1);

        size_t i_idx = i * n;
        for (size_t j = 0; j < n; ++j) {
            U[i][j] = input[batch_idx + i_idx + j];
        }
    }

    for (size_t k = 0; k < n; ++k) {
        // Partial Pivoting
        size_t pivot_row = k;
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(U[i][k]) > std::abs(U[pivot_row][k])) {
                pivot_row = i;
            }
        }

        if (pivot_row != k) {
            // Swap rows in L, U (A) and P
            std::swap(U[k], U[pivot_row]);
            std::swap(L[k], L[pivot_row]);
            std::swap(P[k], P[pivot_row]);
            sign = !sign;
        }

        for (size_t i = k + 1; i < n; ++i) {
            L[i][k] = U[i][k] / U[k][k];
            for (size_t j = k; j < n; ++j) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }
    }
}

template <typename T>
void lu_solve(T* output,
              std::vector<std::vector<T>>& L,
              std::vector<std::vector<T>>& U,
              std::vector<T>& P,
              size_t b,
              size_t n,
              size_t column) {
    std::vector<T> B(n, static_cast<T>(0));
    std::vector<T> X(n, static_cast<T>(0));
    std::vector<T> Y(n, static_cast<T>(0));
    B[column] = static_cast<T>(1);

    // Forward substitution: Ly = Pb
    for (size_t i = 0; i < n; ++i) {
        Y[i] = B[P[i]];
        for (size_t j = 0; j < i; ++j) {
            Y[i] -= L[i][j] * Y[j];
        }
    }

    // Backward substitution: Ux = y
    for (size_t i = n - 1; i >= 0; --i) {
        X[i] = Y[i];
        for (size_t j = i + 1; j < n; ++j) {
            X[i] -= U[i][j] * X[j];
        }

        // Necessary since for i = 0, i-- underflows back to max(size_t)
        X[i] /= U[i][i];
        if (i == 0) {
            break;
        }
    }

    size_t batch_idx = b * n * n;
    for (size_t row = 0; row < n; ++row) {
        output[batch_idx + row * n + column] = X[row];
    }
}

template <typename T>
void to_adjoint(T* output, std::vector<std::vector<T>>& U, bool sign, size_t b, size_t n) {
    T determinant = sign ? 1.0f : -1.0f;

    for (size_t i = 0; i < n; ++i) {
        determinant *= U[i][i];
    }

    size_t batch_idx = b * n * n;
    for (size_t idx = 0; idx < n * n; ++idx) {
        output[batch_idx + idx] *= determinant;
    }
}

/**
 * @brief Inverse operation computes the inverse of the input tensor.
 *
 * @param input Input square matrix (matrices) to compute the inverse for.
 * @param shape Shape of the input matrix.
 * @param output Output matrix, inverse of the input matrix.
 * @param adjoint Boolean that determines whether to return a normal inverse or adjoint (conjugate transpose) of the
 *input matrix.
 **/
template <typename T>
void inverse(const T* input, T* output, const Shape& shape, const bool adjoint) {
    const size_t n = shape.back();
    const auto total_elements = shape_size<Shape>(shape);
    const auto batch_size = total_elements / n / n;

    for (size_t b = 0; b < batch_size; ++b) {
        std::vector<std::vector<T>> L(n, std::vector<T>(n, static_cast<T>(0)));
        std::vector<std::vector<T>> U(n, std::vector<T>(n, static_cast<T>(0)));
        std::vector<T> P(n);
        bool sign = true;

        lu_decomposition(input, L, U, P, sign, b, n);

        for (size_t column = 0; column < n; ++column) {
            lu_solve(output, L, U, P, b, n, column);
        }

        if (adjoint) {
            // Multiply by det(A) (= det(U) * sign)
            to_adjoint(output, U, sign, b, n);
        }
    }
}
}  // namespace inverse
}  // namespace reference

namespace op {
namespace inverse {
namespace validate {
void input_types(const Node* op);
}  // namespace validate
}  // namespace inverse
}  // namespace op
}  // namespace ov
