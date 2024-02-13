// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstdlib>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
namespace inverse {
namespace internal {

template <typename T>
void lu_decomposition(const T* input,
                      std::vector<std::vector<T>>& L,
                      std::vector<std::vector<T>>& U,
                      std::vector<T>& P,
                      bool& sign,
                      size_t b,
                      size_t n) {
    // Make L identity, U a copy of input and P a range(0, n)
    const auto batch_idx = b * n * n;
    for (size_t i = 0; i < n; ++i) {
        P[i] = static_cast<T>(i);
        L[i][i] = T{1};

        auto i_idx = i * n;
        for (size_t j = 0; j < n; ++j) {
            U[i][j] = input[batch_idx + i_idx + j];
        }
    }

    for (size_t k = 0; k < n; ++k) {
        // Partial Pivoting
        auto pivot_row = k;
        for (auto i = k + 1; i < n; ++i) {
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

        for (auto i = k + 1; i < n; ++i) {
            L[i][k] = U[i][k] / U[k][k];
            for (auto j = k; j < n; ++j) {
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
    std::vector<T> B(n, T{0});
    std::vector<T> X(n, T{0});
    std::vector<T> Y(n, T{0});
    B[column] = T{1};

    // Forward substitution: Ly = Pb
    for (size_t i = 0; i < n; ++i) {
        Y[i] = B[P[i]];
        for (size_t j = 0; j < i; ++j) {
            Y[i] -= L[i][j] * Y[j];
        }
    }

    // Backward substitution: Ux = y
    for (int i = static_cast<int>(n - 1); i >= 0; --i) {
        X[i] = Y[i];
        for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
            X[i] -= U[i][j] * X[j];
        }
        X[i] /= U[i][i];
    }

    size_t batch_idx = b * n * n;
    for (size_t row = 0; row < n; ++row) {
        output[batch_idx + row * n + column] = X[row];
    }
}

template <typename T>
void to_adjoint(T* output, std::vector<std::vector<T>>& U, bool sign, size_t b, size_t n) {
    T determinant = sign ? T{1} : T{-1};

    for (size_t i = 0; i < n; ++i) {
        determinant *= U[i][i];
    }

    const auto batch_idx = b * n * n;
    for (auto idx = batch_idx; idx < batch_idx + n * n; ++idx) {
        output[idx] *= determinant;
    }
}
}  // namespace internal

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
        std::vector<std::vector<T>> L(n, std::vector<T>(n, T{0}));
        std::vector<std::vector<T>> U(n, std::vector<T>(n, T{0}));
        std::vector<T> P(n);
        bool sign = true;

        internal::lu_decomposition(input, L, U, P, sign, b, n);

        for (size_t column = 0; column < n; ++column) {
            internal::lu_solve(output, L, U, P, b, n, column);
        }

        if (adjoint) {
            internal::to_adjoint(output, U, sign, b, n);
        }
    }
}
}  // namespace inverse
}  // namespace reference
}  // namespace ov
