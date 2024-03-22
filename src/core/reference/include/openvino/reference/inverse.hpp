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
namespace internal {

template <typename T>
void lu_decomposition(const T* input,
                      std::vector<T>& L,
                      std::vector<T>& U,
                      std::vector<size_t>& P,
                      bool& sign,
                      size_t b,
                      size_t n,
                      size_t n_squared) {
    // Make L identity, U a copy of input and P a range(0, n)
    const auto batch_idx = b * n_squared;

    std::fill(L.begin(), L.end(), T{0});
    memcpy(&U[0], &input[batch_idx], sizeof(T) * n_squared);

    for (size_t i = 0; i < n; ++i) {
        L[i * n + i] = T{1};
        P[i] = i;
    }

    for (size_t k = 0; k < n; ++k) {
        // Partial Pivoting
        auto pivot_row = k;
        auto pivot_idx = pivot_row * n;
        const auto k_idx = k * n;

        for (auto i = (k + 1) * n, j = k + 1; i < n_squared; i += n, ++j) {
            if (std::abs(U[i + k]) > std::abs(U[pivot_idx + k])) {
                pivot_row = j;
                pivot_idx = pivot_row * n;
            }
        }

        if (pivot_row != k) {
            // Swap rows in L, U (A) and P
            sign = !sign;
            std::swap(P[k], P[pivot_row]);
            std::swap_ranges(&U[k_idx], &U[k_idx + n], &U[pivot_idx]);
            std::swap_ranges(&L[k_idx], &L[k_idx + n], &L[pivot_idx]);
        }

        const auto remaining_columns = n - k;
        const auto remaining_rows = remaining_columns - 1;

        for (size_t i = 0; i < remaining_rows; ++i) {
            const auto i_idx = (i + k + 1) * n;
            L[i_idx + k] = U[i_idx + k] / U[k_idx + k];
        }

        for (size_t i = 0; i < remaining_rows * remaining_columns; ++i) {
            const auto i_idx = (i / remaining_columns + k + 1) * n;
            const auto j_idx = i % remaining_columns + k;
            U[i_idx + j_idx] = U[i_idx + j_idx] - L[i_idx + k] * U[k_idx + j_idx];
        }
    }
}

template <typename T>
void lu_solve(T* output,
              std::vector<T>& L,
              std::vector<T>& U,
              std::vector<size_t>& P,
              size_t b,
              size_t n,
              size_t n_squared) {
    std::vector<T> X(n);
    std::vector<T> Y(n);

    for (size_t column = 0; column < n; ++column) {
        std::fill(X.begin(), X.end(), T{0});
        std::fill(Y.begin(), Y.end(), T{0});

        // Forward substitution: Ly = Pb
        for (size_t i = 0; i < n; ++i) {
            if (P[i] == column) {
                Y[i] = T{1};
            }
            const auto i_idx = i * n;
            for (size_t j = 0; j < i; ++j) {
                Y[i] -= L[i_idx + j] * Y[j];
            }
        }

        // Backward substitution: Ux = y
        for (size_t i = 0; i < n; ++i) {
            const auto i_adj = n - i - 1;
            const auto i_idx = i_adj * n;
            X[i_adj] = Y[i_adj];
            for (size_t j = i_adj + 1; j < n; ++j) {
                X[i_adj] = X[i_adj] - U[i_idx + j] * X[j];
            }
            X[i_adj] = X[i_adj] / U[i_idx + i_adj];
        }

        const auto batch_column_idx = b * n_squared + column;
        for (size_t row = 0; row < n; ++row) {
            output[batch_column_idx + row * n] = X[row];
        }
    }
}

template <typename T>
void to_adjoint(T* output, std::vector<T>& U, bool sign, size_t b, size_t n, size_t n_squared) {
    T determinant = sign ? T{1} : T{-1};

    for (size_t i = 0; i < n; ++i) {
        determinant *= U[i * n + i];
    }

    const auto batch_idx = b * n_squared;
    for (size_t i = 0; i < n_squared; ++i) {
        output[batch_idx + i] *= determinant;
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
    const auto n_squared = n * n;
    size_t batch_size = 1;

    std::cout << "In Ref:\n";
    if (shape.size() == 3) {
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t x = 0; x < shape[1]; ++x) {
                for (size_t y = 0; y < shape[2]; ++y) {
                    const auto val = input[i * shape[0] + x * shape[1] + y];
                    std::cout << val << ' ';
                }
                std::cout << '\n';
            }
            std::cout << '\n';
        }
    } else if (shape.size() == 2) {
        for (size_t x = 0; x < shape[0]; ++x) {
            for (size_t y = 0; y < shape[1]; ++y) {
                const auto val = input[x * shape[0] + y];
                std::cout << val << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

    for (size_t i = 0; i < shape.size() - 2; ++i) {
        batch_size = batch_size * shape[i];
    }

    std::vector<T> L(n_squared);
    std::vector<T> U(n_squared);
    std::vector<size_t> P(n);

    for (size_t b = 0; b < batch_size; ++b) {
        bool sign = true;

        internal::lu_decomposition(input, L, U, P, sign, b, n, n_squared);

        internal::lu_solve(output, L, U, P, b, n, n_squared);

        if (adjoint) {
            // Multiply by det(A) = det(U)
            internal::to_adjoint(output, U, sign, b, n, n_squared);
        }
    }
}
}  // namespace reference
}  // namespace ov
