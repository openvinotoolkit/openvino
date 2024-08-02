// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/inverse.hpp"

namespace ov {
namespace reference {
namespace internal {

void lu_decomposition(std::vector<float>& input,
                      std::vector<float>& L,
                      std::vector<float>& U,
                      std::vector<size_t>& P,
                      const bool adjoint,
                      const size_t b,
                      const size_t n,
                      const size_t n_squared) {
    // Make L identity, U a copy of input and P a range(0, n)
    const auto batch_idx = b * n_squared;

    std::fill(L.begin(), L.end(), 0.0f);

    // To compute adjoint (conjugate transpose)
    // it is enough to input a tranpose of the input matrix
    if (!adjoint) {
        memcpy(&U[0], &input[batch_idx], sizeof(float) * n_squared);
    } else {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                U[j * n + i] = input[batch_idx + i * n + j];
            }
        }
    }

    for (size_t i = 0; i < n; ++i) {
        L[i * n + i] = 1.0f;
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

void lu_solve(std::vector<float>& output,
              std::vector<float>& L,
              std::vector<float>& U,
              std::vector<size_t>& P,
              const size_t b,
              const size_t n,
              const size_t n_squared) {
    std::vector<float> X(n);
    std::vector<float> Y(n);

    for (size_t column = 0; column < n; ++column) {
        std::fill(X.begin(), X.end(), 0.0f);
        std::fill(Y.begin(), Y.end(), 0.0f);

        // Forward substitution: Ly = Pb
        for (size_t i = 0; i < n; ++i) {
            if (P[i] == column) {
                Y[i] = 1.0f;
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
}  // namespace internal
}  // namespace reference
}  // namespace ov
