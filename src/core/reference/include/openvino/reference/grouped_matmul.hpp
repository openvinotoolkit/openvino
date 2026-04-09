// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov::reference {
namespace details {

/// \brief Simple 2D matmul: out = A @ B
/// \param A Input matrix A of shape (M, K)
/// \param B Input matrix B of shape (K, N)
/// \param out Output matrix of shape (M, N) - must be pre-zeroed
/// \param M Number of rows in A
/// \param K Shared dimension
/// \param N Number of columns in B
template <typename T>
void simple_matmul(const T* A, const T* B, T* out, size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const T a_val = A[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                out[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

}  // namespace details

/// \brief Reference kernel for GroupedMatMul computation.
///
/// Supports three input combinations:
/// - Case 1 (2D × 3D): MoE forward pass with offsets
/// - Case 2 (3D × 3D): Batched uniform groups, no offsets
/// - Case 3 (2D × 2D): MoE weight gradient with offsets
///
/// \tparam T Data type of input and output tensors.
/// \tparam TIdx Data type for offset indices.
///
/// \param mat_a Pointer to first input tensor.
/// \param mat_b Pointer to second input tensor.
/// \param offsets Pointer to offsets tensor (nullptr for 3D×3D case).
/// \param out Pointer to output tensor (pre-allocated).
/// \param mat_a_shape Shape of mat_a.
/// \param mat_b_shape Shape of mat_b.
/// \param out_shape Shape of output.
/// \param num_groups Number of groups (inferred from offsets or mat_b).
template <typename T, typename TIdx = int32_t>
void grouped_matmul(const T* mat_a,
                    const T* mat_b,
                    const TIdx* offsets,
                    T* out,
                    const Shape& mat_a_shape,
                    const Shape& mat_b_shape,
                    const Shape& out_shape,
                    size_t num_groups) {
    const auto a_ndim = mat_a_shape.size();
    const auto b_ndim = mat_b_shape.size();

    // Case 2: 3D × 3D (batched, uniform group sizes)
    if (a_ndim == 3 && b_ndim == 3) {
        const size_t G = mat_a_shape[0];
        const size_t M = mat_a_shape[1];
        const size_t K = mat_a_shape[2];
        const size_t N = mat_b_shape[2];

        const size_t mat_a_group_stride = M * K;
        const size_t mat_b_group_stride = K * N;
        const size_t out_group_stride = M * N;

        for (size_t g = 0; g < G; ++g) {
            const T* a_ptr = mat_a + g * mat_a_group_stride;
            const T* b_ptr = mat_b + g * mat_b_group_stride;
            T* out_ptr = out + g * out_group_stride;

            // Zero the output for this group
            std::fill(out_ptr, out_ptr + out_group_stride, T{0});

            details::simple_matmul(a_ptr, b_ptr, out_ptr, M, K, N);
        }
        return;
    }

    // Case 1: 2D × 3D (MoE forward pass)
    if (a_ndim == 2 && b_ndim == 3) {
        const size_t total_rows = mat_a_shape[0];
        const size_t K = mat_a_shape[1];
        const size_t G = mat_b_shape[0];
        const size_t N = mat_b_shape[2];

        const size_t mat_b_group_stride = K * N;

        // Zero the output
        std::fill(out, out + total_rows * N, T{0});

        size_t start = 0;
        for (size_t g = 0; g < G; ++g) {
            const size_t end = static_cast<size_t>(offsets[g]);
            const size_t num_rows = end - start;

            if (num_rows > 0) {
                const T* a_ptr = mat_a + start * K;
                const T* b_ptr = mat_b + g * mat_b_group_stride;
                T* out_ptr = out + start * N;

                details::simple_matmul(a_ptr, b_ptr, out_ptr, num_rows, K, N);
            }
            start = end;
        }
        return;
    }

    // Case 3: 2D × 2D (MoE weight gradient)
    if (a_ndim == 2 && b_ndim == 2) {
        const size_t K = mat_a_shape[0];
        const size_t total_tokens = mat_a_shape[1];
        const size_t N = mat_b_shape[1];

        const size_t out_group_stride = K * N;

        size_t start = 0;
        for (size_t g = 0; g < num_groups; ++g) {
            const size_t end = static_cast<size_t>(offsets[g]);
            const size_t num_tokens = end - start;

            T* out_ptr = out + g * out_group_stride;

            // Zero the output for this group
            std::fill(out_ptr, out_ptr + out_group_stride, T{0});

            if (num_tokens > 0) {
                // mat_a[:, start:end] @ mat_b[start:end, :]
                // mat_a is (K, total_tokens), so slice along columns
                // mat_b is (total_tokens, N), so slice along rows
                for (size_t row = 0; row < K; ++row) {
                    for (size_t t = 0; t < num_tokens; ++t) {
                        const T a_val = mat_a[row * total_tokens + (start + t)];
                        for (size_t col = 0; col < N; ++col) {
                            out_ptr[row * N + col] += a_val * mat_b[(start + t) * N + col];
                        }
                    }
                }
            }
            start = end;
        }
        return;
    }
}

}  // namespace ov::reference
