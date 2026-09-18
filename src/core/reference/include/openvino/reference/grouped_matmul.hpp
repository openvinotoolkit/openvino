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
namespace func {

/// @brief Simple 2D matmul with transposed B: out = A @ B^T
/// @param A Input matrix A of shape (M, K)
/// @param B Input matrix B of shape (N, K)  -- stored as [N, K], i.e. B transposed
/// @param out Output matrix of shape (M, N) - must be pre-zeroed
/// @param M Number of rows in A
/// @param N Number of rows in B (= columns in the logical B^T)
/// @param K Shared dimension
template <typename T>
void simple_matmul_transposed_b(const T* A, const T* B, T* out, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < K; ++k) {
                out[i * N + j] += A[i * K + k] * B[j * K + k];
            }
        }
    }
}

}  // namespace func

/// @brief 3D × 3D batched uniform groups (no offsets).
template <typename T>
void grouped_matmul_3d_3d(const T* mat_a, const T* mat_b, T* out, const Shape& mat_a_shape, const Shape& mat_b_shape) {
    const size_t G = mat_a_shape[0];
    const size_t M = mat_a_shape[1];
    const size_t K = mat_a_shape[2];
    const size_t N = mat_b_shape[1];  // mat_b is [G, N, K]

    const size_t mat_a_group_stride = M * K;
    const size_t mat_b_group_stride = N * K;
    const size_t out_group_stride = M * N;

    for (size_t g = 0; g < G; ++g) {
        const T* a_ptr = mat_a + g * mat_a_group_stride;
        const T* b_ptr = mat_b + g * mat_b_group_stride;
        T* out_ptr = out + g * out_group_stride;

        std::fill(out_ptr, out_ptr + out_group_stride, T{0});
        func::simple_matmul_transposed_b(a_ptr, b_ptr, out_ptr, M, N, K);
    }
}

/// @brief 2D × 3D MoE forward pass with offsets.
template <typename T, typename TIdx>
void grouped_matmul_2d_3d(const T* mat_a,
                          const T* mat_b,
                          const TIdx* offsets,
                          T* out,
                          const Shape& mat_a_shape,
                          const Shape& mat_b_shape) {
    const size_t total_rows = mat_a_shape[0];
    const size_t K = mat_a_shape[1];
    const size_t G = mat_b_shape[0];
    const size_t N = mat_b_shape[1];  // mat_b is [G, N, K]

    const size_t mat_b_group_stride = N * K;

    std::fill(out, out + total_rows * N, T{0});

    size_t start = 0;
    for (size_t g = 0; g < G; ++g) {
        const size_t end = static_cast<size_t>(offsets[g]);
        const size_t num_rows = end - start;

        if (num_rows > 0) {
            const T* a_ptr = mat_a + start * K;
            const T* b_ptr = mat_b + g * mat_b_group_stride;
            T* out_ptr = out + start * N;
            func::simple_matmul_transposed_b(a_ptr, b_ptr, out_ptr, num_rows, N, K);
        }
        start = end;
    }
}

/// @brief Reference kernel for GroupedMatMul computation.
///
/// Supports two input combinations:
/// - 2D × 3D: MoE forward pass with offsets
/// - 3D × 3D: Batched uniform groups, no offsets
///
/// @tparam T Data type of input and output tensors.
/// @tparam TIdx Data type for offset indices.
///
/// @param mat_a Pointer to first input tensor.
/// @param mat_b Pointer to second input tensor.
/// @param offsets Pointer to offsets tensor (nullptr for 3D×3D case).
/// @param out Pointer to output tensor (pre-allocated).
/// @param mat_a_shape Shape of mat_a.
/// @param mat_b_shape Shape of mat_b.
/// @param out_shape Shape of output.
/// @param num_groups Number of groups (inferred from offsets or mat_b).
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

    if (a_ndim == 3 && b_ndim == 3) {
        grouped_matmul_3d_3d(mat_a, mat_b, out, mat_a_shape, mat_b_shape);
    } else if (a_ndim == 2 && b_ndim == 3) {
        grouped_matmul_2d_3d(mat_a, mat_b, offsets, out, mat_a_shape, mat_b_shape);
    }
}

}  // namespace ov::reference
