// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstdlib>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/convert.hpp"

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
                      const size_t n_squared);

void lu_solve(std::vector<float>& output,
              std::vector<float>& L,
              std::vector<float>& U,
              std::vector<size_t>& P,
              const size_t b,
              const size_t n,
              const size_t n_squared);
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

    for (size_t i = 0; i < shape.size() - 2; ++i) {
        batch_size = batch_size * shape[i];
    }

    auto input_conv = std::vector<float>(batch_size * n_squared);
    auto output_conv = std::vector<float>(batch_size * n_squared);

    convert<T, float>(input, input_conv.data(), batch_size * n_squared);

    std::vector<float> L(n_squared);
    std::vector<float> U(n_squared);
    std::vector<size_t> P(n);

    for (size_t b = 0; b < batch_size; ++b) {
        internal::lu_decomposition(input_conv, L, U, P, adjoint, b, n, n_squared);
        internal::lu_solve(output_conv, L, U, P, b, n, n_squared);
    }

    convert<float, T>(output_conv.data(), output, batch_size * n_squared);
}
}  // namespace reference
}  // namespace ov
