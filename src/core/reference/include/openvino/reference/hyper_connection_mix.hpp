// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ov::reference {

/// Reference for ov::op::internal::HyperConnectionMix. mixes [rows, (2 + n) * n], scale [3],
/// base [(2 + n) * n]; pre/post [rows, n], comb [rows, n, n] as [src][dst]. Computes in float.
template <typename T>
void hyper_connection_mix(const T* mixes,
                          const T* scale,
                          const T* base,
                          T* pre,
                          T* post,
                          T* comb,
                          size_t rows,
                          size_t n,
                          size_t sinkhorn_iterations,
                          float eps) {
    const size_t mix = (2 + n) * n;
    const float s0 = static_cast<float>(scale[0]);
    const float s1 = static_cast<float>(scale[1]);
    const float s2 = static_cast<float>(scale[2]);
    auto sigmoid = [](float x) {
        return 1.0f / (1.0f + std::exp(-x));
    };
    std::vector<float> c(n * n);
    for (size_t r = 0; r < rows; r++) {
        const T* m = mixes + r * mix;
        for (size_t i = 0; i < n; i++) {
            pre[r * n + i] = static_cast<T>(sigmoid(static_cast<float>(m[i]) * s0 + static_cast<float>(base[i])) + eps);
            post[r * n + i] =
                static_cast<T>(2.0f * sigmoid(static_cast<float>(m[n + i]) * s1 + static_cast<float>(base[n + i])));
        }
        for (size_t k = 0; k < n * n; k++)
            c[k] = static_cast<float>(m[2 * n + k]) * s2 + static_cast<float>(base[2 * n + k]);
        // c[src * n + dst]: softmax over dst for each src, then + eps
        for (size_t s = 0; s < n; s++) {
            float mx = c[s * n];
            for (size_t d = 1; d < n; d++)
                mx = std::max(mx, c[s * n + d]);
            float sum = 0.0f;
            for (size_t d = 0; d < n; d++) {
                c[s * n + d] = std::exp(c[s * n + d] - mx);
                sum += c[s * n + d];
            }
            for (size_t d = 0; d < n; d++)
                c[s * n + d] = c[s * n + d] / sum + eps;
        }
        auto norm_cols = [&]() {  // divide by the sum over src, per dst
            for (size_t d = 0; d < n; d++) {
                float sum = 0.0f;
                for (size_t s = 0; s < n; s++)
                    sum += c[s * n + d];
                sum += eps;
                for (size_t s = 0; s < n; s++)
                    c[s * n + d] /= sum;
            }
        };
        auto norm_rows = [&]() {  // divide by the sum over dst, per src
            for (size_t s = 0; s < n; s++) {
                float sum = 0.0f;
                for (size_t d = 0; d < n; d++)
                    sum += c[s * n + d];
                sum += eps;
                for (size_t d = 0; d < n; d++)
                    c[s * n + d] /= sum;
            }
        };
        norm_cols();
        for (size_t it = 1; it < sinkhorn_iterations; it++) {
            norm_rows();
            norm_cols();
        }
        for (size_t k = 0; k < n * n; k++)
            comb[r * n * n + k] = static_cast<T>(c[k]);
    }
}

}  // namespace ov::reference
