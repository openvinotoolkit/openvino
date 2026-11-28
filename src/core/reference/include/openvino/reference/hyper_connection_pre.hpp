// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "openvino/reference/hyper_connection_mix.hpp"

namespace ov::reference {

/// Pre side of one manifold-constrained hyper-connection site, per row of n streams of width e:
///   x = flatten(streams)                       [n * e]
///   x = x / sqrt(mean(x^2) + rms_eps)          (RMS norm without a gain)
///   mixes = fn * x                             [(2 + n) * n], fn laid out [(2 + n) * n][n * e]
///   pre, post, comb = hyper_connection_mix(mixes, scale, base)
///   y = sum_i pre[i] * streams[i]              [e]
/// Accumulates in float.
template <typename T, typename W>
void hyper_connection_pre(const T* streams,
                          const W* fn,
                          const T* scale,
                          const T* base,
                          T* y,
                          T* post,
                          T* comb,
                          size_t rows,
                          size_t n,
                          size_t e,
                          size_t sinkhorn_iterations,
                          float eps,
                          float rms_eps) {
    const size_t m = (2 + n) * n, k = n * e;
    std::vector<float> xn(k), scale_f(3), base_f(m), mixes(m), pre(n), post_f(n), comb_f(n * n);
    for (size_t i = 0; i < 3; i++)
        scale_f[i] = static_cast<float>(scale[i]);
    for (size_t i = 0; i < m; i++)
        base_f[i] = static_cast<float>(base[i]);
    for (size_t r = 0; r < rows; r++) {
        const T* x = streams + r * k;
        double ss = 0.0;
        for (size_t j = 0; j < k; j++) {
            const double v = static_cast<double>(x[j]);
            ss += v * v;
        }
        const float inv = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(k) + rms_eps));
        for (size_t j = 0; j < k; j++)
            xn[j] = static_cast<float>(x[j]) * inv;
        for (size_t o = 0; o < m; o++) {
            const W* row = fn + o * k;
            float acc = 0.f;
            for (size_t j = 0; j < k; j++)
                acc += static_cast<float>(row[j]) * xn[j];
            mixes[o] = acc;
        }
        hyper_connection_mix<float>(mixes.data(), scale_f.data(), base_f.data(), pre.data(), post_f.data(),
                                    comb_f.data(), 1, n, sinkhorn_iterations, eps);
        for (size_t c = 0; c < e; c++) {
            float acc = 0.f;
            for (size_t i = 0; i < n; i++)
                acc += pre[i] * static_cast<float>(x[i * e + c]);
            y[r * e + c] = static_cast<T>(acc);
        }
        for (size_t i = 0; i < n; i++)
            post[r * n + i] = static_cast<T>(post_f[i]);
        for (size_t i = 0; i < n * n; i++)
            comb[r * n * n + i] = static_cast<T>(comb_f[i]);
    }
}

}  // namespace ov::reference
