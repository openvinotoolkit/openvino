// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <openvino/core/except.hpp>

namespace ov::intel_gpu::cm::turboquant {

constexpr size_t kTurboQuantBits = 4;
constexpr size_t kTurboQuantCentroidsCount = static_cast<size_t>(1u << kTurboQuantBits);
constexpr size_t kTurboQuantBoundariesCount = kTurboQuantCentroidsCount - 1;
constexpr uint64_t kTurboSeedRotation = 42;
constexpr uint64_t kTurboSeedQJL = 1042;

// 4-bit TurboQuant fallback static codebook/boundaries.
// Runtime path uses get_turbo_codebook_tables(d) for per-head-size Lloyd-Max tables,
// and keeps these constants as a deterministic fallback/reference.
constexpr std::array<float, kTurboQuantCentroidsCount> kTurboQuantCodebook4Bit = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f,
};

constexpr std::array<float, kTurboQuantBoundariesCount> kTurboQuantBoundaries4Bit = {
    (-2.7326f + -2.0690f) / 2.0f,  // -2.4008
    (-2.0690f + -1.6180f) / 2.0f,  // -1.8435
    (-1.6180f + -1.2562f) / 2.0f,  // -1.4371
    (-1.2562f + -0.9424f) / 2.0f,  // -1.0993
    (-0.9424f + -0.6568f) / 2.0f,  // -0.7996
    (-0.6568f + -0.3881f) / 2.0f,  // -0.5225 (approx)
    (-0.3881f + -0.1284f) / 2.0f,  // -0.2583 (approx)
    (-0.1284f +  0.1284f) / 2.0f,  //  0.0000
    ( 0.1284f +  0.3881f) / 2.0f,  //  0.2583 (approx)
    ( 0.3881f +  0.6568f) / 2.0f,  //  0.5225 (approx)
    ( 0.6568f +  0.9424f) / 2.0f,  //  0.7996
    ( 0.9424f +  1.2562f) / 2.0f,  //  1.0993
    ( 1.2562f +  1.6180f) / 2.0f,  //  1.4371
    ( 1.6180f +  2.0690f) / 2.0f,  //  1.8435
    ( 2.0690f +  2.7326f) / 2.0f,  //  2.4008
};

struct TurboMatrices {
    std::vector<float> q;    // d x d
    std::vector<float> q_t;  // d x d
    std::vector<float> qjl;  // d x d (reserved for future QJL residual path)
};

struct TurboCodebookTables {
    std::vector<float> centroids;
    std::vector<float> boundaries;
};

inline std::vector<float> build_lloyd_max_codebook(size_t d,
                                                   size_t bits = kTurboQuantBits,
                                                   size_t n_iter = 300,
                                                   size_t grid_size = 50000) {
    OPENVINO_ASSERT(bits > 0 && bits <= 8, "Unsupported TurboQuant bits: ", bits);
    OPENVINO_ASSERT(grid_size >= 2, "grid_size must be >= 2, got ", grid_size);

    const size_t n_levels = static_cast<size_t>(1u << bits);

    const float sigma = d > 1 ? 1.0f / std::sqrt(static_cast<float>(d)) : 0.5f;
    const float lo = std::max(-1.0f + 1e-7f, -6.0f * sigma);
    const float hi = std::min(1.0f - 1e-7f,  6.0f * sigma);

    std::vector<float> grid(grid_size, 0.0f);
    const float step = (hi - lo) / static_cast<float>(grid_size - 1);
    for (size_t i = 0; i < grid_size; ++i) {
        grid[i] = lo + static_cast<float>(i) * step;
    }

    std::vector<float> pdf(grid_size, 0.0f);
    const float alpha = (static_cast<float>(d) - 1.0f) / 2.0f;
    const float alpha_m1 = alpha - 1.0f;
    double pdf_sum = 0.0;
    for (size_t i = 0; i < grid_size; ++i) {
        const float x = grid[i];
        const float one_minus_x2 = std::max(1.0f - x * x, 1e-30f);
        const float p = std::exp(alpha_m1 * std::log(one_minus_x2));
        pdf[i] = p;
        pdf_sum += static_cast<double>(p);
    }

    if (pdf_sum <= std::numeric_limits<double>::epsilon()) {
        return std::vector<float>(kTurboQuantCodebook4Bit.begin(), kTurboQuantCodebook4Bit.end());
    }

    const float inv_pdf_sum = static_cast<float>(1.0 / pdf_sum);
    for (auto& v : pdf)
        v *= inv_pdf_sum;

    std::vector<float> cdf(grid_size, 0.0f);
    float running = 0.0f;
    for (size_t i = 0; i < grid_size; ++i) {
        running += pdf[i];
        cdf[i] = running;
    }
    const float cdf_last = std::max(cdf.back(), 1e-12f);
    for (auto& v : cdf)
        v /= cdf_last;

    std::vector<float> centroids(n_levels, 0.0f);
    for (size_t i = 0; i < n_levels; ++i) {
        const float target = (static_cast<float>(i) + 0.5f) / static_cast<float>(n_levels);
        auto it = std::lower_bound(cdf.begin(), cdf.end(), target);
        const size_t idx = (it == cdf.end()) ? (grid_size - 1) : static_cast<size_t>(std::distance(cdf.begin(), it));
        centroids[i] = grid[idx];
    }

    std::vector<size_t> assignments(grid_size, 0);
    std::vector<double> weighted_sum(n_levels, 0.0);
    std::vector<double> weight_sum(n_levels, 0.0);

    for (size_t iter = 0; iter < n_iter; ++iter) {
        std::fill(weighted_sum.begin(), weighted_sum.end(), 0.0);
        std::fill(weight_sum.begin(), weight_sum.end(), 0.0);

        for (size_t gi = 0; gi < grid_size; ++gi) {
            const float x = grid[gi];
            size_t best_idx = 0;
            float best_dist = std::abs(x - centroids[0]);
            for (size_t c = 1; c < n_levels; ++c) {
                const float dist = std::abs(x - centroids[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = c;
                }
            }

            assignments[gi] = best_idx;
            const double w = static_cast<double>(pdf[gi]);
            weighted_sum[best_idx] += static_cast<double>(x) * w;
            weight_sum[best_idx] += w;
        }

        for (size_t c = 0; c < n_levels; ++c) {
            if (weight_sum[c] > std::numeric_limits<double>::epsilon()) {
                centroids[c] = static_cast<float>(weighted_sum[c] / weight_sum[c]);
            }
        }
    }

    std::sort(centroids.begin(), centroids.end());
    return centroids;
}

inline TurboCodebookTables build_lloyd_max_tables(size_t d,
                                                   size_t bits = kTurboQuantBits,
                                                   size_t n_iter = 300,
                                                   size_t grid_size = 50000) {
    auto centroids = build_lloyd_max_codebook(d, bits, n_iter, grid_size);
    OPENVINO_ASSERT(!centroids.empty(), "Lloyd-Max centroids cannot be empty");

    std::vector<float> boundaries;
    boundaries.reserve(centroids.size() - 1);
    for (size_t i = 0; i + 1 < centroids.size(); ++i) {
        boundaries.push_back(0.5f * (centroids[i] + centroids[i + 1]));
    }

    return {std::move(centroids), std::move(boundaries)};
}

inline double lcg_normal(uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = static_cast<double>(state >> 11) / static_cast<double>(1ULL << 53);
    if (u1 < 1e-15)
        u1 = 1e-15;

    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u2 = static_cast<double>(state >> 11) / static_cast<double>(1ULL << 53);

    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
}

inline void qr_orthonormalize(std::vector<float>& q, size_t d) {
    std::vector<float> r(d * d, 0.0f);

    for (size_t col = 0; col < d; ++col) {
        for (size_t prev_col = 0; prev_col < col; ++prev_col) {
            float dot = 0.0f;
            for (size_t row = 0; row < d; ++row) {
                dot += q[row * d + prev_col] * q[row * d + col];
            }
            r[prev_col * d + col] = dot;
            for (size_t row = 0; row < d; ++row) {
                q[row * d + col] -= dot * q[row * d + prev_col];
            }
        }

        float norm = 0.0f;
        for (size_t row = 0; row < d; ++row) {
            const float value = q[row * d + col];
            norm += value * value;
        }
        norm = std::sqrt(norm);

        if (norm <= 1e-12f) {
            for (size_t row = 0; row < d; ++row) {
                q[row * d + col] = (row == col) ? 1.0f : 0.0f;
            }
            r[col * d + col] = 1.0f;
            continue;
        }

        r[col * d + col] = norm;
        const float inv_norm = 1.0f / norm;
        for (size_t row = 0; row < d; ++row) {
            q[row * d + col] *= inv_norm;
        }
    }

    // Match TurboQuantMSE.__init__ sign disambiguation:
    //   Q = Q * sign(diag(R)).unsqueeze(0)
    for (size_t col = 0; col < d; ++col) {
        const float sign = r[col * d + col] < 0.0f ? -1.0f : 1.0f;
        for (size_t row = 0; row < d; ++row) {
            q[row * d + col] *= sign;
        }
    }
}

inline const TurboMatrices& get_turbo_matrices(size_t d) {
    static std::mutex m;
    static std::unordered_map<size_t, TurboMatrices> cache;

    std::lock_guard<std::mutex> lock(m);
    auto it = cache.find(d);
    if (it != cache.end())
        return it->second;

    TurboMatrices mats;
    mats.q.resize(d * d);
    mats.q_t.resize(d * d);
    mats.qjl.resize(d * d);

    // Deterministic Gaussian -> QR decomposition -> orthonormal Q.
    uint64_t state = kTurboSeedRotation;
    for (size_t i = 0; i < d * d; i++) {
        mats.q[i] = static_cast<float>(lcg_normal(state));
    }
    qr_orthonormalize(mats.q, d);

    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            mats.q_t[i * d + j] = mats.q[j * d + i];
        }
    }

    // Deterministic Gaussian QJL matrix, kept for algorithm parity with OCL SDPA implementation.
    state = kTurboSeedQJL;
    for (size_t i = 0; i < d * d; i++) {
        mats.qjl[i] = static_cast<float>(lcg_normal(state));
    }

    auto [inserted_it, _] = cache.emplace(d, std::move(mats));
    return inserted_it->second;
}

inline const TurboCodebookTables& get_turbo_codebook_tables(size_t d) {
    static std::unordered_map<size_t, TurboCodebookTables> table_cache;
    static std::mutex table_mtx;

    std::lock_guard<std::mutex> lock(table_mtx);
    auto it = table_cache.find(d);
    if (it == table_cache.end()) {
        TurboCodebookTables tables = build_lloyd_max_tables(d, kTurboQuantBits);

        if (tables.centroids.size() != kTurboQuantCentroidsCount ||
            tables.boundaries.size() != kTurboQuantBoundariesCount) {
            tables.centroids.assign(kTurboQuantCodebook4Bit.begin(), kTurboQuantCodebook4Bit.end());
            tables.boundaries.assign(kTurboQuantBoundaries4Bit.begin(), kTurboQuantBoundaries4Bit.end());
        }

        it = table_cache.emplace(d, std::move(tables)).first;
    }
    return it->second;
}

}  // namespace ov::intel_gpu::cm::turboquant
