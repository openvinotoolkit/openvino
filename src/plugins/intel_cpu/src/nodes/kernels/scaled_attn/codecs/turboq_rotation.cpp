// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "turboq_rotation.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace ov::Extensions::Cpu {

static constexpr uint64_t TURBOQ_SEED = 0x517cc1b727220a95ULL;
static constexpr uint64_t TURBOQ_QJL_SEED = TURBOQ_SEED ^ 0x1234567890abcdefULL;
static constexpr uint64_t TURBOQ_WHT_SEED = TURBOQ_SEED ^ 0xfedcba9876543210ULL;
static constexpr uint64_t TURBOQ_QJL_WHT_SEED = TURBOQ_SEED ^ 0xa5a5a5a5a5a5a5a5ULL;

// ---------------------------------------------------------------------------
// Householder QR decomposition of a random Gaussian matrix.
// Produces an orthogonal Q that is Haar-distributed (uniform over O(n)).
// ---------------------------------------------------------------------------
static void householder_qr(float* A, float* Q, int n) {
    std::memset(Q, 0, n * n * sizeof(float));
    for (int i = 0; i < n; i++) {
        Q[i * n + i] = 1.0F;
    }

    for (int k = 0; k < n; k++) {
        float norm_sq = 0.0F;
        for (int i = k; i < n; i++) {
            float val = A[i * n + k];
            norm_sq += val * val;
        }
        float norm_x = std::sqrt(norm_sq);
        if (norm_x < 1e-30F) {
            continue;
        }

        float sign = (A[k * n + k] >= 0.0F) ? 1.0F : -1.0F;
        A[k * n + k] += sign * norm_x;

        float v_norm_sq = 0.0F;
        for (int i = k; i < n; i++) {
            float val = A[i * n + k];
            v_norm_sq += val * val;
        }
        float inv_v_norm = 1.0F / std::sqrt(v_norm_sq);
        for (int i = k; i < n; i++) {
            A[i * n + k] *= inv_v_norm;
        }

        for (int j = k + 1; j < n; j++) {
            float dot = 0.0F;
            for (int i = k; i < n; i++) {
                dot += A[i * n + k] * A[i * n + j];
            }
            dot *= 2.0F;
            for (int i = k; i < n; i++) {
                A[i * n + j] -= dot * A[i * n + k];
            }
        }

        for (int i = 0; i < n; i++) {
            float dot = 0.0F;
            for (int j = k; j < n; j++) {
                dot += Q[i * n + j] * A[j * n + k];
            }
            dot *= 2.0F;
            for (int j = k; j < n; j++) {
                Q[i * n + j] -= dot * A[j * n + k];
            }
        }
    }
}

// Transpose an n×n row-major matrix: dst[i][j] = src[j][i].
static void transpose(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[i * n + j] = src[j * n + i];
        }
    }
}

// Multiply two n×n row-major matrices: dst = A * B.
static void matmul(float* dst, const float* A, const float* B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0F;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            dst[i * n + j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-dimension matrix cache. All matrices for a given dim are generated
// together on first access and stored permanently.
// ---------------------------------------------------------------------------

struct RotationCache {
    std::vector<float> Q;              // Rotation matrix (Haar orthogonal)
    std::vector<float> QT;             // Q transposed
    std::vector<float> S;              // QJL projection (raw Gaussian)
    std::vector<float> ST;             // S transposed
    std::vector<float> SQ;             // S * Q (precomputed for fused Q projection)
    std::vector<float> wht_signs;      // Random ±1 signs for MSE WHT rotation
    std::vector<float> qjl_wht_signs;  // Independent ±1 signs for QJL WHT projection
};

static std::mutex g_cache_mutex;              // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::map<int, RotationCache> g_cache;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

static const RotationCache& get_cache(int dim) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    auto [it, inserted] = g_cache.try_emplace(dim);
    if (!inserted) {
        return it->second;
    }

    auto& c = it->second;
    const int n2 = dim * dim;

    // Generate Q via Householder QR of a Gaussian matrix.
    c.Q.resize(n2);
    c.QT.resize(n2);
    {
        std::mt19937_64 rng(TURBOQ_SEED);
        std::normal_distribution<float> dist(0.0F, 1.0F);
        std::vector<float> A(n2);
        for (int i = 0; i < n2; i++) {
            A[i] = dist(rng);
        }
        householder_qr(A.data(), c.Q.data(), dim);
        transpose(c.QT.data(), c.Q.data(), dim);
    }

    // Generate S — raw Gaussian (NOT orthogonalized).
    // Row norms are ~sqrt(dim); the QJL coefficient accounts for this.
    c.S.resize(n2);
    c.ST.resize(n2);
    {
        std::mt19937_64 rng(TURBOQ_QJL_SEED);
        std::normal_distribution<float> dist(0.0F, 1.0F);
        for (int i = 0; i < n2; i++) {
            c.S[i] = dist(rng);
        }
        transpose(c.ST.data(), c.S.data(), dim);
    }

    // Precompute SQ = S * Q.
    c.SQ.resize(n2);
    matmul(c.SQ.data(), c.S.data(), c.Q.data(), dim);

    // Generate random ±1 signs for WHT rotation.
    c.wht_signs.resize(dim);
    {
        std::mt19937_64 rng(TURBOQ_WHT_SEED);
        std::uniform_int_distribution<int> dist(0, 1);
        for (int i = 0; i < dim; i++) {
            c.wht_signs[i] = dist(rng) ? 1.0F : -1.0F;
        }
    }

    // Independent ±1 signs for QJL WHT projection (must differ from wht_signs).
    c.qjl_wht_signs.resize(dim);
    {
        std::mt19937_64 rng(TURBOQ_QJL_WHT_SEED);
        std::uniform_int_distribution<int> dist(0, 1);
        for (int i = 0; i < dim; i++) {
            c.qjl_wht_signs[i] = dist(rng) ? 1.0F : -1.0F;
        }
    }

    return c;
}

// ---------------------------------------------------------------------------
// Public API — all take dim, delegate to the cache.
// ---------------------------------------------------------------------------

const float* turboq_get_rotation_matrix(int dim) {
    return get_cache(dim).Q.data();
}

const float* turboq_get_rotation_matrix_t(int dim) {
    return get_cache(dim).QT.data();
}

const float* turboq_get_projection_matrix(int dim) {
    return get_cache(dim).S.data();
}

const float* turboq_get_projection_matrix_t(int dim) {
    return get_cache(dim).ST.data();
}

const float* turboq_get_SQ_matrix(int dim) {
    return get_cache(dim).SQ.data();
}

const float* turboq_get_wht_signs(int dim) {
    return get_cache(dim).wht_signs.data();
}

const float* turboq_get_qjl_wht_signs(int dim) {
    return get_cache(dim).qjl_wht_signs.data();
}

// Rotation mode is controlled at process start via env var OV_TURBOQ_ROTATION.
// Values: "wht" (default), "dense", "none".
// QJL projection uses the same env var (OV_TURBOQ_QJL_PROJECTION), defaulting to WHT.
static TurboqRotationMode parse_rotation_mode_env(const char* var_name) {
    const char* v = std::getenv(var_name);
    if (v == nullptr) {
        return TurboqRotationMode::WHT;
    }
    std::string s(v);
    if (s == "dense" || s == "DENSE") {
        return TurboqRotationMode::DENSE;
    }
    if (s == "none" || s == "NONE") {
        return TurboqRotationMode::NONE;
    }
    return TurboqRotationMode::WHT;
}

TurboqRotationMode turboq_get_rotation_mode() {
    static const TurboqRotationMode mode = parse_rotation_mode_env("OV_TURBOQ_ROTATION");
    return mode;
}

TurboqRotationMode turboq_get_qjl_projection_mode() {
    static const TurboqRotationMode mode = parse_rotation_mode_env("OV_TURBOQ_QJL_PROJECTION");
    return mode;
}

namespace XARCH {

void turboq_matvec_ref(const float* M, const float* x, float* y, int dim) {
    for (int i = 0; i < dim; i++) {
        float sum = 0.0F;
        for (int j = 0; j < dim; j++) {
            sum += M[i * dim + j] * x[j];
        }
        y[i] = sum;
    }
}

}  // namespace XARCH
}  // namespace ov::Extensions::Cpu
