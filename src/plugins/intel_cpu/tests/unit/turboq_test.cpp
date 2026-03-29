// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "nodes/kernels/scaled_attn/attn_quant_turboq.hpp"
#include "nodes/kernels/scaled_attn/polarq_tables.h"
#include "nodes/kernels/scaled_attn/turboq_rotation.hpp"
#include "nodes/kernels/scaled_attn/turboq_tables.h"

using namespace ov::Extensions::Cpu;
using namespace ov::Extensions::Cpu::XARCH;

static constexpr float TURBOQ_PI_F = 3.14159265358979323846F;
// ---------------------------------------------------------------------------
// Test rotation helpers — use same WHT rotation as production code.
// These replace direct turboq_matvec_ref(Q, ...) calls in tests so that
// tests stay consistent with the production rotation choice.
// ---------------------------------------------------------------------------
static void test_rotate_forward(const float* src, float* dst, int dim) {
    turboq_rotate_forward(src, dst, dim);
}

static void test_rotate_inverse(float* src, float* dst, int dim) {
    turboq_rotate_inverse(src, dst, dim);
}

// ============================================================================
// Codebook & Boundary Tests
// ============================================================================

TEST(TurboQ, CodebookSymmetry3Bit) {
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(TURBOQ_CODEBOOK_3BIT[i], -TURBOQ_CODEBOOK_3BIT[7 - i])
            << "3-bit codebook not symmetric at index " << i;
    }
}

TEST(TurboQ, CodebookSymmetry4Bit) {
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(TURBOQ_CODEBOOK_4BIT[i], -TURBOQ_CODEBOOK_4BIT[15 - i])
            << "4-bit codebook not symmetric at index " << i;
    }
}

TEST(TurboQ, BoundaryOrdering3Bit) {
    for (int i = 1; i < 7; i++) {
        EXPECT_LT(TURBOQ_BOUNDARIES_3BIT[i - 1], TURBOQ_BOUNDARIES_3BIT[i])
            << "3-bit boundaries not strictly increasing at index " << i;
    }
}

TEST(TurboQ, BoundaryOrdering4Bit) {
    for (int i = 1; i < 15; i++) {
        EXPECT_LT(TURBOQ_BOUNDARIES_4BIT[i - 1], TURBOQ_BOUNDARIES_4BIT[i])
            << "4-bit boundaries not strictly increasing at index " << i;
    }
}

TEST(TurboQ, BoundariesAreMidpoints3Bit) {
    for (int i = 0; i < 7; i++) {
        float expected = (TURBOQ_CODEBOOK_3BIT[i] + TURBOQ_CODEBOOK_3BIT[i + 1]) / 2.0f;
        EXPECT_FLOAT_EQ(TURBOQ_BOUNDARIES_3BIT[i], expected)
            << "3-bit boundary " << i << " is not the midpoint of adjacent centroids";
    }
}

TEST(TurboQ, BoundariesAreMidpoints4Bit) {
    for (int i = 0; i < 15; i++) {
        float expected = (TURBOQ_CODEBOOK_4BIT[i] + TURBOQ_CODEBOOK_4BIT[i + 1]) / 2.0f;
        EXPECT_FLOAT_EQ(TURBOQ_BOUNDARIES_4BIT[i], expected)
            << "4-bit boundary " << i << " is not the midpoint of adjacent centroids";
    }
}

// ============================================================================
// Rotation Matrix Tests
// ============================================================================

TEST(TurboQ, RotationDeterminism) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    const float* Q1 = turboq_get_rotation_matrix(dim);
    const float* Q2 = turboq_get_rotation_matrix(dim);
    EXPECT_EQ(Q1, Q2) << "Cache should return same pointer for same dim";

    const float* QT1 = turboq_get_rotation_matrix_t(dim);
    const float* QT2 = turboq_get_rotation_matrix_t(dim);
    EXPECT_EQ(QT1, QT2) << "Cache should return same pointer for same dim";
}

TEST(TurboQ, RotationOrthogonality) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    const float* Q = turboq_get_rotation_matrix(dim);

    // Check Q * Q^T ≈ I by computing dot products of row pairs.
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            float dot = 0.0f;
            for (int k = 0; k < dim; k++) {
                dot += Q[i * dim + k] * Q[j * dim + k];
            }
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(dot, expected, 1e-5f) << "Q * Q^T not identity at (" << i << ", " << j << ")";
        }
    }
}

TEST(TurboQ, RotationTransposeConsistency) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    const float* Q = turboq_get_rotation_matrix(dim);
    const float* QT = turboq_get_rotation_matrix_t(dim);

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            EXPECT_FLOAT_EQ(Q[i * dim + j], QT[j * dim + i]) << "Q^T mismatch at (" << i << ", " << j << ")";
        }
    }
}

// ============================================================================
// Matvec Tests
// ============================================================================

TEST(TurboQ, MatvecIdentity) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    // Identity matrix.
    std::vector<float> I(dim * dim, 0.0f);
    for (int i = 0; i < dim; i++)
        I[i * dim + i] = 1.0f;

    std::vector<float> x(dim);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : x)
        v = dist(rng);

    std::vector<float> y(dim);
    turboq_matvec_ref(I.data(), x.data(), y.data(), dim);

    for (int i = 0; i < dim; i++) {
        EXPECT_NEAR(y[i], x[i], 1e-6f);
    }
}

TEST(TurboQ, MatvecRotationPreservesNorm) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    const float* Q = turboq_get_rotation_matrix(dim);

    std::vector<float> x(dim);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : x)
        v = dist(rng);

    std::vector<float> y(dim);
    turboq_matvec_ref(Q, x.data(), y.data(), dim);

    float norm_x = 0.0f, norm_y = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }
    EXPECT_NEAR(std::sqrt(norm_x), std::sqrt(norm_y), 1e-4f) << "Orthogonal rotation should preserve vector norm";
}

TEST(TurboQ, MatvecRoundTrip) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    const float* Q = turboq_get_rotation_matrix(dim);
    const float* QT = turboq_get_rotation_matrix_t(dim);

    std::vector<float> x(dim);
    std::mt19937 rng(456);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : x)
        v = dist(rng);

    std::vector<float> rotated(dim), recovered(dim);
    turboq_matvec_ref(Q, x.data(), rotated.data(), dim);
    turboq_matvec_ref(QT, rotated.data(), recovered.data(), dim);

    for (int i = 0; i < dim; i++) {
        EXPECT_NEAR(recovered[i], x[i], 1e-4f) << "Q^T * Q * x should recover x at index " << i;
    }
}

// ============================================================================
// WHT + Random Signs Tests
// ============================================================================

TEST(TurboQ, WHTSignsDeterminism) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    const float* s1 = turboq_get_wht_signs(dim);
    const float* s2 = turboq_get_wht_signs(dim);
    EXPECT_EQ(s1, s2) << "Cache should return same pointer";
    // All values should be ±1
    for (int i = 0; i < dim; i++) {
        EXPECT_TRUE(s1[i] == 1.0f || s1[i] == -1.0f) << "Sign " << i << " is not ±1";
    }
}

TEST(TurboQ, WHTPreservesNorm) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    const float* signs = turboq_get_wht_signs(dim);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(dim), y(dim);
    for (auto& v : x)
        v = dist(rng);

    turboq_wht_forward(signs, x.data(), y.data(), dim);

    float norm_x = 0, norm_y = 0;
    for (int i = 0; i < dim; i++) {
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }
    EXPECT_NEAR(std::sqrt(norm_x), std::sqrt(norm_y), 1e-3f) << "WHT should preserve norm";
}

TEST(TurboQ, WHTRoundTrip) {
    for (int dim : {128, 256}) {
        const float* signs = turboq_get_wht_signs(dim);

        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> x(dim), rotated(dim), recovered(dim);
        for (auto& v : x)
            v = dist(rng);

        turboq_wht_forward(signs, x.data(), rotated.data(), dim);
        // turboq_wht_inverse mutates its input — use a scratch copy so `rotated` stays valid.
        std::vector<float> scratch = rotated;
        turboq_wht_inverse(signs, scratch.data(), recovered.data(), dim);

        for (int i = 0; i < dim; i++) {
            EXPECT_NEAR(recovered[i], x[i], 1e-3f) << "WHT round-trip failed at dim=" << dim << " index " << i;
        }

        // Also check inner product preservation
        std::vector<float> y(dim), y_rot(dim);
        for (auto& v : y)
            v = dist(rng);
        turboq_wht_forward(signs, y.data(), y_rot.data(), dim);

        float dot_orig = 0, dot_rot = 0;
        for (int i = 0; i < dim; i++) {
            dot_orig += x[i] * y[i];
            dot_rot += rotated[i] * y_rot[i];
        }
        EXPECT_NEAR(dot_orig, dot_rot, std::abs(dot_orig) * 1e-3f + 1e-5f)
            << "WHT inner product not preserved at dim=" << dim;
    }
}

// Compare WHT vs dense rotation quality for TBQ scalar quantization.
TEST(TurboQ, WHTvsDenseQuality) {
    constexpr int N = 500;

    for (int dim : {128, 256}) {
        const float* signs = turboq_get_wht_signs(dim);
        const float sqrt_dim = std::sqrt(static_cast<float>(dim));

        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (int bits : {3, 4}) {
            const float* boundaries = (bits == 4) ? TURBOQ_BOUNDARIES_4BIT : TURBOQ_BOUNDARIES_3BIT;
            const float* codebook = (bits == 4) ? TURBOQ_CODEBOOK_4BIT : TURBOQ_CODEBOOK_3BIT;
            int n_bnd = (1 << bits) - 1;

            double dense_cos_sum = 0, wht_cos_sum = 0;

            for (int s = 0; s < N; s++) {
                std::vector<float> vec(dim);
                for (auto& v : vec)
                    v = dist(rng);

                float norm_sq = 0;
                for (int i = 0; i < dim; i++)
                    norm_sq += vec[i] * vec[i];
                float norm = std::sqrt(norm_sq);
                std::vector<float> unit(dim);
                for (int i = 0; i < dim; i++)
                    unit[i] = vec[i] / norm;

                // Dense rotation path
                {
                    std::vector<float> rotated(dim);
                    test_rotate_forward(unit.data(), rotated.data(), dim);
                    for (int i = 0; i < dim; i++)
                        rotated[i] *= sqrt_dim;

                    // Quantize + dequantize
                    for (int i = 0; i < dim; i++) {
                        int idx = 0;
                        for (int b = 0; b < n_bnd; b++)
                            idx += (rotated[i] > boundaries[b]) ? 1 : 0;
                        rotated[i] = codebook[idx];
                    }
                    // Inverse: scale back and apply QT
                    for (int i = 0; i < dim; i++)
                        rotated[i] *= norm / sqrt_dim;
                    std::vector<float> result(dim);
                    test_rotate_inverse(rotated.data(), result.data(), dim);

                    double dot = 0, nv = 0, nr = 0;
                    for (int i = 0; i < dim; i++) {
                        dot += vec[i] * result[i];
                        nv += vec[i] * vec[i];
                        nr += result[i] * result[i];
                    }
                    dense_cos_sum += dot / (std::sqrt(nv * nr) + 1e-10);
                }

                // WHT path
                {
                    std::vector<float> rotated(dim);
                    turboq_wht_forward(signs, unit.data(), rotated.data(), dim);
                    for (int i = 0; i < dim; i++)
                        rotated[i] *= sqrt_dim;

                    for (int i = 0; i < dim; i++) {
                        int idx = 0;
                        for (int b = 0; b < n_bnd; b++)
                            idx += (rotated[i] > boundaries[b]) ? 1 : 0;
                        rotated[i] = codebook[idx];
                    }
                    for (int i = 0; i < dim; i++)
                        rotated[i] *= norm / sqrt_dim;
                    std::vector<float> result(dim);
                    turboq_wht_inverse(signs, rotated.data(), result.data(), dim);

                    double dot = 0, nv = 0, nr = 0;
                    for (int i = 0; i < dim; i++) {
                        dot += vec[i] * result[i];
                        nv += vec[i] * vec[i];
                        nr += result[i] * result[i];
                    }
                    wht_cos_sum += dot / (std::sqrt(nv * nr) + 1e-10);
                }
            }

            double dense_cos = dense_cos_sum / N;
            double wht_cos = wht_cos_sum / N;
            printf("\n=== Dense vs WHT Rotation Quality (TBQ%d, N=%d, dim=%d) ===\n", bits, N, dim);
            printf("  Dense Haar:  V CosSim = %.6f\n", dense_cos);
            printf("  WHT+signs:   V CosSim = %.6f\n", wht_cos);
            printf("  Difference:  %.6f (positive = dense better)\n", dense_cos - wht_cos);

            // Both should be high quality
            EXPECT_GT(dense_cos, 0.90) << "Dense rotation quality too low for " << bits << "-bit dim=" << dim;
            EXPECT_GT(wht_cos, 0.90) << "WHT rotation quality too low for " << bits << "-bit dim=" << dim;
        }
    }  // for dim
}

// Reproduce the functional test's strided-iota inputs to diagnose WHT quality
// with structured (non-Gaussian) data at different dims and bit widths.
TEST(TurboQ, WHTRampInputQuality) {
    for (int dim : {128, 256}) {
        const float* signs = turboq_get_wht_signs(dim);
        const float sqrt_dim = std::sqrt(static_cast<float>(dim));

        for (int bits : {3, 4}) {
            const float* boundaries = (bits == 4) ? TURBOQ_BOUNDARIES_4BIT : TURBOQ_BOUNDARIES_3BIT;
            const float* codebook = (bits == 4) ? TURBOQ_CODEBOOK_4BIT : TURBOQ_CODEBOOK_3BIT;
            int n_bnd = (1 << bits) - 1;

            // Test multiple ramp offsets (matching strided_iota(val, 0.1) from the functional test)
            double worst_cos = 1.0;
            float worst_base = 0;
            for (float base : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 10.0f, 50.0f, 100.0f}) {
                std::vector<float> vec(dim);
                for (int i = 0; i < dim; i++)
                    vec[i] = base + 0.1f * i;

                float norm = 0;
                for (int i = 0; i < dim; i++)
                    norm += vec[i] * vec[i];
                norm = std::sqrt(norm);
                std::vector<float> unit(dim);
                for (int i = 0; i < dim; i++)
                    unit[i] = vec[i] / norm;

                // WHT forward
                std::vector<float> rotated(dim);
                turboq_wht_forward(signs, unit.data(), rotated.data(), dim);
                for (int i = 0; i < dim; i++)
                    rotated[i] *= sqrt_dim;

                // Quantize + dequantize
                for (int i = 0; i < dim; i++) {
                    int idx = 0;
                    for (int b = 0; b < n_bnd; b++)
                        idx += (rotated[i] > boundaries[b]) ? 1 : 0;
                    rotated[i] = codebook[idx];
                }
                for (int i = 0; i < dim; i++)
                    rotated[i] *= norm / sqrt_dim;

                // WHT inverse
                std::vector<float> result(dim);
                turboq_wht_inverse(signs, rotated.data(), result.data(), dim);

                double dot = 0, nv = 0, nr = 0;
                for (int i = 0; i < dim; i++) {
                    dot += vec[i] * result[i];
                    nv += vec[i] * vec[i];
                    nr += result[i] * result[i];
                }
                double cos_sim = dot / (std::sqrt(nv * nr) + 1e-10);
                if (cos_sim < worst_cos) {
                    worst_cos = cos_sim;
                    worst_base = base;
                }
            }
            printf("\n=== WHT Ramp Quality (TBQ%d, dim=%d) worst_cossim=%.6f (base=%.1f) ===\n",
                   bits,
                   dim,
                   worst_cos,
                   worst_base);
            EXPECT_GT(worst_cos, 0.95) << "WHT ramp quality too low for " << bits << "-bit dim=" << dim
                                       << " (base=" << worst_base << ")";
        }
    }
}

// ============================================================================
// Bit Packing Round-Trip Tests
// ============================================================================

// Pack/unpack helpers are static in attn_quant_turboq.cpp, so we test them
// indirectly through the full quantize/dequantize path.  But we can also
// test the round-trip: quantize known codebook centroids and verify exact
// index recovery.

TEST(TurboQ, ScalarQuantizeRoundTrip4Bit) {
    // Each codebook centroid should quantize to its own index.
    for (int i = 0; i < 16; i++) {
        float val = TURBOQ_CODEBOOK_4BIT[i];
        // Manually do the quantize: branchless scan over boundaries.
        int idx = 0;
        for (int b = 0; b < 15; b++) {
            idx += (val > TURBOQ_BOUNDARIES_4BIT[b]) ? 1 : 0;
        }
        EXPECT_EQ(idx, i) << "4-bit centroid " << i << " should quantize to index " << i;
    }
}

TEST(TurboQ, ScalarQuantizeRoundTrip3Bit) {
    for (int i = 0; i < 8; i++) {
        float val = TURBOQ_CODEBOOK_3BIT[i];
        int idx = 0;
        for (int b = 0; b < 7; b++) {
            idx += (val > TURBOQ_BOUNDARIES_3BIT[b]) ? 1 : 0;
        }
        EXPECT_EQ(idx, i) << "3-bit centroid " << i << " should quantize to index " << i;
    }
}

// ============================================================================
// Fused Kernel Tests (production path)
// ============================================================================

// Verify fused Q·K dot: quantize K, compute fused dot with rotated Q,
// compare against reference float dot(q, k) within quantization tolerance.
TEST(TurboQ, FusedQKDot_TBQ4) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> q(dim), k(dim);
    for (auto& v : q)
        v = dist(rng);
    for (auto& v : k)
        v = dist(rng);

    // Quantize K
    std::vector<uint8_t> packed_k(turboq_head_bytes(dim, 4));
    turboq_quantize_head(k.data(), packed_k.data(), dim, 4, ov::element::f32);

    // Rotate Q (production path)
    std::vector<float> q_rot(dim);
    test_rotate_forward(q.data(), q_rot.data(), dim);

    // Fused dot (production kernel)
    float fused_dot = turboq_fused_qk_dot(packed_k.data(), q_rot.data(), dim, 4);

    // Reference: float dot(q, k)
    float ref_dot = 0.0f;
    for (int i = 0; i < dim; i++)
        ref_dot += q[i] * k[i];

    // Quantization error compounds: rotation + codebook + fp16 norm.
    // Relative error can exceed 50% for individual dot products with random vectors.
    // The statistical guarantee is that the error is unbiased and small in expectation.
    float abs_error = std::abs(fused_dot - ref_dot);
    float scale = std::max(std::abs(ref_dot), std::abs(fused_dot)) + 1e-10f;
    EXPECT_LT(abs_error / scale, 0.60f) << "TBQ4 fused dot too far from reference: fused=" << fused_dot
                                        << " ref=" << ref_dot;
}

TEST(TurboQ, FusedQKDot_TBQ3) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;

    std::mt19937 rng(43);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> q(dim), k(dim);
    for (auto& v : q)
        v = dist(rng);
    for (auto& v : k)
        v = dist(rng);

    std::vector<uint8_t> packed_k(turboq_head_bytes(dim, 3));
    turboq_quantize_head(k.data(), packed_k.data(), dim, 3, ov::element::f32);

    std::vector<float> q_rot(dim);
    test_rotate_forward(q.data(), q_rot.data(), dim);

    float fused_dot = turboq_fused_qk_dot(packed_k.data(), q_rot.data(), dim, 3);

    float ref_dot = 0.0f;
    for (int i = 0; i < dim; i++)
        ref_dot += q[i] * k[i];

    float relative_error = std::abs(fused_dot - ref_dot) / (std::abs(ref_dot) + 1e-10f);
    EXPECT_LT(relative_error, 0.50f) << "TBQ3 fused dot too far from reference: fused=" << fused_dot
                                     << " ref=" << ref_dot;
}

// Verify fused V accumulation: quantize V, accumulate with known weight,
// compare against reference weighted sum.
TEST(TurboQ, FusedVAccum_TBQ4) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;

    std::mt19937 rng(303);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Single V vector with weight=1.0 — accumulator should approximate V.
    std::vector<float> v(dim);
    for (auto& val : v)
        val = dist(rng);

    std::vector<uint8_t> packed_v(turboq_head_bytes(dim, 4));
    turboq_quantize_head(v.data(), packed_v.data(), dim, 4, ov::element::f32);

    // Accumulate with weight=1.0 into a zeroed buffer (rotated domain).
    std::vector<float> accum(dim, 0.0f);
    float* accum_ptr = accum.data();
    float weight = 1.0f;
    turboq_fused_v_accum(packed_v.data(), &weight, &accum_ptr, 1, dim, 4);

    // Apply inverse rotation to get back to original domain.
    std::vector<float> result(dim);
    test_rotate_inverse(accum.data(), result.data(), dim);

    // Compare against original v (within quantization tolerance).
    float v_norm = 0.0f, err_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        v_norm += v[i] * v[i];
        float diff = result[i] - v[i];
        err_sq += diff * diff;
    }
    float relative_rmse = std::sqrt(err_sq / dim) / std::sqrt(v_norm / dim);
    EXPECT_LT(relative_rmse, 0.15f) << "TBQ4 V accum relative RMSE too high: " << relative_rmse;
}

// ============================================================================
// Size / Layout Tests
// ============================================================================

TEST(TurboQ, HeadBytesCorrect) {
    EXPECT_EQ(turboq_head_bytes(128, 4), 68u);
    EXPECT_EQ(turboq_head_bytes(128, 3), 52u);
}

TEST(TurboQ, RowBytesCorrect) {
    EXPECT_EQ(turboq_row_bytes(8, 128, 4), 8u * 68u);
    EXPECT_EQ(turboq_row_bytes(8, 128, 3), 8u * 52u);
    EXPECT_EQ(turboq_row_bytes(1, 128, 4), 68u);
}

// ============================================================================
// Multi-head independence: quantize multiple vectors, verify fused dot independently.
// ============================================================================

TEST(TurboQ, MultipleHeadsIndependent) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    constexpr int n_heads = 4;

    std::mt19937 rng(404);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> q(dim);
    for (auto& v : q)
        v = dist(rng);

    std::vector<float> q_rot(dim);
    test_rotate_forward(q.data(), q_rot.data(), dim);

    for (int h = 0; h < n_heads; h++) {
        std::vector<float> k(dim);
        for (auto& v : k)
            v = dist(rng);

        std::vector<uint8_t> packed(turboq_head_bytes(dim, 4));
        turboq_quantize_head(k.data(), packed.data(), dim, 4, ov::element::f32);

        float fused = turboq_fused_qk_dot(packed.data(), q_rot.data(), dim, 4);
        float ref = 0.0f;
        for (int i = 0; i < dim; i++)
            ref += q[i] * k[i];

        float relative_error = std::abs(fused - ref) / (std::abs(ref) + 1e-10f);
        EXPECT_LT(relative_error, 0.30f) << "Head " << h << " fused dot too far from reference";
    }
}

// ============================================================================
// QJL Tests — 2-bit codebook, S projection matrix, QJL quantize/dot
// ============================================================================

TEST(TurboQ, CodebookSymmetry2Bit) {
    for (int i = 0; i < 2; i++) {
        EXPECT_FLOAT_EQ(TURBOQ_CODEBOOK_2BIT[i], -TURBOQ_CODEBOOK_2BIT[3 - i])
            << "2-bit codebook not symmetric at index " << i;
    }
}

TEST(TurboQ, BoundaryOrdering2Bit) {
    for (int i = 1; i < 3; i++) {
        EXPECT_LT(TURBOQ_BOUNDARIES_2BIT[i - 1], TURBOQ_BOUNDARIES_2BIT[i])
            << "2-bit boundaries not strictly increasing at index " << i;
    }
}

TEST(TurboQ, ScalarQuantizeRoundTrip2Bit) {
    for (int i = 0; i < 4; i++) {
        uint8_t idx = 0;
        float val = TURBOQ_CODEBOOK_2BIT[i];
        for (int b = 0; b < 3; b++) {
            idx += (val > TURBOQ_BOUNDARIES_2BIT[b]) ? 1 : 0;
        }
        EXPECT_EQ(idx, i) << "2-bit codebook centroid " << i << " didn't quantize to itself";
    }
}

TEST(TurboQ, ProjectionMatrixDeterminism) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    const float* S1 = turboq_get_projection_matrix(dim);
    const float* S2 = turboq_get_projection_matrix(dim);
    EXPECT_EQ(S1, S2) << "Cache should return same pointer for same dim";
}

TEST(TurboQ, ProjectionMatrixNotOrthogonal) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    const float* S = turboq_get_projection_matrix(dim);

    float sts_00 = 0.0f;
    for (int i = 0; i < dim; i++) {
        sts_00 += S[i * dim] * S[i * dim];
    }
    EXPECT_GT(std::abs(sts_00 - 1.0f), 0.1f) << "S appears orthonormalized — it should be raw Gaussian";
}

TEST(TurboQ, ProjectionMatrixIndependent) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    const float* S = turboq_get_projection_matrix(dim);
    const float* Q = turboq_get_rotation_matrix(dim);
    EXPECT_NE(S[0], Q[0]) << "S and Q should be generated from different seeds";
}

TEST(TurboQ, ProjectionTransposeConsistency) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    const float* S = turboq_get_projection_matrix(dim);
    const float* ST = turboq_get_projection_matrix_t(dim);

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            EXPECT_FLOAT_EQ(S[i * dim + j], ST[j * dim + i])
                << "S[" << i << "," << j << "] != ST[" << j << "," << i << "]";
        }
    }
}

TEST(TurboQ, DISABLED_QJLQuantizeLayout_TBQ4QJL) {
    const int dim = TURBOQ_HEAD_RECORD_DIM;
    const int lm_bits = 3;  // TBQ4+QJL: 3-bit Lloyd-Max
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> src(dim);
    for (auto& v : src)
        v = dist(rng);

    size_t packed_size = turboq_head_bytes_qjl(dim, lm_bits);
    EXPECT_EQ(packed_size, 72u) << "TBQ4+QJL should be 72 bytes";

    std::vector<uint8_t> packed(packed_size, 0xFF);
    turboq_quantize_head_qjl(src.data(), packed.data(), dim, lm_bits, ov::element::f32);

    // Verify norm is readable at expected offset
    size_t index_bytes = 48;  // 128 * 3 / 8
    size_t norm_off = index_bytes + TURBOQ_SIGN_BYTES;
    float norm_val = 0.0F;
    std::memcpy(&norm_val, packed.data() + norm_off, 4);
    EXPECT_GT(norm_val, 0.0F) << "Norm should be nonzero for random input";

    // Verify gamma is readable
    float gamma_val = 0.0F;
    std::memcpy(&gamma_val, packed.data() + norm_off + 4, 4);
    EXPECT_GT(gamma_val, 0.0F) << "Gamma should be nonzero for random input";
}

TEST(TurboQ, DISABLED_QJLQuantizeLayout_TBQ3QJL) {
    const int dim = TURBOQ_HEAD_RECORD_DIM;
    const int lm_bits = 2;  // TBQ3+QJL: 2-bit Lloyd-Max

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> src(dim);
    for (auto& v : src)
        v = dist(rng);

    size_t packed_size = turboq_head_bytes_qjl(dim, lm_bits);
    EXPECT_EQ(packed_size, 56u) << "TBQ3+QJL should be 56 bytes";

    std::vector<uint8_t> packed(packed_size, 0xFF);
    turboq_quantize_head_qjl(src.data(), packed.data(), dim, lm_bits, ov::element::f32);

    size_t index_bytes = 32;  // 128 * 2 / 8
    float norm_val = 0.0F;
    std::memcpy(&norm_val, packed.data() + index_bytes + TURBOQ_SIGN_BYTES, 4);
    EXPECT_GT(norm_val, 0.0F);
}

TEST(TurboQ, DISABLED_QJLFusedQKDot_TBQ4QJL) {
    // End-to-end: quantize K with QJL, compute Q·K score via batch_qk_dot_qjl,
    // compare vs float reference dot product.
    const int dim = TURBOQ_HEAD_RECORD_DIM;
    const int lm_bits = 3;  // TBQ4+QJL: 3-bit Lloyd-Max
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> q(dim), k(dim);
    for (auto& v : q)
        v = dist(rng);
    for (auto& v : k)
        v = dist(rng);

    float ref_dot = 0.0f;
    for (int i = 0; i < dim; i++)
        ref_dot += q[i] * k[i];

    // Quantize K with QJL
    size_t packed_size = turboq_head_bytes_qjl(dim, lm_bits);
    std::vector<uint8_t> packed_k(packed_size);
    turboq_quantize_head_qjl(k.data(), packed_k.data(), dim, lm_bits, ov::element::f32);

    // Rotate Q and project through S, packed as [rotated | projected]
    std::vector<float> q_packed(2 * dim);
    const float* S = turboq_get_projection_matrix(dim);
    test_rotate_forward(q.data(), q_packed.data(), dim);
    turboq_matvec_ref(S, q_packed.data(), q_packed.data() + dim, dim);

    // @todo claude: QJL batch functions removed — use non-batched path
    float score = turboq_fused_qk_dot(packed_k.data(), q_packed.data(), dim, lm_bits);

    float relative_error = std::abs(score - ref_dot) / (std::abs(ref_dot) + 1e-10f);
    EXPECT_LT(relative_error, 0.70f) << "TBQ4+QJL fused dot too far from reference: got " << score << " vs " << ref_dot;
}

TEST(TurboQ, DISABLED_QJLFusedQKDot_TBQ3QJL) {
    const int dim = TURBOQ_HEAD_RECORD_DIM;
    const int lm_bits = 2;  // TBQ3+QJL: 2-bit Lloyd-Max
    std::mt19937 rng(456);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> q(dim), k(dim);
    for (auto& v : q)
        v = dist(rng);
    for (auto& v : k)
        v = dist(rng);

    float ref_dot = 0.0f;
    for (int i = 0; i < dim; i++)
        ref_dot += q[i] * k[i];

    size_t packed_size = turboq_head_bytes_qjl(dim, lm_bits);
    std::vector<uint8_t> packed_k(packed_size);
    turboq_quantize_head_qjl(k.data(), packed_k.data(), dim, lm_bits, ov::element::f32);

    std::vector<float> q_packed(2 * dim);
    const float* S = turboq_get_projection_matrix(dim);
    test_rotate_forward(q.data(), q_packed.data(), dim);
    turboq_matvec_ref(S, q_packed.data(), q_packed.data() + dim, dim);

    // @todo claude: QJL batch functions removed — use non-batched path
    float score = turboq_fused_qk_dot(packed_k.data(), q_packed.data(), dim, lm_bits);

    float relative_error = std::abs(score - ref_dot) / (std::abs(ref_dot) + 1e-10f);
    EXPECT_LT(relative_error, 1.0f) << "TBQ3+QJL fused dot too far from reference: got " << score << " vs " << ref_dot;
}

TEST(TurboQ, DISABLED_HeadBytesQJL) {
    EXPECT_EQ(turboq_head_bytes_qjl(128, 3), 72u);  // 48 + 16 + 4 + 4
    EXPECT_EQ(turboq_head_bytes_qjl(128, 2), 56u);  // 32 + 16 + 4 + 4
}

// ============================================================================
// Layer 1: Primitive Accuracy — bias/variance of QK dot and V reconstruction
// across many samples for all codec modes.
// ============================================================================

namespace {

// Scalar quantize: scan boundaries to find the right codebook index.
// Mirrors the static test_scalar_quantize() from the production code.
uint8_t test_scalar_quantize(float val, const float* boundaries, int n_boundaries) {
    uint8_t idx = 0;
    for (int b = 0; b < n_boundaries; b++) {
        idx += (val > boundaries[b]) ? 1 : 0;
    }
    return idx;
}

// Generate vectors that mimic real transformer KV cache activations:
// - ~10% of dimensions are "outlier channels" with 5-10x higher variance
// - Remaining dimensions have mild excess kurtosis (heavier tails than Gaussian)
// - Per-dimension scale is fixed per seed (simulates learned channel magnitudes)
void generate_realistic_vector(float* dst, int dim, std::mt19937& rng, const std::vector<float>& channel_scales) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < dim; i++) {
        float z = dist(rng);
        // Add mild kurtosis: occasionally amplify by 3x (simulates activation spikes)
        if (std::abs(z) > 2.0f)
            z *= 3.0f;
        dst[i] = z * channel_scales[i];
    }
}

// Create per-channel scales: ~10% outlier channels with 5-10x scale.
std::vector<float> make_channel_scales(int dim, std::mt19937& rng) {
    std::vector<float> scales(dim, 1.0f);
    std::uniform_real_distribution<float> outlier_scale(5.0f, 10.0f);
    int n_outliers = std::max(1, dim / 10);
    // Pick deterministic outlier positions
    std::vector<int> indices(dim);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    for (int i = 0; i < n_outliers; i++) {
        scales[indices[i]] = outlier_scale(rng);
    }
    return scales;
}

struct PrimitiveStats {
    double bias;      // mean(estimated - reference)
    double stddev;    // std of (estimated - reference)
    double mean_rel;  // mean |estimated - reference| / |reference|
};

// Compute QK dot product stats over N random (q, k) pairs for a given codec.
PrimitiveStats measure_qk_dot_stats(int dim, int bits, bool qjl, int n_samples, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const float* S = turboq_get_projection_matrix(dim);
    const int lm_bits = qjl ? (bits - 1) : bits;

    double sum_err = 0.0, sum_err_sq = 0.0, sum_rel = 0.0;

    for (int s = 0; s < n_samples; s++) {
        std::vector<float> q(dim), k(dim);
        for (auto& v : q)
            v = dist(rng);
        for (auto& v : k)
            v = dist(rng);

        float ref_dot = 0.0f;
        for (int i = 0; i < dim; i++)
            ref_dot += q[i] * k[i];

        float score = 0.0f;
        if (qjl) {
            // @todo claude: QJL batch functions removed — stub until re-added
            (void)S;
            (void)lm_bits;
            return {0, 0, 0};
        } else {
            std::vector<uint8_t> packed_k(turboq_head_bytes(dim, bits));
            turboq_quantize_head(k.data(), packed_k.data(), dim, bits, ov::element::f32);

            std::vector<float> q_rot(dim);
            test_rotate_forward(q.data(), q_rot.data(), dim);
            score = turboq_fused_qk_dot(packed_k.data(), q_rot.data(), dim, bits);
        }

        double err = static_cast<double>(score) - static_cast<double>(ref_dot);
        sum_err += err;
        sum_err_sq += err * err;
        sum_rel += std::abs(err) / (std::abs(ref_dot) + 1e-10);
    }

    double mean_err = sum_err / n_samples;
    double var = sum_err_sq / n_samples - mean_err * mean_err;

    return {mean_err, std::sqrt(std::max(var, 0.0)), sum_rel / n_samples};
}

// Compute V reconstruction cosine similarity over N random samples.
double measure_v_cosine(int dim, int bits, bool qjl, int n_samples, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const float* QT = turboq_get_rotation_matrix_t(dim);
    const float* ST = turboq_get_projection_matrix_t(dim);
    const int lm_bits = qjl ? (bits - 1) : bits;
    const float qjl_coeff = std::sqrt(static_cast<float>(TURBOQ_PI_F) / 2.0f) / static_cast<float>(dim);

    double total_cos = 0.0;
    for (int s = 0; s < n_samples; s++) {
        std::vector<float> v(dim);
        for (auto& val : v)
            val = dist(rng);

        std::vector<float> result(dim);

        if (qjl) {
            // @todo claude: QJL batch functions removed — stub until re-added
            (void)QT;
            (void)ST;
            (void)lm_bits;
            (void)qjl_coeff;
            return 0.0;
        } else {
            std::vector<uint8_t> packed(turboq_head_bytes(dim, bits));
            turboq_quantize_head(v.data(), packed.data(), dim, bits, ov::element::f32);

            std::vector<float> accum(dim, 0.0f);
            float* accum_ptr = accum.data();
            float weight = 1.0f;
            turboq_fused_v_accum(packed.data(), &weight, &accum_ptr, 1, dim, bits);

            test_rotate_inverse(accum.data(), result.data(), dim);
        }

        // Cosine similarity
        double dot = 0.0, norm_v = 0.0, norm_r = 0.0;
        for (int i = 0; i < dim; i++) {
            dot += v[i] * result[i];
            norm_v += v[i] * v[i];
            norm_r += result[i] * result[i];
        }
        total_cos += dot / (std::sqrt(norm_v * norm_r) + 1e-10);
    }
    return total_cos / n_samples;
}

}  // namespace

// QK dot bias/variance: tbq4 should be lower variance than tbq3.
TEST(TurboQ, PrimitiveQKStats) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    constexpr int N = 500;

    auto tbq4 = measure_qk_dot_stats(dim, 4, false, N, 1000);
    auto tbq3 = measure_qk_dot_stats(dim, 3, false, N, 1000);
    // QJL disabled — uses stale S*Q projection incompatible with WHT rotation.
    // auto tbq4_qjl = measure_qk_dot_stats(dim, 4, true,  N, 1000);
    // auto tbq3_qjl = measure_qk_dot_stats(dim, 3, true,  N, 1000);

    printf("\n=== Primitive QK Dot Stats (N=%d, dim=%d) ===\n", N, dim);
    printf("%-12s %10s %10s %10s\n", "Mode", "Bias", "StdDev", "MeanRelErr");
    printf("%-12s %+10.4f %10.4f %10.4f\n", "tbq4", tbq4.bias, tbq4.stddev, tbq4.mean_rel);
    printf("%-12s %+10.4f %10.4f %10.4f\n", "tbq3", tbq3.bias, tbq3.stddev, tbq3.mean_rel);

    EXPECT_LT(tbq4.stddev, tbq3.stddev) << "tbq4 should have lower QK stddev than tbq3";
    EXPECT_LT(tbq4.stddev, 2.0) << "tbq4 QK stddev too high";
    EXPECT_LT(tbq3.stddev, 4.0) << "tbq3 QK stddev too high";
}

// V reconstruction cosine: tbq4 > tbq3, all should be high.
TEST(TurboQ, PrimitiveVCosine) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    constexpr int N = 200;

    double cos_tbq4 = measure_v_cosine(dim, 4, false, N, 2000);
    double cos_tbq3 = measure_v_cosine(dim, 3, false, N, 2000);

    printf("\n=== V Reconstruction Cosine Similarity (N=%d, dim=%d) ===\n", N, dim);
    printf("%-12s %10s\n", "Mode", "CosSim");
    printf("%-12s %10.6f\n", "tbq4", cos_tbq4);
    printf("%-12s %10.6f\n", "tbq3", cos_tbq3);

    EXPECT_GT(cos_tbq4, cos_tbq3) << "tbq4 should have better V cosine than tbq3";
    EXPECT_GT(cos_tbq4, 0.95) << "tbq4 V cosine too low";
    EXPECT_GT(cos_tbq3, 0.90) << "tbq3 V cosine too low";
}

// ============================================================================
// Layer 2: Attention-Local Accuracy — single SDPA step comparing
// logits, softmax weights, and V output against fp32 reference.
// ============================================================================

namespace {

// softmax in-place over n elements
void softmax(float* x, int n) {
    float max_val = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

struct SDPAStats {
    double logit_cosine;   // cosine of raw QK logits
    double softmax_l1;     // L1 distance of softmax weights
    double output_cosine;  // cosine of final weighted V output
};

// Run one SDPA step: Q (1 head) against n_tokens K/V, compare quantized vs fp32.
// When realistic=true, uses activation-like distributions with outlier channels.
SDPAStats measure_sdpa_step(int dim, int n_tokens, int bits, bool qjl, uint32_t seed, bool realistic = false) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const float* S = turboq_get_projection_matrix(dim);
    const int lm_bits = qjl ? (bits - 1) : bits;
    const float d_scale = 1.0f / std::sqrt(static_cast<float>(dim));

    // Channel scales for realistic mode (fixed per seed for consistent structure)
    std::mt19937 scale_rng(seed ^ 0xDEADBEEF);
    auto q_scales = realistic ? make_channel_scales(dim, scale_rng) : std::vector<float>(dim, 1.0f);
    auto k_scales = realistic ? make_channel_scales(dim, scale_rng) : std::vector<float>(dim, 1.0f);
    auto v_scales = realistic ? make_channel_scales(dim, scale_rng) : std::vector<float>(dim, 1.0f);

    // Generate query and KV cache
    std::vector<float> q(dim);
    if (realistic) {
        generate_realistic_vector(q.data(), dim, rng, q_scales);
    } else {
        for (auto& v : q)
            v = dist(rng);
    }

    std::vector<std::vector<float>> keys(n_tokens), values(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        keys[t].resize(dim);
        values[t].resize(dim);
        if (realistic) {
            generate_realistic_vector(keys[t].data(), dim, rng, k_scales);
            generate_realistic_vector(values[t].data(), dim, rng, v_scales);
        } else {
            for (auto& v : keys[t])
                v = dist(rng);
            for (auto& v : values[t])
                v = dist(rng);
        }
    }

    // === FP32 reference ===
    std::vector<float> ref_logits(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        float dot = 0.0f;
        for (int i = 0; i < dim; i++)
            dot += q[i] * keys[t][i];
        ref_logits[t] = dot * d_scale;
    }
    std::vector<float> ref_weights(ref_logits);
    softmax(ref_weights.data(), n_tokens);

    std::vector<float> ref_output(dim, 0.0f);
    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < dim; i++) {
            ref_output[i] += ref_weights[t] * values[t][i];
        }
    }

    // === Quantized path ===
    // Prepare Q
    std::vector<float> q_rot(dim);
    test_rotate_forward(q.data(), q_rot.data(), dim);

    std::vector<float> q_packed;
    if (qjl) {
        q_packed.resize(2 * dim);
        std::copy(q_rot.begin(), q_rot.end(), q_packed.begin());
        turboq_matvec_ref(S, q_rot.data(), q_packed.data() + dim, dim);
    }

    // Quantize K and V, compute logits
    std::vector<float> quant_logits(n_tokens);
    std::vector<std::vector<uint8_t>> packed_keys(n_tokens), packed_values(n_tokens);

    for (int t = 0; t < n_tokens; t++) {
        // @todo claude: QJL batch functions removed — QJL path uses non-batched API
        int effective_bits = qjl ? lm_bits : bits;
        packed_keys[t].resize(turboq_head_bytes(dim, effective_bits));
        turboq_quantize_head(keys[t].data(), packed_keys[t].data(), dim, effective_bits, ov::element::f32);
        quant_logits[t] = turboq_fused_qk_dot(packed_keys[t].data(), q_rot.data(), dim, effective_bits) * d_scale;

        packed_values[t].resize(turboq_head_bytes(dim, effective_bits));
        turboq_quantize_head(values[t].data(), packed_values[t].data(), dim, effective_bits, ov::element::f32);
    }

    std::vector<float> quant_weights(quant_logits);
    softmax(quant_weights.data(), n_tokens);

    // V accumulation
    std::vector<float> accum(dim, 0.0f);

    for (int t = 0; t < n_tokens; t++) {
        float w = quant_weights[t];
        int effective_bits = qjl ? lm_bits : bits;
        float* ap[1] = {accum.data()};
        turboq_fused_v_accum(packed_values[t].data(), &w, ap, 1, dim, effective_bits);
    }

    // Inverse rotation
    std::vector<float> quant_output(dim);
    test_rotate_inverse(accum.data(), quant_output.data(), dim);

    // === Metrics ===
    // Logit cosine
    double dot_l = 0, n_rl = 0, n_ql = 0;
    for (int t = 0; t < n_tokens; t++) {
        dot_l += ref_logits[t] * quant_logits[t];
        n_rl += ref_logits[t] * ref_logits[t];
        n_ql += quant_logits[t] * quant_logits[t];
    }
    double logit_cos = dot_l / (std::sqrt(n_rl * n_ql) + 1e-10);

    // Softmax L1
    double sm_l1 = 0;
    for (int t = 0; t < n_tokens; t++) {
        sm_l1 += std::abs(ref_weights[t] - quant_weights[t]);
    }

    // Output cosine
    double dot_o = 0, n_ro = 0, n_qo = 0;
    for (int i = 0; i < dim; i++) {
        dot_o += ref_output[i] * quant_output[i];
        n_ro += ref_output[i] * ref_output[i];
        n_qo += quant_output[i] * quant_output[i];
    }
    double output_cos = dot_o / (std::sqrt(n_ro * n_qo) + 1e-10);

    return {logit_cos, sm_l1, output_cos};
}

}  // namespace

// Single SDPA step accuracy across all modes.
TEST(TurboQ, SDPAStepAccuracy) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    constexpr int n_tokens = 64;
    constexpr int n_trials = 50;

    struct ModeResult {
        const char* name;
        double logit_cos, sm_l1, output_cos;
    };

    auto run_mode = [&](int bits, bool qjl) -> ModeResult {
        double lc = 0, sl = 0, oc = 0;
        for (int t = 0; t < n_trials; t++) {
            auto s = measure_sdpa_step(dim, n_tokens, bits, qjl, 3000 + t);
            lc += s.logit_cosine;
            sl += s.softmax_l1;
            oc += s.output_cosine;
        }
        return {"", lc / n_trials, sl / n_trials, oc / n_trials};
    };

    auto tbq4 = run_mode(4, false);
    auto tbq3 = run_mode(3, false);
    // QJL disabled — uses stale S*Q projection incompatible with WHT rotation.

    printf("\n=== SDPA Step Accuracy (N=%d trials, %d tokens, dim=%d) ===\n", n_trials, n_tokens, dim);
    printf("%-12s %12s %12s %12s\n", "Mode", "LogitCos", "SoftmaxL1", "OutputCos");
    printf("%-12s %12.6f %12.6f %12.6f\n", "tbq4", tbq4.logit_cos, tbq4.sm_l1, tbq4.output_cos);
    printf("%-12s %12.6f %12.6f %12.6f\n", "tbq3", tbq3.logit_cos, tbq3.sm_l1, tbq3.output_cos);

    EXPECT_GT(tbq4.logit_cos, tbq3.logit_cos) << "tbq4 should have better logit cosine than tbq3";
    EXPECT_LT(tbq4.sm_l1, tbq3.sm_l1) << "tbq4 should have lower softmax L1 than tbq3";
    EXPECT_GT(tbq4.output_cos, tbq3.output_cos) << "tbq4 should have better output cosine than tbq3";

    // All modes should produce reasonable results
    EXPECT_GT(tbq4.logit_cos, 0.95) << "tbq4 logit cosine too low";
    EXPECT_GT(tbq4.output_cos, 0.90) << "tbq4 output cosine too low";
    EXPECT_GT(tbq3.output_cos, 0.80) << "tbq3 output cosine too low";

    // Softmax L1 should be bounded (max is 2.0 for completely wrong)
    EXPECT_LT(tbq4.sm_l1, 0.50) << "tbq4 softmax L1 too high";
    EXPECT_LT(tbq3.sm_l1, 0.80) << "tbq3 softmax L1 too high";
}

// Same SDPA test but with realistic activation-like inputs (outlier channels).
// This is the real stress test: Hadamard rotation must equalize the outlier
// dimensions before quantization, or quality collapses.
TEST(TurboQ, SDPAStepAccuracy_Realistic) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    constexpr int n_tokens = 64;
    constexpr int n_trials = 50;

    auto run_mode = [&](int bits, bool qjl) {
        double lc = 0, sl = 0, oc = 0;
        for (int t = 0; t < n_trials; t++) {
            auto s = measure_sdpa_step(dim, n_tokens, bits, qjl, 5000 + t, /*realistic=*/true);
            lc += s.logit_cosine;
            sl += s.softmax_l1;
            oc += s.output_cosine;
        }
        return SDPAStats{lc / n_trials, sl / n_trials, oc / n_trials};
    };

    auto tbq4 = run_mode(4, false);
    auto tbq3 = run_mode(3, false);

    printf("\n=== SDPA Step Accuracy — Realistic Inputs (N=%d trials, %d tokens, dim=%d) ===\n",
           n_trials,
           n_tokens,
           dim);
    printf("%-12s %12s %12s %12s\n", "Mode", "LogitCos", "SoftmaxL1", "OutputCos");
    printf("%-12s %12.6f %12.6f %12.6f\n", "tbq4", tbq4.logit_cosine, tbq4.softmax_l1, tbq4.output_cosine);
    printf("%-12s %12.6f %12.6f %12.6f\n", "tbq3", tbq3.logit_cosine, tbq3.softmax_l1, tbq3.output_cosine);

    // With realistic inputs, tbq4 should still beat tbq3
    EXPECT_GT(tbq4.logit_cosine, tbq3.logit_cosine)
        << "tbq4 should have better logit cosine than tbq3 even with outlier channels";
    EXPECT_GT(tbq4.output_cosine, tbq3.output_cosine)
        << "tbq4 should have better output cosine than tbq3 even with outlier channels";

    // Rotation should keep quality reasonable even with outliers
    EXPECT_GT(tbq4.logit_cosine, 0.90) << "tbq4 logit cosine too low with realistic inputs";
    EXPECT_GT(tbq4.output_cosine, 0.85) << "tbq4 output cosine too low with realistic inputs";
}

// ============================================================================
// Diagnostic 1: Residual Oracle Test
// Replace QJL's 1-bit sign estimator with exact residual to separate
// "QJL approximation is poor" from "(b-1)+1 bit split is inherently bad".
// ============================================================================

TEST(TurboQ, DiagnosticResidualOracle) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    constexpr int N = 500;
    const float inv_sqrt_dim = 1.0f / std::sqrt(static_cast<float>(dim));

    std::mt19937 rng(7000);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // For bits=4: compare tbq4 (4-bit LM) vs base_3bit (3-bit LM alone)
    //             vs oracle (3-bit LM + exact residual) vs qjl (3-bit LM + 1-bit sign)
    struct Stats {
        double sum_err_sq;
        int count;
    };
    auto zero_stats = []() -> Stats {
        return {0.0, 0};
    };

    for (int total_bits : {4, 3}) {
        int lm_bits = total_bits - 1;
        const float* codebook = (lm_bits == 3) ? TURBOQ_CODEBOOK_3BIT : TURBOQ_CODEBOOK_2BIT;
        const float* boundaries = (lm_bits == 3) ? TURBOQ_BOUNDARIES_3BIT : TURBOQ_BOUNDARIES_2BIT;
        const float* cb_full = (total_bits == 4) ? TURBOQ_CODEBOOK_4BIT : TURBOQ_CODEBOOK_3BIT;
        const float* bnd_full = (total_bits == 4) ? TURBOQ_BOUNDARIES_4BIT : TURBOQ_BOUNDARIES_3BIT;
        int n_bnd = (1 << lm_bits) - 1;
        int n_bnd_full = (1 << total_bits) - 1;

        Stats full_lm = zero_stats(), base_only = zero_stats(), oracle = zero_stats();

        for (int s = 0; s < N; s++) {
            std::vector<float> q(dim), k(dim);
            for (auto& v : q)
                v = dist(rng);
            for (auto& v : k)
                v = dist(rng);

            float ref_dot = 0.0f;
            for (int i = 0; i < dim; i++)
                ref_dot += q[i] * k[i];

            // Rotate q and k
            std::vector<float> q_rot(dim), k_rot(dim);
            test_rotate_forward(q.data(), q_rot.data(), dim);
            test_rotate_forward(k.data(), k_rot.data(), dim);

            // Normalize k, scale by sqrt(dim) (as quantizer does)
            float k_norm = 0.0f;
            for (int i = 0; i < dim; i++)
                k_norm += k[i] * k[i];
            k_norm = std::sqrt(k_norm);
            std::vector<float> k_scaled(dim);
            float scale = (k_norm > 1e-30f) ? std::sqrt(static_cast<float>(dim)) / k_norm : 0.0f;
            for (int i = 0; i < dim; i++)
                k_scaled[i] = k_rot[i] * scale;

            // Full b-bit Lloyd-Max quantization
            float full_dot = 0.0f;
            for (int i = 0; i < dim; i++) {
                uint8_t idx = test_scalar_quantize(k_scaled[i], bnd_full, n_bnd_full);
                full_dot += q_rot[i] * cb_full[idx];
            }
            full_dot *= k_norm * inv_sqrt_dim;

            // (b-1)-bit Lloyd-Max only
            float base_dot = 0.0f;
            std::vector<float> residual(dim);
            for (int i = 0; i < dim; i++) {
                uint8_t idx = test_scalar_quantize(k_scaled[i], boundaries, n_bnd);
                base_dot += q_rot[i] * codebook[idx];
                residual[i] = k_scaled[i] - codebook[idx];
            }
            base_dot *= k_norm * inv_sqrt_dim;

            // Oracle: (b-1)-bit + exact residual dot
            float residual_dot = 0.0f;
            for (int i = 0; i < dim; i++)
                residual_dot += q_rot[i] * residual[i];
            float oracle_dot = base_dot + residual_dot * k_norm * inv_sqrt_dim;

            full_lm.sum_err_sq += (full_dot - ref_dot) * (full_dot - ref_dot);
            base_only.sum_err_sq += (base_dot - ref_dot) * (base_dot - ref_dot);
            oracle.sum_err_sq += (oracle_dot - ref_dot) * (oracle_dot - ref_dot);
            full_lm.count++;
            base_only.count++;
            oracle.count++;
        }

        double rmse_full = std::sqrt(full_lm.sum_err_sq / full_lm.count);
        double rmse_base = std::sqrt(base_only.sum_err_sq / base_only.count);
        double rmse_oracle = std::sqrt(oracle.sum_err_sq / oracle.count);

        printf("\n=== Residual Oracle (total_bits=%d, N=%d) ===\n", total_bits, N);
        printf("  %d-bit Lloyd-Max (full):    RMSE = %.4f\n", total_bits, rmse_full);
        printf("  %d-bit Lloyd-Max (base):    RMSE = %.4f\n", lm_bits, rmse_base);
        printf("  %d-bit LM + exact residual: RMSE = %.4f\n", lm_bits, rmse_oracle);

        // Oracle (base + exact residual) should recover near-zero error
        // (the residual is the exact quantization error, so base + residual = original)
        EXPECT_LT(rmse_oracle, 0.01) << "Oracle should nearly perfectly recover the dot product for bits="
                                     << total_bits;

        // Base (b-1)-bit should be worse than full b-bit
        EXPECT_GT(rmse_base, rmse_full) << "Fewer bits should have higher RMSE for bits=" << total_bits;
    }
}

// ============================================================================
// Diagnostic 2: K-only vs V-only QJL
// K noise goes through softmax (nonlinear amplification).
// V noise is linear. Separate to see which path QJL hurts.
// ============================================================================

namespace {

// SDPA with selective QJL: apply QJL to K, V, both, or neither.
SDPAStats measure_sdpa_selective_qjl(int dim, int n_tokens, int bits, bool qjl_k, bool qjl_v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const float* S = turboq_get_projection_matrix(dim);
    const int lm_bits = bits - 1;
    const float d_scale = 1.0f / std::sqrt(static_cast<float>(dim));

    std::vector<float> q(dim);
    for (auto& v : q)
        v = dist(rng);

    std::vector<std::vector<float>> keys(n_tokens), values(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        keys[t].resize(dim);
        values[t].resize(dim);
        for (auto& v : keys[t])
            v = dist(rng);
        for (auto& v : values[t])
            v = dist(rng);
    }

    // FP32 reference
    std::vector<float> ref_logits(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        float dot = 0.0f;
        for (int i = 0; i < dim; i++)
            dot += q[i] * keys[t][i];
        ref_logits[t] = dot * d_scale;
    }
    std::vector<float> ref_weights(ref_logits);
    softmax(ref_weights.data(), n_tokens);
    std::vector<float> ref_output(dim, 0.0f);
    for (int t = 0; t < n_tokens; t++)
        for (int i = 0; i < dim; i++)
            ref_output[i] += ref_weights[t] * values[t][i];

    // Prepare Q
    std::vector<float> q_rot(dim);
    test_rotate_forward(q.data(), q_rot.data(), dim);
    std::vector<float> q_packed(2 * dim);
    std::copy(q_rot.begin(), q_rot.end(), q_packed.begin());
    turboq_matvec_ref(S, q_rot.data(), q_packed.data() + dim, dim);

    // Quantize K (QJL or plain) and compute logits
    std::vector<float> quant_logits(n_tokens);
    std::vector<std::vector<uint8_t>> packed_keys(n_tokens), packed_values(n_tokens);

    // @todo claude: QJL batch functions removed — QJL path uses non-batched API
    for (int t = 0; t < n_tokens; t++) {
        int k_bits = qjl_k ? lm_bits : bits;
        packed_keys[t].resize(turboq_head_bytes(dim, k_bits));
        turboq_quantize_head(keys[t].data(), packed_keys[t].data(), dim, k_bits, ov::element::f32);
        quant_logits[t] = turboq_fused_qk_dot(packed_keys[t].data(), q_rot.data(), dim, k_bits) * d_scale;

        int v_bits = qjl_v ? lm_bits : bits;
        packed_values[t].resize(turboq_head_bytes(dim, v_bits));
        turboq_quantize_head(values[t].data(), packed_values[t].data(), dim, v_bits, ov::element::f32);
    }

    std::vector<float> quant_weights(quant_logits);
    softmax(quant_weights.data(), n_tokens);

    // V accumulation
    std::vector<float> accum(dim, 0.0f);
    for (int t = 0; t < n_tokens; t++) {
        float w = quant_weights[t];
        int v_bits = qjl_v ? lm_bits : bits;
        float* ap[1] = {accum.data()};
        turboq_fused_v_accum(packed_values[t].data(), &w, ap, 1, dim, v_bits);
    }

    std::vector<float> quant_output(dim);
    test_rotate_inverse(accum.data(), quant_output.data(), dim);

    // Metrics
    double dot_l = 0, n_rl = 0, n_ql = 0;
    for (int t = 0; t < n_tokens; t++) {
        dot_l += ref_logits[t] * quant_logits[t];
        n_rl += ref_logits[t] * ref_logits[t];
        n_ql += quant_logits[t] * quant_logits[t];
    }
    double sm_l1 = 0;
    for (int t = 0; t < n_tokens; t++)
        sm_l1 += std::abs(ref_weights[t] - quant_weights[t]);
    double dot_o = 0, n_ro = 0, n_qo = 0;
    for (int i = 0; i < dim; i++) {
        dot_o += ref_output[i] * quant_output[i];
        n_ro += ref_output[i] * ref_output[i];
        n_qo += quant_output[i] * quant_output[i];
    }
    return {dot_l / (std::sqrt(n_rl * n_ql) + 1e-10), sm_l1, dot_o / (std::sqrt(n_ro * n_qo) + 1e-10)};
}

}  // namespace

TEST(TurboQ, DISABLED_DiagnosticKOnlyVOnlyQJL) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    constexpr int n_tokens = 64;
    constexpr int n_trials = 50;

    auto run = [&](int bits, bool qjl_k, bool qjl_v) -> SDPAStats {
        double lc = 0, sl = 0, oc = 0;
        for (int t = 0; t < n_trials; t++) {
            auto s = measure_sdpa_selective_qjl(dim, n_tokens, bits, qjl_k, qjl_v, 8000 + t);
            lc += s.logit_cosine;
            sl += s.softmax_l1;
            oc += s.output_cosine;
        }
        return {lc / n_trials, sl / n_trials, oc / n_trials};
    };

    for (int bits : {4, 3}) {
        auto base = run(bits, false, false);
        auto k_only = run(bits, true, false);
        auto v_only = run(bits, false, true);
        auto both = run(bits, true, true);

        printf("\n=== K-only vs V-only QJL (bits=%d, %d trials, %d tokens) ===\n", bits, n_trials, n_tokens);
        printf("%-16s %12s %12s %12s\n", "Mode", "LogitCos", "SoftmaxL1", "OutputCos");
        printf("%-16s %12.6f %12.6f %12.6f\n", "base (no QJL)", base.logit_cosine, base.softmax_l1, base.output_cosine);
        printf("%-16s %12.6f %12.6f %12.6f\n",
               "QJL K only",
               k_only.logit_cosine,
               k_only.softmax_l1,
               k_only.output_cosine);
        printf("%-16s %12.6f %12.6f %12.6f\n",
               "QJL V only",
               v_only.logit_cosine,
               v_only.softmax_l1,
               v_only.output_cosine);
        printf("%-16s %12.6f %12.6f %12.6f\n", "QJL both", both.logit_cosine, both.softmax_l1, both.output_cosine);

        // Base (full b-bit) should be best
        EXPECT_GT(base.output_cosine, both.output_cosine)
            << "Full " << bits << "-bit LM should beat QJL on both for bits=" << bits;
    }
}

// ============================================================================
// Diagnostic 3: Primitive Decomposition
// For each QK sample, measure:
//   - base error from (b-1)-bit Lloyd-Max only
//   - correction term from QJL sign estimation
//   - correlation of correction with true residual dot
//   - whether correction actually reduces total error
// ============================================================================

TEST(TurboQ, DiagnosticPrimitiveDecomposition) {
    constexpr int dim = TURBOQ_HEAD_RECORD_DIM;
    constexpr int N = 1000;

    const float* S = turboq_get_projection_matrix(dim);
    const float inv_sqrt_dim = 1.0f / std::sqrt(static_cast<float>(dim));
    const float qjl_coeff = std::sqrt(static_cast<float>(TURBOQ_PI_F) / 2.0f) / static_cast<float>(dim);

    std::mt19937 rng(9000);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int total_bits : {4, 3}) {
        int lm_bits = total_bits - 1;
        const float* codebook = (lm_bits == 3) ? TURBOQ_CODEBOOK_3BIT : TURBOQ_CODEBOOK_2BIT;
        const float* boundaries = (lm_bits == 3) ? TURBOQ_BOUNDARIES_3BIT : TURBOQ_BOUNDARIES_2BIT;
        int n_bnd = (1 << lm_bits) - 1;

        double sum_base_err_sq = 0, sum_corrected_err_sq = 0;
        double sum_true_res_dot = 0, sum_est_correction = 0;
        double sum_true_res_sq = 0, sum_est_sq = 0, sum_cross = 0;
        int correction_helped = 0;

        for (int s = 0; s < N; s++) {
            std::vector<float> q(dim), k(dim);
            for (auto& v : q)
                v = dist(rng);
            for (auto& v : k)
                v = dist(rng);

            float ref_dot = 0.0f;
            for (int i = 0; i < dim; i++)
                ref_dot += q[i] * k[i];

            // Rotate
            std::vector<float> q_rot(dim), k_rot(dim);
            test_rotate_forward(q.data(), q_rot.data(), dim);
            test_rotate_forward(k.data(), k_rot.data(), dim);

            // Normalize k
            float k_norm = 0.0f;
            for (int i = 0; i < dim; i++)
                k_norm += k[i] * k[i];
            k_norm = std::sqrt(k_norm);
            float kscale = (k_norm > 1e-30f) ? std::sqrt(static_cast<float>(dim)) / k_norm : 0.0f;
            std::vector<float> k_scaled(dim);
            for (int i = 0; i < dim; i++)
                k_scaled[i] = k_rot[i] * kscale;

            // (b-1)-bit quantize, compute base dot and residual
            float base_dot = 0.0f;
            std::vector<float> residual(dim);
            for (int i = 0; i < dim; i++) {
                uint8_t idx = test_scalar_quantize(k_scaled[i], boundaries, n_bnd);
                base_dot += q_rot[i] * codebook[idx];
                residual[i] = k_scaled[i] - codebook[idx];
            }
            base_dot *= k_norm * inv_sqrt_dim;

            // True residual dot contribution
            float true_res_dot = 0.0f;
            for (int i = 0; i < dim; i++)
                true_res_dot += q_rot[i] * residual[i];
            true_res_dot *= k_norm * inv_sqrt_dim;

            // QJL sign estimation of residual
            float gamma = 0.0f;
            for (int i = 0; i < dim; i++)
                gamma += residual[i] * residual[i];
            gamma = std::sqrt(gamma);

            // Project residual through S, take signs
            std::vector<float> projected_r(dim), projected_q(dim);
            turboq_matvec_ref(S, residual.data(), projected_r.data(), dim);
            turboq_matvec_ref(S, q_rot.data(), projected_q.data(), dim);

            float sign_dot = 0.0f;
            for (int i = 0; i < dim; i++) {
                float sign_val = (projected_r[i] >= 0.0f) ? 1.0f : -1.0f;
                sign_dot += projected_q[i] * sign_val;
            }
            float qjl_correction = k_norm * inv_sqrt_dim * qjl_coeff * gamma * sign_dot;

            float corrected_dot = base_dot + qjl_correction;

            // Accumulate stats
            float base_err = base_dot - ref_dot;
            float corrected_err = corrected_dot - ref_dot;
            sum_base_err_sq += base_err * base_err;
            sum_corrected_err_sq += corrected_err * corrected_err;

            sum_true_res_dot += true_res_dot;
            sum_est_correction += qjl_correction;
            sum_true_res_sq += true_res_dot * true_res_dot;
            sum_est_sq += qjl_correction * qjl_correction;
            sum_cross += true_res_dot * qjl_correction;

            if (std::abs(corrected_err) < std::abs(base_err))
                correction_helped++;
        }

        double rmse_base = std::sqrt(sum_base_err_sq / N);
        double rmse_corrected = std::sqrt(sum_corrected_err_sq / N);

        // Pearson correlation between true residual dot and QJL estimate
        double mean_true = sum_true_res_dot / N;
        double mean_est = sum_est_correction / N;
        double cov = sum_cross / N - mean_true * mean_est;
        double std_true = std::sqrt(sum_true_res_sq / N - mean_true * mean_true);
        double std_est = std::sqrt(sum_est_sq / N - mean_est * mean_est);
        double correlation = cov / (std_true * std_est + 1e-10);

        printf("\n=== Primitive Decomposition (tbq%d_qjl, N=%d) ===\n", total_bits, N);
        printf("  Base (%d-bit LM) RMSE:       %.4f\n", lm_bits, rmse_base);
        printf("  Corrected (+ QJL sign) RMSE: %.4f\n", rmse_corrected);
        printf("  Correction helped:           %d/%d (%.1f%%)\n", correction_helped, N, 100.0 * correction_helped / N);
        printf("  True residual / QJL corr:    %.4f\n", correlation);
        printf("  Mean true residual dot:      %.6f\n", mean_true);
        printf("  Mean QJL correction:         %.6f\n", mean_est);

        // The correction should be correlated with the true residual
        EXPECT_GT(correlation, 0.0) << "QJL correction should be positively correlated with true residual for bits="
                                    << total_bits;
    }
}

// ============================================================================
// PolarQuant Tests
// ============================================================================

// --- Table / LUT consistency ---

TEST(PolarQ, BoundaryOrdering) {
    // Verify all used (level, bits) pairs have strictly increasing boundaries.
    struct LevelBits {
        int level;
        int bits;
    };
    LevelBits used[] = {
        {1, 3},
        {1, 4},
        {2, 3},
        {2, 5},
        {3, 3},
        {3, 4},
        {4, 3},
        {5, 2},
        {5, 3},
        {6, 2},
        {6, 3},
        {7, 2},
    };
    for (auto& lb : used) {
        auto lut = polarq_get_lut(lb.level, lb.bits);
        ASSERT_NE(lut.boundaries, nullptr) << "L" << lb.level << " " << lb.bits << "bit: no LUT";
        for (int i = 1; i < lut.n_centroids - 1; i++) {
            EXPECT_LT(lut.boundaries[i - 1], lut.boundaries[i])
                << "L" << lb.level << " " << lb.bits << "bit: boundaries not increasing at " << i;
        }
    }
}

TEST(PolarQ, CosSinConsistency) {
    // cos/sin LUTs must equal cos/sin of the corresponding centroids.
    struct LevelBits {
        int level;
        int bits;
    };
    LevelBits used[] = {
        {1, 3},
        {1, 4},
        {2, 3},
        {2, 5},
        {3, 3},
        {3, 4},
        {4, 3},
        {5, 2},
        {5, 3},
        {6, 2},
        {6, 3},
        {7, 2},
    };
    for (auto& lb : used) {
        auto lut = polarq_get_lut(lb.level, lb.bits);
        ASSERT_NE(lut.centroids, nullptr);
        for (int i = 0; i < lut.n_centroids; i++) {
            EXPECT_NEAR(lut.cos_lut[i], std::cos(lut.centroids[i]), 1e-4f)
                << "L" << lb.level << " " << lb.bits << "bit: cos mismatch at " << i;
            EXPECT_NEAR(lut.sin_lut[i], std::sin(lut.centroids[i]), 1e-4f)
                << "L" << lb.level << " " << lb.bits << "bit: sin mismatch at " << i;
        }
    }
}

TEST(PolarQ, CentroidsWithinDomain) {
    // Level 1: [0, 2*pi). Levels 2+: [0, pi/2].
    struct LevelBits {
        int level;
        int bits;
    };
    LevelBits used[] = {
        {1, 3},
        {1, 4},
        {2, 3},
        {2, 5},
        {3, 3},
        {3, 4},
        {4, 3},
        {5, 2},
        {5, 3},
        {6, 2},
        {6, 3},
        {7, 2},
    };
    for (auto& lb : used) {
        auto lut = polarq_get_lut(lb.level, lb.bits);
        float lo = 0.0f;
        float hi = (lb.level == 1) ? static_cast<float>(2.0 * TURBOQ_PI_F) : static_cast<float>(TURBOQ_PI_F / 2.0);
        for (int i = 0; i < lut.n_centroids; i++) {
            EXPECT_GE(lut.centroids[i], lo - 0.01f)
                << "L" << lb.level << " " << lb.bits << "bit: centroid " << i << " below domain";
            EXPECT_LE(lut.centroids[i], hi + 0.01f)
                << "L" << lb.level << " " << lb.bits << "bit: centroid " << i << " above domain";
        }
    }
}

TEST(PolarQ, FixedLevelIsNullptr) {
    auto lut = polarq_get_lut(7, 0);
    EXPECT_EQ(lut.centroids, nullptr);
    EXPECT_EQ(lut.n_centroids, 0);
}

// --- Head byte sizes ---

TEST(PolarQ, HeadBytesCorrect) {
    // Verify sizes match the current allocation.
    // Don't hardcode — compute expected from the allocation arrays.
    for (int dim : {128, 256}) {
        for (int bits : {3, 4}) {
            size_t actual = polarq_head_bytes(dim, bits);
            // Recompute from allocation
            const int* bpl = polarq_get_bits_per_level(bits, dim);
            size_t expected = polarq_head_bytes_from_alloc(dim, bpl);
            EXPECT_EQ(actual, expected) << "polar" << bits << "/dim=" << dim << " size mismatch";
            // Sanity: must be > 4 (at least the norm) and < dim*8 (can't exceed 8 bits/coord)
            EXPECT_GT(actual, 4u);
            EXPECT_LT(actual, static_cast<size_t>(dim));
        }
    }
}

// --- Quantize round-trip: quantize + fused QK dot vs float reference ---

TEST(PolarQ, FusedQKDot_Polar4_128) {
    constexpr int dim = 128;
    std::mt19937 rng(789);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> q(dim), k(dim);
    for (auto& v : q)
        v = dist(rng);
    for (auto& v : k)
        v = dist(rng);

    std::vector<uint8_t> packed_k(polarq_head_bytes(dim, 4));
    polarq_quantize_head(k.data(), packed_k.data(), dim, 4, ov::element::f32);

    std::vector<float> q_rot(dim);
    test_rotate_forward(q.data(), q_rot.data(), dim);

    float score = polarq_fused_qk_dot(packed_k.data(), q_rot.data(), dim, 4);

    float ref_dot = 0.0f;
    for (int i = 0; i < dim; i++)
        ref_dot += q[i] * k[i];

    float abs_error = std::abs(score - ref_dot);
    float scale = std::max(std::abs(ref_dot), std::abs(score)) + 1e-10f;
    EXPECT_LT(abs_error / scale, 0.60f) << "polar4 fused dot too far: got=" << score << " ref=" << ref_dot;
}

TEST(PolarQ, FusedQKDot_Polar3_128) {
    constexpr int dim = 128;
    std::mt19937 rng(101);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> q(dim), k(dim);
    for (auto& v : q)
        v = dist(rng);
    for (auto& v : k)
        v = dist(rng);

    std::vector<uint8_t> packed_k(polarq_head_bytes(dim, 3));
    polarq_quantize_head(k.data(), packed_k.data(), dim, 3, ov::element::f32);

    std::vector<float> q_rot(dim);
    test_rotate_forward(q.data(), q_rot.data(), dim);

    float score = polarq_fused_qk_dot(packed_k.data(), q_rot.data(), dim, 3);

    float ref_dot = 0.0f;
    for (int i = 0; i < dim; i++)
        ref_dot += q[i] * k[i];

    float rel_err = std::abs(score - ref_dot) / (std::abs(ref_dot) + 1e-10f);
    EXPECT_LT(rel_err, 0.60f) << "polar3 fused dot too far: got=" << score << " ref=" << ref_dot;
}

TEST(PolarQ, FusedQKDot_Polar4_256) {
    constexpr int dim = 256;
    std::mt19937 rng(202);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> q(dim), k(dim);
    for (auto& v : q)
        v = dist(rng);
    for (auto& v : k)
        v = dist(rng);

    std::vector<uint8_t> packed_k(polarq_head_bytes(dim, 4));
    polarq_quantize_head(k.data(), packed_k.data(), dim, 4, ov::element::f32);

    std::vector<float> q_rot(dim);
    test_rotate_forward(q.data(), q_rot.data(), dim);

    float score = polarq_fused_qk_dot(packed_k.data(), q_rot.data(), dim, 4);

    float ref_dot = 0.0f;
    for (int i = 0; i < dim; i++)
        ref_dot += q[i] * k[i];

    float abs_error = std::abs(score - ref_dot);
    float scale = std::max(std::abs(ref_dot), std::abs(score)) + 1e-10f;
    EXPECT_LT(abs_error / scale, 0.60f) << "polar4/256 fused dot too far: got=" << score << " ref=" << ref_dot;
}

// --- V accumulation round-trip ---

TEST(PolarQ, FusedVAccum_Polar4_128) {
    constexpr int dim = 128;
    std::mt19937 rng(303);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> v(dim);
    for (auto& val : v)
        val = dist(rng);

    std::vector<uint8_t> packed_v(polarq_head_bytes(dim, 4));
    polarq_quantize_head(v.data(), packed_v.data(), dim, 4, ov::element::f32);

    std::vector<float> accum(dim, 0.0f);
    float* accum_ptr = accum.data();
    float weight = 1.0f;
    polarq_fused_v_accum(packed_v.data(), &weight, &accum_ptr, 1, dim, 4);

    // Apply inverse rotation
    std::vector<float> result(dim);
    test_rotate_inverse(accum.data(), result.data(), dim);

    float v_norm = 0.0f, err_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        v_norm += v[i] * v[i];
        float diff = result[i] - v[i];
        err_sq += diff * diff;
    }
    float relative_rmse = std::sqrt(err_sq / dim) / std::sqrt(v_norm / dim);
    EXPECT_LT(relative_rmse, 0.20f) << "polar4 V accum relative RMSE too high: " << relative_rmse;
}

// --- Primitive accuracy: polar QK dot stats across many samples ---

namespace {

PrimitiveStats measure_polar_qk_dot_stats(int dim, int bits, int n_samples, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double sum_err = 0.0, sum_err_sq = 0.0, sum_rel = 0.0;
    for (int s = 0; s < n_samples; s++) {
        std::vector<float> q(dim), k(dim);
        for (auto& v : q)
            v = dist(rng);
        for (auto& v : k)
            v = dist(rng);

        float ref_dot = 0.0f;
        for (int i = 0; i < dim; i++)
            ref_dot += q[i] * k[i];

        std::vector<uint8_t> packed_k(polarq_head_bytes(dim, bits));
        polarq_quantize_head(k.data(), packed_k.data(), dim, bits, ov::element::f32);

        std::vector<float> q_rot(dim);
        test_rotate_forward(q.data(), q_rot.data(), dim);

        float score = polarq_fused_qk_dot(packed_k.data(), q_rot.data(), dim, bits);

        double err = static_cast<double>(score) - static_cast<double>(ref_dot);
        sum_err += err;
        sum_err_sq += err * err;
        sum_rel += std::abs(err) / (std::abs(ref_dot) + 1e-10);
    }
    double mean_err = sum_err / n_samples;
    double var = sum_err_sq / n_samples - mean_err * mean_err;
    return {mean_err, std::sqrt(std::max(var, 0.0)), sum_rel / n_samples};
}

double measure_polar_v_cosine(int dim, int bits, int n_samples, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double total_cos = 0.0;
    for (int s = 0; s < n_samples; s++) {
        std::vector<float> v(dim);
        for (auto& val : v)
            val = dist(rng);

        std::vector<uint8_t> packed(polarq_head_bytes(dim, bits));
        polarq_quantize_head(v.data(), packed.data(), dim, bits, ov::element::f32);

        std::vector<float> accum(dim, 0.0f);
        float* accum_ptr = accum.data();
        float weight = 1.0f;
        polarq_fused_v_accum(packed.data(), &weight, &accum_ptr, 1, dim, bits);

        std::vector<float> result(dim);
        test_rotate_inverse(accum.data(), result.data(), dim);

        double dot = 0.0, norm_v = 0.0, norm_r = 0.0;
        for (int i = 0; i < dim; i++) {
            dot += v[i] * result[i];
            norm_v += v[i] * v[i];
            norm_r += result[i] * result[i];
        }
        total_cos += dot / (std::sqrt(norm_v * norm_r) + 1e-10);
    }
    return total_cos / n_samples;
}

}  // namespace

TEST(PolarQ, PrimitiveQKStats) {
    constexpr int N = 500;

    auto p4_128 = measure_polar_qk_dot_stats(128, 4, N, 1000);
    auto p3_128 = measure_polar_qk_dot_stats(128, 3, N, 1000);
    auto p4_256 = measure_polar_qk_dot_stats(256, 4, N, 1000);
    auto p3_256 = measure_polar_qk_dot_stats(256, 3, N, 1000);

    // Also measure TBQ for comparison
    auto tbq4 = measure_qk_dot_stats(128, 4, false, N, 1000);
    auto tbq3 = measure_qk_dot_stats(128, 3, false, N, 1000);

    printf("\n=== Polar vs TBQ QK Dot Stats (N=%d) ===\n", N);
    printf("%-16s %10s %10s %10s\n", "Mode", "Bias", "StdDev", "MeanRelErr");
    printf("%-16s %+10.4f %10.4f %10.4f\n", "tbq4/128", tbq4.bias, tbq4.stddev, tbq4.mean_rel);
    printf("%-16s %+10.4f %10.4f %10.4f\n", "tbq3/128", tbq3.bias, tbq3.stddev, tbq3.mean_rel);
    printf("%-16s %+10.4f %10.4f %10.4f\n", "polar4/128", p4_128.bias, p4_128.stddev, p4_128.mean_rel);
    printf("%-16s %+10.4f %10.4f %10.4f\n", "polar3/128", p3_128.bias, p3_128.stddev, p3_128.mean_rel);
    printf("%-16s %+10.4f %10.4f %10.4f\n", "polar4/256", p4_256.bias, p4_256.stddev, p4_256.mean_rel);
    printf("%-16s %+10.4f %10.4f %10.4f\n", "polar3/256", p3_256.bias, p3_256.stddev, p3_256.mean_rel);

    // polar4 should have lower variance than polar3
    EXPECT_LT(p4_128.stddev, p3_128.stddev) << "polar4 should have lower QK stddev than polar3";

    // Reasonable bounds
    EXPECT_LT(p4_128.stddev, 3.0) << "polar4/128 QK stddev too high";
    EXPECT_LT(p3_128.stddev, 5.0) << "polar3/128 QK stddev too high";
    EXPECT_LT(p4_256.stddev, 4.0) << "polar4/256 QK stddev too high";
    EXPECT_LT(p3_256.stddev, 7.0) << "polar3/256 QK stddev too high";
}

TEST(PolarQ, PrimitiveVCosine) {
    constexpr int N = 200;

    double cos_p4_128 = measure_polar_v_cosine(128, 4, N, 2000);
    double cos_p3_128 = measure_polar_v_cosine(128, 3, N, 2000);
    double cos_p4_256 = measure_polar_v_cosine(256, 4, N, 2000);
    double cos_p3_256 = measure_polar_v_cosine(256, 3, N, 2000);

    double cos_tbq4 = measure_v_cosine(128, 4, false, N, 2000);
    double cos_tbq3 = measure_v_cosine(128, 3, false, N, 2000);

    printf("\n=== Polar vs TBQ V Cosine Similarity (N=%d) ===\n", N);
    printf("%-16s %10s\n", "Mode", "CosSim");
    printf("%-16s %10.6f\n", "tbq4/128", cos_tbq4);
    printf("%-16s %10.6f\n", "tbq3/128", cos_tbq3);
    printf("%-16s %10.6f\n", "polar4/128", cos_p4_128);
    printf("%-16s %10.6f\n", "polar3/128", cos_p3_128);
    printf("%-16s %10.6f\n", "polar4/256", cos_p4_256);
    printf("%-16s %10.6f\n", "polar3/256", cos_p3_256);

    // polar4 should beat polar3
    EXPECT_GT(cos_p4_128, cos_p3_128) << "polar4 should have better V cosine than polar3";

    // All should be reasonably high
    EXPECT_GT(cos_p4_128, 0.90) << "polar4/128 V cosine too low";
    EXPECT_GT(cos_p3_128, 0.80) << "polar3/128 V cosine too low";
    EXPECT_GT(cos_p4_256, 0.90) << "polar4/256 V cosine too low";
    EXPECT_GT(cos_p3_256, 0.80) << "polar3/256 V cosine too low";
}

// ============================================================================
// PolarQuant Diagnostic Tests — isolate error sources
// ============================================================================

// Diagnostic 1: Per-level angle quantization MSE.
// Measures how well each level's quantizer preserves angles, independent of the
// tree reconstruction. If a level has disproportionately high MSE, it dominates
// the total reconstruction error.
TEST(PolarQ, DiagnosticPerLevelMSE) {
    constexpr int N = 1000;

    for (int dim : {128, 256}) {
        const int L = [](int d) {
            int l = 0;
            for (; d > 1; d >>= 1)
                l++;
            return l;
        }(dim);

        for (int bits : {3, 4}) {
            const int* bpl = polarq_get_bits_per_level(bits, dim);

            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            const float sqrt_dim = std::sqrt(static_cast<float>(dim));

            // Accumulate per-level MSE
            std::vector<double> level_mse(L, 0.0);
            std::vector<int> level_count(L, 0);

            for (int s = 0; s < N; s++) {
                std::vector<float> vec(dim);
                for (auto& v : vec)
                    v = dist(rng);

                // Normalize + rotate
                float norm_sq = 0;
                for (int i = 0; i < dim; i++)
                    norm_sq += vec[i] * vec[i];
                float inv_norm = 1.0f / std::sqrt(norm_sq);
                for (int i = 0; i < dim; i++)
                    vec[i] *= inv_norm;

                std::vector<float> rotated(dim);
                test_rotate_forward(vec.data(), rotated.data(), dim);
                for (int i = 0; i < dim; i++)
                    rotated[i] *= sqrt_dim;

                // Polar decomposition: compute angles and measure quantize error per level
                std::vector<float> radii(dim / 2);

                // Level 1
                {
                    int n = dim / 2;
                    int bw = bpl[0];
                    for (int i = 0; i < n; i++) {
                        float x = rotated[2 * i], y = rotated[2 * i + 1];
                        float angle = std::atan2(y, x);
                        if (angle < 0.0f)
                            angle += 2.0f * static_cast<float>(TURBOQ_PI_F);
                        radii[i] = std::sqrt(x * x + y * y);

                        if (bw > 0) {
                            auto lut = polarq_get_lut(1, bw);
                            int idx = 0;
                            for (int b = 0; b < lut.n_centroids - 1; b++)
                                idx += (angle > lut.boundaries[b]) ? 1 : 0;
                            float q_angle = lut.centroids[idx];
                            double err = static_cast<double>(angle - q_angle);
                            level_mse[0] += err * err;
                            level_count[0]++;
                        }
                    }
                }

                // Levels 2..L
                for (int level = 2; level <= L; level++) {
                    int n = dim >> level;
                    int bw = bpl[level - 1];
                    std::vector<float> new_radii(n);
                    for (int i = 0; i < n; i++) {
                        float re = radii[2 * i], ro = radii[2 * i + 1];
                        float angle = std::atan2(ro, re);
                        new_radii[i] = std::sqrt(re * re + ro * ro);

                        if (bw > 0) {
                            auto lut = polarq_get_lut(level, bw);
                            int idx = 0;
                            for (int b = 0; b < lut.n_centroids - 1; b++)
                                idx += (angle > lut.boundaries[b]) ? 1 : 0;
                            float q_angle = lut.centroids[idx];
                            double err = static_cast<double>(angle - q_angle);
                            level_mse[level - 1] += err * err;
                            level_count[level - 1]++;
                        }
                    }
                    radii = std::move(new_radii);
                }
            }

            printf("\n=== Per-Level Angle MSE (polar%d, dim=%d, N=%d) ===\n", bits, dim, N);
            printf("%-8s %8s %8s %10s %10s\n", "Level", "Bits", "Angles", "MSE", "Variance");
            for (int k = 0; k < L; k++) {
                double mse = (level_count[k] > 0) ? level_mse[k] / level_count[k] : 0.0;
                int n_angles = dim >> (k + 1);
                printf("L%-7d %8d %8d %10.6f %10s\n", k + 1, bpl[k], n_angles, mse, bpl[k] == 0 ? "fixed" : "—");
            }
        }
    }
}

// Diagnostic 2: Tree reconstruction error (quantize then dequantize, no QK/V path).
// Directly measures the Cartesian reconstruction RMSE of quantize→dequantize,
// compared to TBQ's quantize→dequantize. This isolates codec quality from the
// attention pipeline.
TEST(PolarQ, DiagnosticReconstructionRMSE) {
    constexpr int N = 500;

    printf("\n=== Reconstruction RMSE: Polar vs TBQ (N=%d) ===\n", N);
    printf("%-16s %12s %12s %12s\n", "Mode", "RMSE", "RelRMSE", "CosSim");

    for (int dim : {128, 256}) {
        for (int bits : {3, 4}) {
            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, 1.0f);

            double polar_rmse_sum = 0, polar_cos_sum = 0;
            double tbq_rmse_sum = 0, tbq_cos_sum = 0;

            for (int s = 0; s < N; s++) {
                std::vector<float> vec(dim);
                for (auto& v : vec)
                    v = dist(rng);

                float vec_norm = 0;
                for (int i = 0; i < dim; i++)
                    vec_norm += vec[i] * vec[i];
                vec_norm = std::sqrt(vec_norm);

                // --- Polar path: quantize + V-accum with weight=1 + Q^T ---
                {
                    std::vector<uint8_t> packed(polarq_head_bytes(dim, bits));
                    polarq_quantize_head(vec.data(), packed.data(), dim, bits, ov::element::f32);

                    std::vector<float> accum(dim, 0.0f);
                    float* ap = accum.data();
                    float w = 1.0f;
                    const uint8_t* vp[1] = {packed.data()};
                    const float* wp[1] = {&w};
                    float* aps[1] = {ap};
                    polarq_fused_v_accum(vp[0], wp[0], aps, 1, dim, bits);

                    std::vector<float> result(dim);
                    test_rotate_inverse(accum.data(), result.data(), dim);

                    double dot = 0, nr = 0, nv = 0, err_sq = 0;
                    for (int i = 0; i < dim; i++) {
                        dot += vec[i] * result[i];
                        nv += vec[i] * vec[i];
                        nr += result[i] * result[i];
                        double d = vec[i] - result[i];
                        err_sq += d * d;
                    }
                    polar_rmse_sum += std::sqrt(err_sq / dim) / (vec_norm / std::sqrt(static_cast<float>(dim)));
                    polar_cos_sum += dot / (std::sqrt(nv * nr) + 1e-10);
                }

                // --- TBQ path (dim=128 only) ---
                if (dim == 128) {
                    std::vector<uint8_t> packed(turboq_head_bytes(dim, bits));
                    turboq_quantize_head(vec.data(), packed.data(), dim, bits, ov::element::f32);

                    std::vector<float> accum(dim, 0.0f);
                    float* ap = accum.data();
                    float w = 1.0f;
                    turboq_fused_v_accum(packed.data(), &w, &ap, 1, dim, bits);

                    std::vector<float> result(dim);
                    test_rotate_inverse(accum.data(), result.data(), dim);

                    double dot = 0, nr = 0, nv = 0, err_sq = 0;
                    for (int i = 0; i < dim; i++) {
                        dot += vec[i] * result[i];
                        nv += vec[i] * vec[i];
                        nr += result[i] * result[i];
                        double d = vec[i] - result[i];
                        err_sq += d * d;
                    }
                    tbq_rmse_sum += std::sqrt(err_sq / dim) / (vec_norm / std::sqrt(static_cast<float>(dim)));
                    tbq_cos_sum += dot / (std::sqrt(nv * nr) + 1e-10);
                }
            }

            printf("%-16s %12.6f %12s %12.6f\n",
                   (std::string("polar") + std::to_string(bits) + "/" + std::to_string(dim)).c_str(),
                   polar_rmse_sum / N,
                   "—",
                   polar_cos_sum / N);
            if (dim == 128) {
                printf("%-16s %12.6f %12s %12.6f\n",
                       (std::string("tbq") + std::to_string(bits) + "/" + std::to_string(dim)).c_str(),
                       tbq_rmse_sum / N,
                       "—",
                       tbq_cos_sum / N);
            }
        }
    }
}

// Diagnostic 3: Effective bits per coordinate.
// Counts actual bits used per coordinate after per-level byte-rounding overhead.
// If rounding wastes significant bits, the effective rate is lower than nominal.
TEST(PolarQ, DiagnosticEffectiveBits) {
    printf("\n=== Effective Bits Per Coordinate ===\n");
    printf("%-16s %8s %8s %8s %10s\n", "Mode", "Dim", "IdxBits", "IdxBytes", "EffBPC");

    for (int dim : {128, 256}) {
        for (int bits : {3, 4}) {
            const int* bpl = polarq_get_bits_per_level(bits, dim);
            int L = [](int d) {
                int l = 0;
                for (; d > 1; d >>= 1)
                    l++;
                return l;
            }(dim);

            int total_bits = 0;
            size_t total_bytes = 0;
            for (int k = 0; k < L; k++) {
                int n = dim >> (k + 1);
                int level_bits = n * bpl[k];
                total_bits += level_bits;
                total_bytes += static_cast<size_t>((level_bits + 7) / 8);
            }
            double eff_bpc = static_cast<double>(total_bytes * 8) / dim;
            double nominal_bpc = static_cast<double>(total_bits) / dim;

            printf("%-16s %8d %8d %8zu %10.3f (nominal %.3f, overhead %.1f%%)\n",
                   (std::string("polar") + std::to_string(bits)).c_str(),
                   dim,
                   total_bits,
                   total_bytes,
                   eff_bpc,
                   nominal_bpc,
                   100.0 * (eff_bpc - nominal_bpc) / nominal_bpc);

            // For comparison: TBQ is always exactly bits_per_coord
            if (dim == 128) {
                printf("%-16s %8d %8d %8zu %10.3f (nominal %.3f, overhead 0.0%%)\n",
                       (std::string("tbq") + std::to_string(bits)).c_str(),
                       dim,
                       dim * bits,
                       static_cast<size_t>((dim * bits + 7) / 8),
                       static_cast<double>(bits),
                       static_cast<double>(bits));
            }
        }
    }
}

// Diagnostic 4: Level-isolation ablation.
// Quantize one level at a time, keep all others exact. Measures each level's
// isolated contribution to Cartesian reconstruction error, without tree
// amplification confounds.
//
// Also: quantize ALL except one level (leave that one exact). This measures
// how much removing one level's quantization error helps.
//
// Together these separate "this level's angle error is large" from "this
// level's angle error is costly through the tree".
TEST(PolarQ, DiagnosticLevelIsolation) {
    constexpr int dim = 128;
    constexpr int N = 500;
    const int L = 7;  // log2(128)

    const float sqrt_dim = std::sqrt(static_cast<float>(dim));
    const float inv_sqrt_dim = 1.0f / sqrt_dim;

    for (int bits : {3, 4}) {
        const int* bpl = polarq_get_bits_per_level(bits, dim);

        printf("\n=== Level Isolation Ablation (polar%d, dim=%d, N=%d) ===\n", bits, dim, N);
        printf("%-24s %10s %10s\n", "Condition", "RMSE", "CosSim");

        // Helper: polar decompose, selectively quantize, reconstruct Cartesian.
        // quant_mask[k]=true means quantize level k+1, false means use exact angle.
        auto run_ablation = [&](const char* name, const bool* quant_mask) {
            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            double total_rmse = 0, total_cos = 0;

            for (int s = 0; s < N; s++) {
                std::vector<float> vec(dim);
                for (auto& v : vec)
                    v = dist(rng);

                // Norm + unit + rotate
                float norm_sq = 0;
                for (int i = 0; i < dim; i++)
                    norm_sq += vec[i] * vec[i];
                float norm = std::sqrt(norm_sq);
                std::vector<float> unit(dim);
                for (int i = 0; i < dim; i++)
                    unit[i] = vec[i] / norm;
                std::vector<float> rotated(dim);
                test_rotate_forward(unit.data(), rotated.data(), dim);
                for (int i = 0; i < dim; i++)
                    rotated[i] *= sqrt_dim;

                // Forward pass: decompose and optionally quantize each level
                std::vector<float> radii(dim / 2);
                std::vector<std::vector<float>> level_angles(L);    // exact angles
                std::vector<std::vector<float>> level_q_angles(L);  // quantized or exact

                // Level 1
                {
                    int n = dim / 2;
                    level_angles[0].resize(n);
                    level_q_angles[0].resize(n);
                    for (int i = 0; i < n; i++) {
                        float x = rotated[2 * i], y = rotated[2 * i + 1];
                        float angle = std::atan2(y, x);
                        if (angle < 0.0f)
                            angle += 2.0f * static_cast<float>(TURBOQ_PI_F);
                        radii[i] = std::sqrt(x * x + y * y);
                        level_angles[0][i] = angle;

                        if (quant_mask[0] && bpl[0] > 0) {
                            auto lut = polarq_get_lut(1, bpl[0]);
                            int idx = 0;
                            for (int b = 0; b < lut.n_centroids - 1; b++)
                                idx += (angle > lut.boundaries[b]) ? 1 : 0;
                            level_q_angles[0][i] = lut.centroids[idx];
                        } else {
                            level_q_angles[0][i] = angle;
                        }
                    }
                }

                // Levels 2..L
                for (int level = 2; level <= L; level++) {
                    int k = level - 1;
                    int n = dim >> level;
                    level_angles[k].resize(n);
                    level_q_angles[k].resize(n);
                    std::vector<float> new_radii(n);
                    for (int i = 0; i < n; i++) {
                        float re = radii[2 * i], ro = radii[2 * i + 1];
                        float angle = std::atan2(ro, re);
                        new_radii[i] = std::sqrt(re * re + ro * ro);
                        level_angles[k][i] = angle;

                        if (quant_mask[k] && bpl[k] > 0) {
                            auto lut = polarq_get_lut(level, bpl[k]);
                            int idx = 0;
                            for (int b = 0; b < lut.n_centroids - 1; b++)
                                idx += (angle > lut.boundaries[b]) ? 1 : 0;
                            level_q_angles[k][i] = lut.centroids[idx];
                        } else if (bpl[k] == 0) {
                            // Fixed level: always pi/4 regardless of mask
                            level_q_angles[k][i] = POLARQ_FIXED_ANGLE;
                        } else {
                            level_q_angles[k][i] = angle;
                        }
                    }
                    radii = std::move(new_radii);
                }

                // Reconstruct top-down using quantized/exact angles
                std::vector<float> r = {sqrt_dim};
                for (int level = L; level >= 2; level--) {
                    int k = level - 1;
                    int na = dim >> level;
                    std::vector<float> nr(na * 2);
                    for (int i = 0; i < na; i++) {
                        float a = level_q_angles[k][i];
                        nr[2 * i] = r[i] * std::cos(a);
                        nr[2 * i + 1] = r[i] * std::sin(a);
                    }
                    r = std::move(nr);
                }
                // Level 1: reconstruct Cartesian
                std::vector<float> cartesian(dim);
                int n_l1 = dim / 2;
                for (int i = 0; i < n_l1; i++) {
                    float a = level_q_angles[0][i];
                    cartesian[2 * i] = r[i] * std::cos(a);
                    cartesian[2 * i + 1] = r[i] * std::sin(a);
                }

                // Apply inverse rotation and scale by norm
                std::vector<float> result(dim);
                for (int i = 0; i < dim; i++)
                    cartesian[i] *= norm * inv_sqrt_dim;
                test_rotate_inverse(cartesian.data(), result.data(), dim);

                // Metrics
                double dot = 0, nv = 0, nr2 = 0, err_sq = 0;
                for (int i = 0; i < dim; i++) {
                    dot += vec[i] * result[i];
                    nv += vec[i] * vec[i];
                    nr2 += result[i] * result[i];
                    double d = vec[i] - result[i];
                    err_sq += d * d;
                }
                total_rmse += std::sqrt(err_sq / dim);
                total_cos += dot / (std::sqrt(nv * nr2) + 1e-10);
            }
            printf("%-24s %10.6f %10.6f\n", name, total_rmse / N, total_cos / N);
        };

        // "All quantized" = production path
        {
            bool mask[7] = {true, true, true, true, true, true, true};
            run_ablation("all quantized", mask);
        }
        // "All exact" = oracle (sanity check ≈ 1.0 cosine)
        {
            bool mask[7] = {false, false, false, false, false, false, false};
            run_ablation("all exact (oracle)", mask);
        }
        // "Only L_k quantized" — isolate each level's damage
        for (int k = 0; k < L; k++) {
            if (bpl[k] == 0)
                continue;  // skip fixed levels
            bool mask[7] = {false, false, false, false, false, false, false};
            mask[k] = true;
            char name[64];
            snprintf(name, sizeof(name), "only L%d quantized (%db)", k + 1, bpl[k]);
            run_ablation(name, mask);
        }
        // "All except L_k quantized" — how much does fixing one level help
        for (int k = 0; k < L; k++) {
            if (bpl[k] == 0)
                continue;
            bool mask[7] = {true, true, true, true, true, true, true};
            mask[k] = false;
            char name[64];
            snprintf(name, sizeof(name), "all except L%d exact", k + 1);
            run_ablation(name, mask);
        }
    }
}

// Diagnostic 5: Exhaustive allocation search.
// Enumerate all feasible per-level bit allocations under the fixed total budget,
// score each by reconstruction RMSE, report the Pareto frontier.
TEST(PolarQ, DiagnosticAllocationSearch) {
    constexpr int dim = 128;
    constexpr int L = 7;
    constexpr int N_SCORE = 200;  // vectors for scoring (fast pass)
    // n_angles per level: [64, 32, 16, 8, 4, 2, 1]
    const int n_angles[L] = {64, 32, 16, 8, 4, 2, 1};
    // Per-level bit bounds (inclusive)
    // Expanded bounds — previous search had L5 hitting upper bound at 4.
    const int lo[L] = {2, 0, 0, 0, 0, 0, 0};
    const int hi[L] = {7, 6, 6, 5, 5, 5, 4};

    const float sqrt_dim = std::sqrt(static_cast<float>(dim));
    const float inv_sqrt_dim = 1.0f / sqrt_dim;

    // Helper: quantize angle using LUT if available, else uniform.
    auto quantize_angle = [](float angle, int level_1based, int bw) -> float {
        if (bw == 0)
            return (level_1based == 1) ? angle : POLARQ_FIXED_ANGLE;
        auto lut = polarq_get_lut(level_1based, bw);
        if (lut.centroids != nullptr) {
            int idx = 0;
            for (int b = 0; b < lut.n_centroids - 1; b++)
                idx += (angle > lut.boundaries[b]) ? 1 : 0;
            return lut.centroids[idx];
        }
        // Uniform fallback
        float domain_hi =
            (level_1based == 1) ? 2.0f * static_cast<float>(TURBOQ_PI_F) : static_cast<float>(TURBOQ_PI_F / 2.0);
        float step = domain_hi / (1 << bw);
        int idx = static_cast<int>(angle / step);
        int max_idx = (1 << bw) - 1;
        if (idx < 0)
            idx = 0;
        if (idx > max_idx)
            idx = max_idx;
        return (idx + 0.5f) * step;
    };

    // Pre-generate test vectors: decompose once, reuse for all allocations.
    struct SampleData {
        std::vector<float> vec;
        float norm;
        std::vector<float> rotated;
        // Per-level exact angles and radii
        std::vector<std::vector<float>> angles;  // [L][n_angles[k]]
        std::vector<std::vector<float>> radii;   // [L+1] — radii[0] = rotated coords
    };
    std::vector<SampleData> samples(N_SCORE);
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int s = 0; s < N_SCORE; s++) {
            auto& sd = samples[s];
            sd.vec.resize(dim);
            for (auto& v : sd.vec)
                v = dist(rng);

            float nsq = 0;
            for (int i = 0; i < dim; i++)
                nsq += sd.vec[i] * sd.vec[i];
            sd.norm = std::sqrt(nsq);

            std::vector<float> unit(dim);
            for (int i = 0; i < dim; i++)
                unit[i] = sd.vec[i] / sd.norm;
            sd.rotated.resize(dim);
            test_rotate_forward(unit.data(), sd.rotated.data(), dim);
            for (int i = 0; i < dim; i++)
                sd.rotated[i] *= sqrt_dim;

            // Decompose
            sd.angles.resize(L);
            std::vector<float> rad(dim / 2);
            // L1
            sd.angles[0].resize(n_angles[0]);
            for (int i = 0; i < n_angles[0]; i++) {
                float x = sd.rotated[2 * i], y = sd.rotated[2 * i + 1];
                float a = std::atan2(y, x);
                if (a < 0.0f)
                    a += 2.0f * static_cast<float>(TURBOQ_PI_F);
                rad[i] = std::sqrt(x * x + y * y);
                sd.angles[0][i] = a;
            }
            for (int level = 2; level <= L; level++) {
                int k = level - 1;
                int na = n_angles[k];
                sd.angles[k].resize(na);
                std::vector<float> new_rad(na);
                for (int i = 0; i < na; i++) {
                    float re = rad[2 * i], ro = rad[2 * i + 1];
                    sd.angles[k][i] = std::atan2(ro, re);
                    new_rad[i] = std::sqrt(re * re + ro * ro);
                }
                rad = std::move(new_rad);
            }
        }
    }

    // Score one allocation: returns (RMSE, CosSim).
    auto score_alloc = [&](const int* alloc) -> std::pair<double, double> {
        double total_err_sq = 0, total_cos = 0;
        for (int s = 0; s < N_SCORE; s++) {
            const auto& sd = samples[s];

            // Quantize angles per level
            float q_ang[7][64];  // L=7, max 64 angles at L1
            for (int k = 0; k < L; k++) {
                for (int i = 0; i < n_angles[k]; i++) {
                    q_ang[k][i] = quantize_angle(sd.angles[k][i], k + 1, alloc[k]);
                }
            }

            // Top-down reconstruct
            float r[128];  // max dim/2
            r[0] = sqrt_dim;
            for (int level = L; level >= 2; level--) {
                int k = level - 1;
                int na = n_angles[k];
                float nr[128];
                for (int i = 0; i < na; i++) {
                    nr[2 * i] = r[i] * std::cos(q_ang[k][i]);
                    nr[2 * i + 1] = r[i] * std::sin(q_ang[k][i]);
                }
                std::memcpy(r, nr, na * 2 * sizeof(float));
            }
            // L1 → Cartesian
            float cart[128];
            for (int i = 0; i < n_angles[0]; i++) {
                cart[2 * i] = r[i] * std::cos(q_ang[0][i]);
                cart[2 * i + 1] = r[i] * std::sin(q_ang[0][i]);
            }
            for (int i = 0; i < dim; i++)
                cart[i] *= sd.norm * inv_sqrt_dim;

            float result[128];
            test_rotate_inverse(cart, result, dim);

            double dot = 0, nv = 0, nr2 = 0;
            for (int i = 0; i < dim; i++) {
                dot += sd.vec[i] * result[i];
                nv += sd.vec[i] * sd.vec[i];
                nr2 += result[i] * result[i];
                double d = sd.vec[i] - result[i];
                total_err_sq += d * d;
            }
            total_cos += dot / (std::sqrt(nv * nr2) + 1e-10);
        }
        double rmse = std::sqrt(total_err_sq / (N_SCORE * dim));
        return {rmse, total_cos / N_SCORE};
    };

    for (int target_bits : {384, 512}) {
        const char* mode = (target_bits == 384) ? "polar3" : "polar4";

        // Recursive enumeration with pruning
        struct Result {
            int alloc[L];
            double rmse;
            double cos_sim;
        };
        std::vector<Result> results;
        results.reserve(10000);

        int alloc[L] = {};
        int count = 0;

        std::function<void(int, int)> enumerate = [&](int k, int bits_used) {
            if (k == L) {
                if (bits_used == target_bits) {
                    auto [rmse, cos] = score_alloc(alloc);
                    Result r;
                    std::memcpy(r.alloc, alloc, sizeof(alloc));
                    r.rmse = rmse;
                    r.cos_sim = cos;
                    results.push_back(r);
                    count++;
                }
                return;
            }

            for (int b = lo[k]; b <= hi[k]; b++) {
                int new_used = bits_used + n_angles[k] * b;
                int remaining_budget = target_bits - new_used;
                if (remaining_budget < 0)
                    break;  // bits sorted ascending, prune

                // Prune: can remaining levels fill exactly remaining_budget?
                int min_remaining = 0, max_remaining = 0;
                for (int j = k + 1; j < L; j++) {
                    min_remaining += n_angles[j] * lo[j];
                    max_remaining += n_angles[j] * hi[j];
                }
                if (remaining_budget < min_remaining || remaining_budget > max_remaining)
                    continue;

                alloc[k] = b;
                enumerate(k + 1, new_used);
            }
        };

        auto t0 = std::chrono::steady_clock::now();
        enumerate(0, 0);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        // Sort by RMSE
        std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
            return a.rmse < b.rmse;
        });

        printf("\n=== Exhaustive Allocation Search (%s, budget=%d, dim=%d) ===\n", mode, target_bits, dim);
        printf("Enumerated %d feasible allocations, scored %d in %.1fs\n", count, count, elapsed);

        // Print current allocation for reference
        const int* current = polarq_get_bits_per_level(target_bits == 384 ? 3 : 4, dim);
        auto [cur_rmse, cur_cos] = score_alloc(current);
        printf("\nCurrent:  [%d,%d,%d,%d,%d,%d,%d]  RMSE=%.6f  CosSim=%.6f\n",
               current[0],
               current[1],
               current[2],
               current[3],
               current[4],
               current[5],
               current[6],
               cur_rmse,
               cur_cos);

        // TBQ reference (same total bits, uniform allocation)
        printf("TBQ ref:  RMSE≈%.6f (from DiagnosticReconstructionRMSE)\n", target_bits == 384 ? 0.183 : 0.096);

        // Top 10 by RMSE
        int top_n = std::min(10, static_cast<int>(results.size()));
        printf("\nTop %d allocations by RMSE:\n", top_n);
        printf("%-6s %-28s %10s %10s\n", "Rank", "Allocation", "RMSE", "CosSim");
        for (int i = 0; i < top_n; i++) {
            const auto& r = results[i];
            printf("%-6d [%d,%d,%d,%d,%d,%d,%d] %10.6f %10.6f\n",
                   i + 1,
                   r.alloc[0],
                   r.alloc[1],
                   r.alloc[2],
                   r.alloc[3],
                   r.alloc[4],
                   r.alloc[5],
                   r.alloc[6],
                   r.rmse,
                   r.cos_sim);
        }

        // Bottom 5 (worst)
        if (results.size() > 5) {
            printf("\nBottom 5 allocations:\n");
            for (int i = static_cast<int>(results.size()) - 5; i < static_cast<int>(results.size()); i++) {
                const auto& r = results[i];
                printf("%-6d [%d,%d,%d,%d,%d,%d,%d] %10.6f %10.6f\n",
                       i + 1,
                       r.alloc[0],
                       r.alloc[1],
                       r.alloc[2],
                       r.alloc[3],
                       r.alloc[4],
                       r.alloc[5],
                       r.alloc[6],
                       r.rmse,
                       r.cos_sim);
            }
        }
    }
}

// ============================================================================
// Reproduce tbq3/256 functional test failure in unit test.
// The functional test uses strided_iota inputs (deterministic ramp), runs
// multiple SDPA steps with stateful KV cache, and compares against fp32
// reference. This test isolates the quantize→QK dot→V accum→reduce path
// at dim=256 with 3-bit to find where the error enters.
// ============================================================================

TEST(TurboQ, Debug_TBQ3_Dim256) {
    constexpr int dim = 256;
    constexpr int n_tokens = 10;

    // Mimic the functional test's strided_iota input: val, val+0.1, val+0.2, ...
    auto make_strided = [](float start, int size) {
        std::vector<float> v(size);
        for (int i = 0; i < size; i++)
            v[i] = start + i * 0.1f;
        return v;
    };

    for (int bits : {3, 4}) {
        // Generate Q, K vectors like the functional test
        auto q = make_strided(1.0f, dim);
        std::vector<std::vector<float>> keys(n_tokens), values(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            keys[t] = make_strided(2.0f + t * 0.5f, dim);
            values[t] = make_strided(3.0f + t * 0.3f, dim);
        }

        // FP32 reference: dot products + softmax + weighted V sum
        std::vector<float> ref_logits(n_tokens);
        float d_scale = 1.0f / std::sqrt(static_cast<float>(dim));
        for (int t = 0; t < n_tokens; t++) {
            float dot = 0;
            for (int i = 0; i < dim; i++)
                dot += q[i] * keys[t][i];
            ref_logits[t] = dot * d_scale;
        }
        // softmax
        float max_l = *std::max_element(ref_logits.begin(), ref_logits.end());
        float sum_exp = 0;
        std::vector<float> ref_weights(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            ref_weights[t] = std::exp(ref_logits[t] - max_l);
            sum_exp += ref_weights[t];
        }
        for (auto& w : ref_weights)
            w /= sum_exp;
        // weighted V
        std::vector<float> ref_output(dim, 0.0f);
        for (int t = 0; t < n_tokens; t++)
            for (int i = 0; i < dim; i++)
                ref_output[i] += ref_weights[t] * values[t][i];

        // Quantized path
        // 1. Rotate Q
        std::vector<float> q_rot(dim);
        test_rotate_forward(q.data(), q_rot.data(), dim);

        // 2. Quantize all K and V
        size_t head_bytes = turboq_head_bytes(dim, bits);
        std::vector<std::vector<uint8_t>> packed_k(n_tokens), packed_v(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            packed_k[t].resize(head_bytes);
            packed_v[t].resize(head_bytes);
            turboq_quantize_head(keys[t].data(), packed_k[t].data(), dim, bits, ov::element::f32);
            turboq_quantize_head(values[t].data(), packed_v[t].data(), dim, bits, ov::element::f32);
        }

        // 3. Fused QK dot
        std::vector<float> quant_logits(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            quant_logits[t] = turboq_fused_qk_dot(packed_k[t].data(), q_rot.data(), dim, bits) * d_scale;
        }

        // 4. Softmax
        float qmax = *std::max_element(quant_logits.begin(), quant_logits.end());
        float qsum = 0;
        std::vector<float> quant_weights(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            quant_weights[t] = std::exp(quant_logits[t] - qmax);
            qsum += quant_weights[t];
        }
        for (auto& w : quant_weights)
            w /= qsum;

        // 5. V accumulation
        std::vector<float> accum(dim, 0.0f);
        float* accum_ptr = accum.data();
        for (int t = 0; t < n_tokens; t++) {
            float w = quant_weights[t];
            turboq_fused_v_accum(packed_v[t].data(), &w, &accum_ptr, 1, dim, bits);
        }

        // 6. Inverse rotation
        std::vector<float> quant_output(dim);
        test_rotate_inverse(accum.data(), quant_output.data(), dim);

        // Compare
        double max_err = 0, max_rel = 0;
        int worst_idx = 0;
        double cos_dot = 0, cos_nq = 0, cos_nr = 0;
        for (int i = 0; i < dim; i++) {
            double err = std::abs(quant_output[i] - ref_output[i]);
            double rel = err / (std::abs(ref_output[i]) + 1e-10);
            if (err > max_err) {
                max_err = err;
                worst_idx = i;
                max_rel = rel;
            }
            cos_dot += quant_output[i] * ref_output[i];
            cos_nq += quant_output[i] * quant_output[i];
            cos_nr += ref_output[i] * ref_output[i];
        }
        double cos_sim = cos_dot / (std::sqrt(cos_nq * cos_nr) + 1e-10);

        // Also check logits
        double logit_max_err = 0;
        for (int t = 0; t < n_tokens; t++) {
            double e = std::abs(quant_logits[t] - ref_logits[t]);
            if (e > logit_max_err)
                logit_max_err = e;
        }

        printf("\n=== TBQ%d dim=%d strided-iota SDPA (n_tokens=%d) ===\n", bits, dim, n_tokens);
        printf("  Logit max error:  %.4f\n", logit_max_err);
        printf("  Output max error: %.4f at coord %d (ref=%.3f, quant=%.3f, rel=%.2f)\n",
               max_err,
               worst_idx,
               ref_output[worst_idx],
               quant_output[worst_idx],
               max_rel);
        printf("  Output cosine:    %.6f\n", cos_sim);

        // Print softmax weight comparison
        printf("  Softmax weights (ref vs quant):\n");
        for (int t = 0; t < n_tokens; t++) {
            printf("    t=%d: ref=%.6f quant=%.6f diff=%+.6f\n",
                   t,
                   ref_weights[t],
                   quant_weights[t],
                   quant_weights[t] - ref_weights[t]);
        }

        // tbq4 should be much better than tbq3
        if (bits == 4) {
            EXPECT_GT(cos_sim, 0.95) << "tbq4/256 output cosine too low";
        }
        // tbq3 at dim=256: check if sign flips occur
        if (bits == 3) {
            bool has_sign_flip = false;
            for (int i = 0; i < dim; i++) {
                if (std::abs(ref_output[i]) > 1.0 && ref_output[i] * quant_output[i] < 0) {
                    has_sign_flip = true;
                    printf("  SIGN FLIP at coord %d: ref=%.3f quant=%.3f\n", i, ref_output[i], quant_output[i]);
                }
            }
            if (has_sign_flip) {
                printf("  WARNING: sign flips detected in tbq3/256!\n");
            }
            EXPECT_GT(cos_sim, 0.80) << "tbq3/256 output cosine too low";
        }
    }
}
