// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the OScaR codec scalar reference (oscar_quantize.hpp).
// These run on every architecture — pure scalar, no SIMD.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "nodes/kernels/scaled_attn/codecs/oscar_quantize.hpp"
#include "nodes/kernels/scaled_attn/codecs/oscar_stage.hpp"
#include "openvino/core/type/float16.hpp"

using namespace ov::Extensions::Cpu::XARCH;

namespace {

struct EncodedBlock {
    std::vector<uint8_t> payload;
    std::vector<ov::float16> deltas;
    std::vector<ov::float16> zps;
    std::vector<ov::float16> norms_q;
};

EncodedBlock make_buffers(int head_dim, bool with_norms) {
    EncodedBlock b;
    b.payload.assign(static_cast<size_t>(OSCAR_R) * head_dim / 4, 0);
    b.deltas.assign(static_cast<size_t>(OSCAR_SUBGROUPS) * head_dim, ov::float16(0.0F));
    b.zps.assign(static_cast<size_t>(OSCAR_SUBGROUPS) * head_dim, ov::float16(0.0F));
    if (with_norms) {
        b.norms_q.assign(OSCAR_R, ov::float16(0.0F));
    }
    return b;
}

// Build [R][head_dim] random unit vectors and matching norms.
void make_random_unit_vectors(int head_dim, unsigned seed,
                              std::vector<float>& unit, std::vector<float>& norms) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    unit.assign(static_cast<size_t>(OSCAR_R) * head_dim, 0.0F);
    norms.assign(OSCAR_R, 0.0F);
    for (int t = 0; t < OSCAR_R; ++t) {
        float* row = unit.data() + t * head_dim;
        float sumsq = 0.0F;
        for (int j = 0; j < head_dim; ++j) {
            row[j] = nd(rng);
            sumsq += row[j] * row[j];
        }
        const float n = std::sqrt(sumsq);
        for (int j = 0; j < head_dim; ++j) {
            row[j] /= n;
        }
        norms[t] = 0.5F + 0.1F * static_cast<float>(t);  // arbitrary positive norms
    }
}

}  // namespace

class OscarCodecRoundtrip : public ::testing::TestWithParam<int> {};

TEST_P(OscarCodecRoundtrip, EncodeDecodeBoundedError) {
    const int head_dim = GetParam();
    std::vector<float> unit, norms;
    make_random_unit_vectors(head_dim, /*seed=*/42u, unit, norms);

    auto buf = make_buffers(head_dim, /*with_norms=*/true);
    oscar_encode_block(unit.data(), norms.data(), head_dim,
                       buf.payload.data(), buf.deltas.data(), buf.zps.data(),
                       buf.norms_q.data());

    // Reconstruct each token via per-subgroup decode and compare.
    std::vector<float> recon(OSCAR_G * head_dim);
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;
    for (int g = 0; g < OSCAR_SUBGROUPS; ++g) {
        oscar_decode_subgroup(buf.payload.data(), buf.deltas.data(), buf.zps.data(),
                              head_dim, g, recon.data());
        for (int t = 0; t < OSCAR_G; ++t) {
            const float* ref = unit.data() + (g * OSCAR_G + t) * head_dim;
            const float* got = recon.data() + t * head_dim;
            for (int j = 0; j < head_dim; ++j) {
                const float e = ref[j] - got[j];
                sum_sq_err += static_cast<double>(e) * e;
                sum_sq_ref += static_cast<double>(ref[j]) * ref[j];
            }
        }
    }
    // INT2 + per-channel grouped over G=32 tokens: relative MSE bounded loosely.
    // Random unit vectors in head_dim=64..256 → empirically <0.6 with this scheme.
    const double rel_mse = sum_sq_err / sum_sq_ref;
    EXPECT_LT(rel_mse, 0.6) << "rel_mse=" << rel_mse << " head_dim=" << head_dim;
}

TEST_P(OscarCodecRoundtrip, NormsRoundtripFp16) {
    const int head_dim = GetParam();
    std::vector<float> unit, norms;
    make_random_unit_vectors(head_dim, /*seed=*/7u, unit, norms);

    auto buf = make_buffers(head_dim, /*with_norms=*/true);
    oscar_encode_block(unit.data(), norms.data(), head_dim,
                       buf.payload.data(), buf.deltas.data(), buf.zps.data(),
                       buf.norms_q.data());
    for (int t = 0; t < OSCAR_R; ++t) {
        const float got = static_cast<float>(buf.norms_q[t]);
        EXPECT_NEAR(got, norms[t], 1e-2F) << "t=" << t;
    }
}

TEST_P(OscarCodecRoundtrip, BlockBytesAccounting) {
    const int head_dim = GetParam();
    const size_t k_bytes = oscar_block_bytes(head_dim, /*with_norms=*/true);
    const size_t v_bytes = oscar_block_bytes(head_dim, /*with_norms=*/false);
    EXPECT_EQ(k_bytes - v_bytes, static_cast<size_t>(OSCAR_R) * sizeof(ov::float16));
    // payload + params for V side.
    const size_t expected_v = static_cast<size_t>(OSCAR_R) * head_dim / 4
                            + static_cast<size_t>(OSCAR_SUBGROUPS) * head_dim * 2 * sizeof(ov::float16);
    EXPECT_EQ(v_bytes, expected_v);
}

INSTANTIATE_TEST_SUITE_P(OscarCodec, OscarCodecRoundtrip, ::testing::Values(64, 128, 256));

TEST(OscarCodec, OutlierWidensSubgroupDelta) {
    // Channel j=0 outlier in only one token of sub-group 0; sub-group 1 lacks it.
    const int head_dim = 64;
    std::vector<float> unit(static_cast<size_t>(OSCAR_R) * head_dim, 0.0F);
    std::vector<float> norms(OSCAR_R, 1.0F);

    // Fill sub-groups 0 and 1 with small values, then poke an outlier into sg=0.
    std::mt19937 rng(123u);
    std::uniform_real_distribution<float> small(-0.05F, 0.05F);
    for (int t = 0; t < 2 * OSCAR_G; ++t) {
        for (int j = 0; j < head_dim; ++j) {
            unit[t * head_dim + j] = small(rng);
        }
    }
    unit[/*sg0 t=0 j=0*/ 0 * head_dim + 0] = 1.0F;

    auto buf = make_buffers(head_dim, /*with_norms=*/false);
    oscar_encode_block(unit.data(), nullptr, head_dim,
                       buf.payload.data(), buf.deltas.data(), buf.zps.data(),
                       /*norms_q=*/nullptr);

    const float delta_sg0_j0 = static_cast<float>(buf.deltas[0 * head_dim + 0]);
    const float delta_sg1_j0 = static_cast<float>(buf.deltas[1 * head_dim + 0]);
    EXPECT_GT(delta_sg0_j0, delta_sg1_j0 * 3.0F)
        << "outlier subgroup delta should be much wider; sg0=" << delta_sg0_j0
        << " sg1=" << delta_sg1_j0;
}

TEST(OscarStage, ResidualBelowRDoesNotFlush) {
    const int head_dim = 64;
    const size_t block_stride = oscar_block_bytes(head_dim, /*with_norms=*/true);
    std::vector<ov::float16> residual_unit(static_cast<size_t>(OSCAR_R) * head_dim, ov::float16(0.0F));
    std::vector<ov::float16> residual_norms(OSCAR_R, ov::float16(0.0F));
    std::vector<uint8_t> packed(block_stride * 4, 0xCD);
    std::vector<float> scratch(static_cast<size_t>(OSCAR_R) * head_dim, 0.0F);

    const size_t L1 = OSCAR_R - 1;
    std::vector<float> unit(L1 * head_dim, 0.0F);
    std::vector<float> norms(L1, 1.0F);
    for (size_t t = 0; t < L1; ++t) {
        unit[t * head_dim] = 1.0F;  // unit vector along axis 0
    }

    auto res = oscar_stage_and_flush(unit.data(), norms.data(), L1, head_dim,
                                     residual_unit.data(), residual_norms.data(),
                                     /*residual_count=*/0, packed.data(), block_stride,
                                     scratch.data());
    EXPECT_EQ(res.flushed_blocks, 0u);
    EXPECT_EQ(res.new_residual_count, L1);
    // Untouched packed buffer.
    EXPECT_EQ(packed[0], 0xCD);
}

TEST(OscarStage, ExactRTokensFlushesOnce) {
    const int head_dim = 64;
    const size_t block_stride = oscar_block_bytes(head_dim, /*with_norms=*/true);
    std::vector<ov::float16> residual_unit(static_cast<size_t>(OSCAR_R) * head_dim, ov::float16(0.0F));
    std::vector<ov::float16> residual_norms(OSCAR_R, ov::float16(0.0F));
    std::vector<uint8_t> packed(block_stride * 4, 0);
    std::vector<float> scratch(static_cast<size_t>(OSCAR_R) * head_dim, 0.0F);

    std::vector<float> unit, norms;
    make_random_unit_vectors(head_dim, /*seed=*/55u, unit, norms);

    auto res = oscar_stage_and_flush(unit.data(), norms.data(), OSCAR_R, head_dim,
                                     residual_unit.data(), residual_norms.data(),
                                     /*residual_count=*/0, packed.data(), block_stride,
                                     scratch.data());
    EXPECT_EQ(res.flushed_blocks, 1u);
    EXPECT_EQ(res.new_residual_count, 0u);
}

TEST(OscarStage, MultiBlockSplitFlush) {
    const int head_dim = 128;
    const size_t block_stride = oscar_block_bytes(head_dim, /*with_norms=*/true);
    std::vector<ov::float16> residual_unit(static_cast<size_t>(OSCAR_R) * head_dim, ov::float16(0.0F));
    std::vector<ov::float16> residual_norms(OSCAR_R, ov::float16(0.0F));
    std::vector<uint8_t> packed(block_stride * 4, 0);
    std::vector<float> scratch(static_cast<size_t>(OSCAR_R) * head_dim, 0.0F);

    // 3*R + 7 tokens → 3 flushes, 7 residual.
    const size_t L1 = 3 * OSCAR_R + 7;
    std::vector<float> unit(L1 * head_dim, 0.0F), norms(L1, 1.0F);
    std::mt19937 rng(11u);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    for (size_t t = 0; t < L1; ++t) {
        float* row = unit.data() + t * head_dim;
        float ss = 0.0F;
        for (int j = 0; j < head_dim; ++j) { row[j] = nd(rng); ss += row[j] * row[j]; }
        const float n = std::sqrt(ss);
        for (int j = 0; j < head_dim; ++j) row[j] /= n;
    }

    auto res = oscar_stage_and_flush(unit.data(), norms.data(), L1, head_dim,
                                     residual_unit.data(), residual_norms.data(),
                                     /*residual_count=*/0, packed.data(), block_stride,
                                     scratch.data());
    EXPECT_EQ(res.flushed_blocks, 3u);
    EXPECT_EQ(res.new_residual_count, 7u);
}

TEST(OscarStage, WarmStartConsumesResidueFirst) {
    const int head_dim = 64;
    const size_t block_stride = oscar_block_bytes(head_dim, /*with_norms=*/true);
    std::vector<ov::float16> residual_unit(static_cast<size_t>(OSCAR_R) * head_dim, ov::float16(0.0F));
    std::vector<ov::float16> residual_norms(OSCAR_R, ov::float16(0.0F));
    std::vector<uint8_t> packed(block_stride * 2, 0);
    std::vector<float> scratch(static_cast<size_t>(OSCAR_R) * head_dim, 0.0F);

    // Pre-populate residual_count=125. One token brings to 126; another fills.
    const size_t initial = OSCAR_R - 3;
    std::vector<float> unit(5 * head_dim, 0.0F), norms(5, 1.0F);
    for (size_t t = 0; t < 5; ++t) unit[t * head_dim] = 1.0F;

    auto res = oscar_stage_and_flush(unit.data(), norms.data(), 5, head_dim,
                                     residual_unit.data(), residual_norms.data(),
                                     initial, packed.data(), block_stride,
                                     scratch.data());
    // 125 + 5 = 130 → 1 flush, 2 residue.
    EXPECT_EQ(res.flushed_blocks, 1u);
    EXPECT_EQ(res.new_residual_count, 2u);
}

TEST(OscarCodec, Determinism) {
    const int head_dim = 128;
    std::vector<float> unit, norms;
    make_random_unit_vectors(head_dim, /*seed=*/99u, unit, norms);

    auto a = make_buffers(head_dim, /*with_norms=*/true);
    auto b = make_buffers(head_dim, /*with_norms=*/true);
    oscar_encode_block(unit.data(), norms.data(), head_dim,
                       a.payload.data(), a.deltas.data(), a.zps.data(), a.norms_q.data());
    oscar_encode_block(unit.data(), norms.data(), head_dim,
                       b.payload.data(), b.deltas.data(), b.zps.data(), b.norms_q.data());
    EXPECT_EQ(a.payload, b.payload);
    EXPECT_EQ(a.deltas, b.deltas);
    EXPECT_EQ(a.zps, b.zps);
    EXPECT_EQ(a.norms_q, b.norms_q);
}
