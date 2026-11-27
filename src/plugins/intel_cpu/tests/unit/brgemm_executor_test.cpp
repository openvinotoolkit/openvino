// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "common_test_utils/test_common.hpp"
#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/system_conf.hpp"

using BrgemmKernelParams = std::tuple<ov::element::Type,
                                      size_t,  // M
                                      size_t,  // N
                                      size_t,  // K
                                      bool>;

namespace brgemmUnitTest {
class BrgemmKernelTest : public ov::test::TestsCommon, public testing::WithParamInterface<BrgemmKernelParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BrgemmKernelParams>& obj) {
        const auto& [rtPrec, M, N, K, postScale] = obj.param;
        std::ostringstream result;
        result << "Prec=" << rtPrec.to_string();
        result << ",M=" << M;
        result << ",N=" << N;
        result << ",K=" << K;
        result << ",WithpostScale=" << postScale;
        return result.str();
    }
};

template <typename T>
void run_test(ov::element::Type rtPrec, size_t M, size_t N, size_t K) {
    ov::intel_cpu::BrgemmKernel gemm(M, N, K, K, N, N, false, rtPrec);
    size_t nthr = 8;
    bool is_f32 = (rtPrec == ov::element::f32);
    std::vector<T> a_data(M * K, (1.0f / K));
    std::vector<T> b_data(K * N, 4.0f);
    std::vector<float> c_data(nthr * M * N, 0.0f);
    std::vector<uint8_t> b_scratch(gemm.get_scratch_b_size(), 0.0f);
    if (!is_f32) {
        gemm.copy_buffer_b(b_data.data(), b_scratch.data());
    }
    auto m_block_size = gemm.get_mblk_size();
    auto m_blocks = (M + gemm.get_mblk_size() - 1) / m_block_size;
    std::vector<size_t> wsp(nthr * m_blocks * 4 * 1024, 0);
    std::vector<std::vector<uint8_t>> a_scratch(nthr * m_blocks, std::vector<uint8_t>(gemm.get_scratch_a_size(), 0));
    void* b_ptr = !is_f32 ? static_cast<void*>(b_scratch.data()) : static_cast<void*>(b_data.data());
    ov::parallel_for2d(nthr, m_blocks, [&](size_t i, size_t m_blk) {
        auto m_start = m_blk * m_block_size;
        auto m_end = std::min(m_start + m_block_size, M);
        auto m_cnt = m_end - m_start;
        gemm.executeGemm(m_cnt < m_block_size,
                         a_data.data() + m_start * K,
                         b_ptr,
                         c_data.data() + i * M * N + m_start * N,
                         nullptr,
                         nullptr,
                         wsp.data() + (i * m_blocks + m_blk) * 4 * 1024,
                         a_scratch[i * m_blocks + m_blk].data());
    });
    ov::parallel_for(nthr, [&](size_t i) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float expected_value = 4.0f;
                double abs = std::fabs(expected_value - c_data[i * M * N + m * N + n]);
                double rel = expected_value ? (abs / std::fabs(expected_value)) : abs;
                if (rel > 0.01f) {
                    std::ostringstream out_stream;
                    out_stream << "actual " << c_data[m * N + n] << "|expected|" << expected_value << std::endl;
                    throw std::runtime_error(out_stream.str());
                }
            }
        }
    });
}

static void fill_int8_matrices(size_t M, size_t N, size_t K, std::vector<int8_t>& a_data, std::vector<int8_t>& b_data) {
    // Nonzero padding detects incorrect row strides; operands cover the full signed-int8 range.
    std::fill(a_data.begin(), a_data.end(), 91);
    std::fill(b_data.begin(), b_data.end(), -73);
    for (size_t m = 0; m < M; m++) {
        for (size_t k = 0; k < K; k++) {
            a_data[4 + m * (K + 4) + k] = static_cast<int8_t>(static_cast<int32_t>((m * 17 + k * 13) % 256) - 128);
        }
    }
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k++) {
            b_data[4 + n * (K + 4) + k] = static_cast<int8_t>(static_cast<int32_t>((n * 29 + k * 7) % 256) - 128);
        }
    }
}

static int32_t reference_int8_gemm(const std::vector<int8_t>& a_data,
                                   const std::vector<int8_t>& b_data,
                                   size_t m,
                                   size_t n,
                                   size_t K) {
    int32_t result = 0;
    for (size_t k = 0; k < K; k++) {
        result += static_cast<int32_t>(a_data[4 + m * (K + 4) + k]) * b_data[4 + n * (K + 4) + k];
    }
    return result;
}

template <>
void run_test<int8_t>(ov::element::Type rtPrec, size_t M, size_t N, size_t K) {
    ov::intel_cpu::BrgemmKernel gemm(M, N, K, K + 4, K + 4, N, true, rtPrec);
    size_t nthr = 8;
    std::vector<int8_t> a_data(M * (K + 4));
    std::vector<int8_t> b_data(N * (K + 4), 0);
    std::vector<int32_t> c_data(nthr * M * N, 0.0f);
    std::vector<uint8_t> b_scratch(gemm.get_scratch_b_size(), 0.0f);
    fill_int8_matrices(M, N, K, a_data, b_data);
    gemm.copy_buffer_b(b_data.data() + 4, b_scratch.data());
    auto m_block_size = gemm.get_mblk_size();
    auto m_blocks = (M + gemm.get_mblk_size() - 1) / m_block_size;
    std::vector<size_t> wsp(nthr * m_blocks * 4 * 1024, 0);
    std::vector<std::vector<uint8_t>> a_scratch(nthr * m_blocks, std::vector<uint8_t>(gemm.get_scratch_a_size(), 0));
    ov::parallel_for2d(nthr, m_blocks, [&](size_t i, size_t m_blk) {
        auto m_start = m_blk * m_block_size;
        auto m_end = std::min(m_start + m_block_size, M);
        auto m_cnt = m_end - m_start;
        gemm.executeGemm(m_cnt < m_block_size,
                         a_data.data() + 4 + m_start * (K + 4),
                         b_scratch.data(),
                         c_data.data() + i * M * N + m_start * N,
                         nullptr,
                         nullptr,
                         wsp.data() + (i * m_blocks + m_blk) * 4 * 1024,
                         a_scratch[i * m_blocks + m_blk].data());
    });
    ov::parallel_for(nthr, [&](size_t i) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                int32_t expected_value = reference_int8_gemm(a_data, b_data, m, n, K);
                if (expected_value != c_data[i * M * N + m * N + n]) {
                    std::ostringstream out_stream;
                    out_stream << m << "|" << n << "|actual " << c_data[i * M * N + m * N + n] << "|expected|"
                               << expected_value << std::endl;
                    throw std::runtime_error(out_stream.str());
                }
            }
        }
    });
}

static void run_test_post_scales(ov::element::Type rtPrec, size_t M, size_t N, size_t K) {
    ov::intel_cpu::BrgemmKernelQuantized gemm(M,
                                              N,
                                              K,
                                              K + 4,
                                              K + 4,
                                              N,
                                              N,
                                              true,
                                              rtPrec,
                                              ov::element::f32,
                                              ov::intel_cpu::BrgemmKernel::ScaleType::PER_CHANNEL,
                                              false);
    size_t nthr = 8;
    std::vector<int8_t> a_data(M * (K + 4));
    std::vector<int8_t> b_data(N * (K + 4), 0);
    std::vector<int32_t> c_data(nthr * M * N, 0.0f);
    std::vector<float> d_data(nthr * M * N, 0.0f);
    std::vector<float> b_scale(N, 2.0f);
    std::vector<uint8_t> b_scratch(gemm.get_scratch_b_size(), 0.0f);
    fill_int8_matrices(M, N, K, a_data, b_data);
    for (size_t n = 0; n < N; n++) {
        b_scale[n] = static_cast<float>(n % 4 + 1) * 0.25f;
    }
    gemm.copy_buffer_b(b_data.data() + 4, b_scratch.data());
    auto m_block_size = gemm.get_mblk_size();
    auto m_blocks = (M + gemm.get_mblk_size() - 1) / m_block_size;
    std::vector<size_t> wsp(nthr * m_blocks * 4 * 1024, 0);
    std::vector<std::vector<uint8_t>> a_scratch(nthr * m_blocks, std::vector<uint8_t>(gemm.get_scratch_a_size(), 0));
    ov::parallel_for2d(nthr, m_blocks, [&](size_t i, size_t m_blk) {
        auto m_start = m_blk * m_block_size;
        auto m_end = std::min(m_start + m_block_size, M);
        auto m_cnt = m_end - m_start;
        gemm.executeGemm(m_cnt < m_block_size,
                         a_data.data() + 4 + m_start * (K + 4),
                         b_scratch.data(),
                         c_data.data() + i * M * N + m_start * N,
                         d_data.data() + i * M * N + m_start * N,
                         b_scale.data(),
                         wsp.data() + (i * m_blocks + m_blk) * 4 * 1024,
                         a_scratch[i * m_blocks + m_blk].data());
    });

    ov::parallel_for(nthr, [&](size_t i) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float expected_value = static_cast<float>(reference_int8_gemm(a_data, b_data, m, n, K)) * b_scale[n];
                if (expected_value != d_data[i * M * N + m * N + n]) {
                    std::ostringstream out_stream;
                    out_stream << m << "|" << n << "|actual " << d_data[i * M * N + m * N + n] << "|expected|"
                               << expected_value << std::endl;
                    throw std::runtime_error(out_stream.str());
                }
            }
        }
    });
}

TEST_P(BrgemmKernelTest, simpleGemmTest) {
    const auto& [rtPrec, M, N, K, postScale] = this->GetParam();
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        GTEST_SKIP() << "BRGEMM requires AVX2 or later.";
    }
    if (rtPrec == ov::element::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();
    if (rtPrec == ov::element::f16 && !ov::with_cpu_x86_avx512_core_fp16())
        GTEST_SKIP();
    if (rtPrec == ov::element::i8 && !(ov::with_cpu_x86_avx512_core_amx_int8() ||
                                       dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::cpu_isa_t::avx2_vnni_2)))
        GTEST_SKIP();

    if (rtPrec == ov::element::bf16) {
        run_test<ov::bfloat16>(rtPrec, M, N, K);
    } else if (rtPrec == ov::element::f16) {
        run_test<ov::float16>(rtPrec, M, N, K);
    } else if (rtPrec == ov::element::f32) {
        run_test<float>(rtPrec, M, N, K);
    } else {
        if (postScale) {
            run_test_post_scales(rtPrec, M, N, K);
        } else {
            run_test<int8_t>(rtPrec, M, N, K);
        }
    }
}

// AmxTileState: back-to-back executeGemm() calls that share a state skip the AMX tile configuration when the palette is
// the one configured last. The result must not depend on it, also when kernels with different palettes (other shapes,
// the M tail) alternate under one state.
class PreparedBf16Gemm {
public:
    PreparedBf16Gemm(size_t M, size_t N, size_t K)
        : M(M),
          N(N),
          K(K),
          gemm(M, N, K, K, N, N, false, ov::element::bf16),
          a_data(M * K),
          b_data(K * N),
          b_scratch(gemm.get_scratch_b_size(), 0),
          wsp(ov::intel_cpu::BrgemmKernel::get_wsp_size(), 0),
          a_scratch(gemm.get_scratch_a_size(), 0) {
        for (size_t i = 0; i < a_data.size(); i++) {
            a_data[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125F;
        }
        for (size_t i = 0; i < b_data.size(); i++) {
            b_data[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25F;
        }
        gemm.copy_buffer_b(b_data.data(), b_scratch.data());
    }

    [[nodiscard]] size_t blocks() const {
        return (M + gemm.get_mblk_size() - 1) / gemm.get_mblk_size();
    }

    // one M block into c (M x N, f32); the overload without a state when tiles is null
    void run_block(size_t blk, std::vector<float>& c, ov::intel_cpu::AmxTileState* tiles) {
        const size_t m_start = blk * gemm.get_mblk_size();
        const size_t m_cnt = std::min(gemm.get_mblk_size(), M - m_start);
        void* a = a_data.data() + m_start * K;
        void* out = c.data() + m_start * N;
        if (tiles != nullptr) {
            gemm.executeGemm(m_cnt < gemm.get_mblk_size(),
                             a,
                             b_scratch.data(),
                             out,
                             nullptr,
                             nullptr,
                             wsp.data(),
                             a_scratch.data(),
                             *tiles);
        } else {
            gemm.executeGemm(m_cnt < gemm.get_mblk_size(),
                             a,
                             b_scratch.data(),
                             out,
                             nullptr,
                             nullptr,
                             wsp.data(),
                             a_scratch.data());
        }
    }

    size_t M, N, K;
    ov::intel_cpu::BrgemmKernel gemm;
    std::vector<ov::bfloat16> a_data;
    std::vector<ov::bfloat16> b_data;
    std::vector<uint8_t> b_scratch;
    std::vector<uint8_t> wsp;
    std::vector<uint8_t> a_scratch;
};

static bool has_nonzero(const std::vector<float>& v) {
    return std::any_of(v.begin(), v.end(), [](float x) {
        return x != 0.0F;
    });
}

TEST(BrgemmKernelTileStateTest, sharedStateOverMBlocksMatchesPlainOverload) {
    if (!ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP() << "bf16 brgemm is not supported";
    }
    // 33 rows: a body block and a tail block (two palettes)
    PreparedBf16Gemm prepared(33, 32, 64);
    std::vector<float> expected(33 * 32, 0.0F);
    std::vector<float> actual(33 * 32, 0.0F);
    for (size_t blk = 0; blk < prepared.blocks(); blk++) {
        prepared.run_block(blk, expected, nullptr);
    }
    ov::intel_cpu::AmxTileState tiles;
    for (size_t repeat = 0; repeat < 3; repeat++) {
        std::fill(actual.begin(), actual.end(), 0.0F);
        for (size_t blk = 0; blk < prepared.blocks(); blk++) {
            prepared.run_block(blk, actual, &tiles);
        }
        ASSERT_TRUE(has_nonzero(expected));
        ASSERT_EQ(expected, actual);
    }
}

TEST(BrgemmKernelTileStateTest, sharedStateAcrossKernelsWithDifferentPalettes) {
    if (!ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP() << "bf16 brgemm is not supported";
    }
    // different K and N give different palettes (and K / N tails with K = 96, N = 48)
    PreparedBf16Gemm first(33, 32, 64);
    PreparedBf16Gemm second(40, 48, 96);
    PreparedBf16Gemm third(8, 16, 32);
    std::vector<float> expected_first(33 * 32, 0.0F);
    std::vector<float> expected_second(40 * 48, 0.0F);
    std::vector<float> expected_third(8 * 16, 0.0F);
    for (size_t blk = 0; blk < first.blocks(); blk++) {
        first.run_block(blk, expected_first, nullptr);
    }
    for (size_t blk = 0; blk < second.blocks(); blk++) {
        second.run_block(blk, expected_second, nullptr);
    }
    third.run_block(0, expected_third, nullptr);

    std::vector<float> actual_first(33 * 32, 0.0F);
    std::vector<float> actual_second(40 * 48, 0.0F);
    std::vector<float> actual_third(8 * 16, 0.0F);
    ov::intel_cpu::AmxTileState tiles;
    // interleave the kernels so that every call may find the palette of another kernel configured
    for (size_t blk = 0; blk < std::max(first.blocks(), second.blocks()); blk++) {
        if (blk < first.blocks()) {
            first.run_block(blk, actual_first, &tiles);
        }
        third.run_block(0, actual_third, &tiles);
        if (blk < second.blocks()) {
            second.run_block(blk, actual_second, &tiles);
        }
    }
    ASSERT_TRUE(has_nonzero(expected_first) && has_nonzero(expected_second) && has_nonzero(expected_third));
    EXPECT_EQ(expected_first, actual_first);
    EXPECT_EQ(expected_second, actual_second);
    EXPECT_EQ(expected_third, actual_third);
}

TEST(BrgemmKernelTileStateTest, freshStateAfterOtherAmxCode) {
    if (!ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP() << "bf16 brgemm is not supported";
    }
    // a new state is "unknown": it must configure even though the tiles hold another palette
    PreparedBf16Gemm first(32, 32, 64);
    PreparedBf16Gemm second(32, 48, 96);
    std::vector<float> expected(32 * 48, 0.0F);
    second.run_block(0, expected, nullptr);
    std::vector<float> scratch(32 * 32, 0.0F);
    std::vector<float> actual(32 * 48, 0.0F);
    {
        ov::intel_cpu::AmxTileState tiles;
        first.run_block(0, scratch, &tiles);
    }
    {
        ov::intel_cpu::AmxTileState tiles;
        second.run_block(0, actual, &tiles);
    }
    ASSERT_TRUE(has_nonzero(expected));
    EXPECT_EQ(expected, actual);
}

TEST(BrgemmKernelTileStateTest, quantizedKernelWithScalesSharesState) {
    if (!(ov::with_cpu_x86_avx512_core_amx_int8() ||
          dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::cpu_isa_t::avx2_vnni_2))) {
        GTEST_SKIP() << "int8 brgemm is not supported";
    }
    const size_t M = 65;
    const size_t N = 32;
    const size_t K = 64;
    ov::intel_cpu::BrgemmKernelQuantized gemm(M,
                                              N,
                                              K,
                                              K + 4,
                                              K + 4,
                                              N,
                                              N,
                                              true,
                                              ov::element::i8,
                                              ov::element::f32,
                                              ov::intel_cpu::BrgemmKernel::ScaleType::PER_CHANNEL,
                                              false);
    std::vector<int8_t> a_data(M * (K + 4));
    std::vector<int8_t> b_data(N * (K + 4), 0);
    std::vector<float> b_scale(N);
    for (size_t n = 0; n < N; n++) {
        b_scale[n] = static_cast<float>(n % 4 + 1) * 0.25F;
    }
    fill_int8_matrices(M, N, K, a_data, b_data);
    std::vector<uint8_t> b_scratch(gemm.get_scratch_b_size(), 0);
    gemm.copy_buffer_b(b_data.data() + 4, b_scratch.data());
    std::vector<uint8_t> wsp(ov::intel_cpu::BrgemmKernel::get_wsp_size(), 0);
    std::vector<uint8_t> a_scratch(gemm.get_scratch_a_size(), 0);
    const size_t m_block = gemm.get_mblk_size();
    const size_t m_blocks = (M + m_block - 1) / m_block;

    // the scaled path (d / scale_b given) and the path without scales (scale_b == nullptr)
    for (const bool with_scale : {true, false}) {
        std::vector<int32_t> c_plain(M * N, 0);
        std::vector<float> d_plain(M * N, 0.0F);
        std::vector<int32_t> c_state(M * N, 0);
        std::vector<float> d_state(M * N, 0.0F);
        ov::intel_cpu::AmxTileState tiles;
        for (size_t blk = 0; blk < m_blocks; blk++) {
            const size_t m_start = blk * m_block;
            const bool is_tail = std::min(m_block, M - m_start) < m_block;
            auto run = [&](std::vector<int32_t>& c, std::vector<float>& d, ov::intel_cpu::AmxTileState* state) {
                void* a = a_data.data() + 4 + m_start * (K + 4);
                void* c_ptr = c.data() + m_start * N;
                void* d_ptr = d.data() + m_start * N;
                float* scale = with_scale ? b_scale.data() : nullptr;
                if (state != nullptr) {
                    gemm.executeGemm(is_tail,
                                     a,
                                     b_scratch.data(),
                                     c_ptr,
                                     d_ptr,
                                     scale,
                                     wsp.data(),
                                     a_scratch.data(),
                                     *state);
                } else {
                    gemm.executeGemm(is_tail, a, b_scratch.data(), c_ptr, d_ptr, scale, wsp.data(), a_scratch.data());
                }
            };
            run(c_plain, d_plain, nullptr);
            run(c_state, d_state, &tiles);
        }
        EXPECT_EQ(c_plain, c_state);
        EXPECT_EQ(d_plain, d_state);
        EXPECT_TRUE(with_scale ? has_nonzero(d_plain) : std::any_of(c_plain.begin(), c_plain.end(), [](int32_t x) {
            return x != 0;
        }));
    }
}

const std::vector<BrgemmKernelParams> params = {{ov::element::f32, 33, 32, 33, false},
                                                {ov::element::bf16, 33, 32, 33, false},
                                                {ov::element::f16, 33, 32, 33, false},
                                                {ov::element::i8, 32, 32, 80, true},
                                                {ov::element::i8, 32, 32, 64, true},
                                                {ov::element::i8, 32, 32, 80, false},
                                                {ov::element::i8, 32, 32, 64, false},
                                                {ov::element::i8, 33, 35, 65, false},
                                                {ov::element::i8, 33, 35, 65, true},
                                                {ov::element::i8, 1, 1, 1, false},
                                                {ov::element::i8, 1, 1, 1, true},
                                                {ov::element::i8, 65, 17, 127, false},
                                                {ov::element::i8, 65, 17, 127, true}};

INSTANTIATE_TEST_SUITE_P(BrgemmKernelUnitTest,
                         BrgemmKernelTest,
                         ::testing::ValuesIn(params),
                         BrgemmKernelTest::getTestCaseName);
}  // namespace brgemmUnitTest
