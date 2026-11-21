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
#include "openvino/core/type/float16.hpp"
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
    if (rtPrec == ov::element::bf16 && !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16))
        GTEST_SKIP();
    if (rtPrec == ov::element::f16 && !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_fp16))
        GTEST_SKIP();
    if (rtPrec == ov::element::i8 && !(dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) ||
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

using BrgemmFP16Params = std::tuple<size_t, size_t, size_t, bool, bool, bool, bool, bool>;

class BrgemmFP16Test : public testing::TestWithParam<BrgemmFP16Params> {};

TEST_P(BrgemmFP16Test, PaddedStridesAndTails) {
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_fp16)) {
        GTEST_SKIP() << "FP16 BRGEMM requires AVX512-FP16.";
    }
    const auto [M, N, K, transposed, accumulate, float_b, float_a, typed_b] = GetParam();
    if ((float_b || float_a || typed_b) && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx_fp16)) {
        GTEST_SKIP() << "Mixed FP16/FP32 operands use the non-AMX path.";
    }
    const size_t lda = K + 5;
    const size_t ldb = transposed ? K + 7 : N + 7;
    const size_t ldc = N + 3;
    const size_t guard = 64;
    std::vector<ov::float16> a(M * lda, ov::float16(13.f));
    std::vector<float> a_float(a.size(), 13.f);
    std::vector<ov::float16> b((transposed ? N : K) * ldb, ov::float16(-17.f));
    std::vector<float> b_float(b.size(), -17.f);
    std::vector<float> c(guard + M * ldc + guard, 19.f);
    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
            a[m * lda + k] = ov::float16((static_cast<int>((m * 17 + k * 13) % 29) - 14) * 0.125f);
            a_float[m * lda + k] = static_cast<float>(a[m * lda + k]) + (k % 5) * 0.001f;
        }
    }
    for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < N; ++n) {
            b[transposed ? n * ldb + k : k * ldb + n] =
                ov::float16((static_cast<int>((k * 7 + n * 11) % 31) - 15) * 0.0625f);
            b_float[transposed ? n * ldb + k : k * ldb + n] =
                static_cast<float>(b[transposed ? n * ldb + k : k * ldb + n]) + (k % 3) * 0.001f;
        }
    }
    ov::intel_cpu::BrgemmKernel gemm(M,
                                     N,
                                     K,
                                     lda,
                                     ldb,
                                     ldc,
                                     transposed,
                                     ov::element::f16,
                                     accumulate,
                                     false,
                                     float_b ? ov::element::f32 : ov::element::f16,
                                     float_a ? ov::element::f32 : ov::element::f16,
                                     !typed_b);
    std::vector<uint8_t> packed_b(guard + gemm.get_scratch_b_size() + guard, 0xA5);
    std::vector<uint8_t> scratch_a(guard + gemm.get_scratch_a_size() + guard, 0xA5);
    std::vector<size_t> workspace(4 * 1024);
    gemm.copy_buffer_b(float_b ? static_cast<void*>(b_float.data()) : static_cast<void*>(b.data()),
                       packed_b.data() + guard);
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx_fp16)) {
        const size_t packed_ld = (N + 15) / 16 * 16;
        for (size_t k = 0; k < K; ++k) {
            for (size_t n = 0; n < packed_ld; ++n) {
                const auto offset = transposed ? n * ldb + k : k * ldb + n;
                if (typed_b && !float_b) {
                    const auto expected = n < N ? b[offset] : ov::float16(0.F);
                    ASSERT_EQ(reinterpret_cast<const ov::float16*>(packed_b.data() + guard)[k * packed_ld + n],
                              expected);
                } else {
                    const auto expected = n < N ? (float_b ? b_float[offset] : static_cast<float>(b[offset])) : 0.F;
                    ASSERT_EQ(reinterpret_cast<const float*>(packed_b.data() + guard)[k * packed_ld + n], expected);
                }
            }
        }
    }
    for (size_t step = 0; step < 2; ++step) {
        for (size_t m = 0; m < M; m += gemm.get_mblk_size()) {
            const auto count = std::min(gemm.get_mblk_size(), M - m);
            if (step == 1 && !float_a && !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx_fp16)) {
                gemm.copy_buffer_a(count < gemm.get_mblk_size(), a.data() + m * lda, scratch_a.data() + guard);
                for (size_t repeat = 0; repeat < 2; ++repeat) {
                    gemm.execute_with_packed_a(count < gemm.get_mblk_size(),
                                               scratch_a.data() + guard,
                                               packed_b.data() + guard,
                                               c.data() + guard + m * ldc,
                                               workspace.data());
                }
            } else {
                gemm.executeGemm(
                    count < gemm.get_mblk_size(),
                    float_a ? static_cast<void*>(a_float.data() + m * lda) : static_cast<void*>(a.data() + m * lda),
                    packed_b.data() + guard,
                    c.data() + guard + m * ldc,
                    nullptr,
                    nullptr,
                    workspace.data(),
                    scratch_a.data() + guard);
            }
        }
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                float dot = 0;
                for (size_t k = 0; k < K; ++k) {
                    const auto offset = transposed ? n * ldb + k : k * ldb + n;
                    dot += (float_a ? a_float[m * lda + k] : static_cast<float>(a[m * lda + k])) *
                           (float_b ? b_float[offset] : static_cast<float>(b[offset]));
                }
                const size_t executions =
                    step == 1 && !float_a && !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx_fp16)
                        ? 3
                        : step + 1;
                const float expected = accumulate ? 19.f + executions * dot : dot;
                ASSERT_NEAR(c[guard + m * ldc + n], expected, 1e-4f) << "step=" << step << " m=" << m << " n=" << n;
            }
            for (size_t n = N; n < ldc; ++n) {
                ASSERT_EQ(c[guard + m * ldc + n], 19.f);
            }
        }
    }
    for (size_t i = 0; i < guard; ++i) {
        ASSERT_EQ(c[i], 19.f);
        ASSERT_EQ(c[guard + M * ldc + i], 19.f);
        ASSERT_EQ(packed_b[i], 0xA5);
        ASSERT_EQ(packed_b[guard + gemm.get_scratch_b_size() + i], 0xA5);
        ASSERT_EQ(scratch_a[i], 0xA5);
        ASSERT_EQ(scratch_a[guard + gemm.get_scratch_a_size() + i], 0xA5);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_BrgemmFP16,
                         BrgemmFP16Test,
                         ::testing::Combine(::testing::Values(1, 33),
                                            ::testing::Values(17, 37),
                                            ::testing::Values(3, 16, 17, 35, 127, 129),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

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
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16)) {
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
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16)) {
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
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16)) {
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
    if (!(dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) ||
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
