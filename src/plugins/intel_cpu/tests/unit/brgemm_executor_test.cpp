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
