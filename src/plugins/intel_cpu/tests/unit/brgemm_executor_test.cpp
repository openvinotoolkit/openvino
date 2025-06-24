// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <utility>

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
        ov::element::Type rtPrec;
        bool postScale;
        size_t M, N, K;
        std::tie(rtPrec, M, N, K, postScale) = obj.param;
        std::ostringstream result;
        result << "Prec=" << rtPrec.to_string();
        result << ",M=" << M;
        result << ",N=" << N;
        result << ",K=" << K;
        result << ",WithpostScale=" << postScale << std::endl;
        return result.str();
    }
};

template <typename T>
void run_test(ov::element::Type rtPrec, size_t M, size_t N, size_t K) {
    M = 33;
    N = 32;
    K = 33;
    ov::intel_cpu::BrgemmKernel gemm(M, N, K, K, N, N, false, rtPrec);
    size_t nthr = 8;
    bool is_f32 = (rtPrec == ov::element::f32);
    std::vector<T> a_data(M * K, (1.0f / K));
    std::vector<T> b_data(K * N, 4.0f);
    std::vector<float> c_data(nthr * M * N, 0.0f);
    std::vector<size_t> wsp(nthr * 4 * 1024, 0.0f);
    std::vector<uint8_t> a_scratch(gemm.get_scratch_a_size(), 0.0f);
    std::vector<uint8_t> b_scratch(gemm.get_scratch_b_size(), 0.0f);
    if (!is_f32) {
        gemm.copy_buffer_b(b_data.data(), b_scratch.data());
    }
    auto m_block_size = gemm.get_mblk_size();
    auto m_blocks = (M + gemm.get_mblk_size() - 1) / m_block_size;
    void* b_ptr = !is_f32 ? static_cast<void*>(b_scratch.data()) : static_cast<void*>(b_data.data());
    ov::parallel_for2d(nthr, m_blocks, [&](size_t i, size_t m_blk) {
        auto m_start = m_blk * m_block_size;
        auto m_end = std::min(m_start + m_block_size, M);
        auto m_cnt = m_end - m_start;
        gemm.executeGemm(m_cnt < m_block_size,
                         a_data.data() + m_start * K,
                         b_ptr,
                         c_data.data() + i * M * N + m_start * N,
                         wsp.data() + i * 4 * 1024,
                         a_scratch.data());
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

template <>
void run_test<int8_t>(ov::element::Type rtPrec, size_t M, size_t N, size_t K) {
    // size_t M = 32;
    // size_t N = 32;
    // size_t K = 80;
    ov::intel_cpu::BrgemmKernel gemm(M, N, K, K + 4, K + 4, N, true, rtPrec);
    size_t nthr = 8;
    bool is_f32 = (rtPrec == ov::element::f32);
    std::vector<int8_t> a_data(M * (K + 4));
    std::vector<int8_t> b_data(N * (K + 4), 0);
    std::vector<int32_t> c_data(nthr * M * N, 0.0f);
    std::vector<size_t> wsp(nthr * 4 * 1024, 0.0f);
    std::vector<uint8_t> a_scratch(gemm.get_scratch_a_size(), 0.0f);
    std::vector<uint8_t> b_scratch(gemm.get_scratch_b_size(), 0.0f);
    for (size_t i = 0; i < M; i++) {
        std::fill_n(a_data.begin() + i * (K + 4), 4, 4096);
        std::iota(a_data.begin() + 4 + i * (K + 4), a_data.begin() + 4 + i * (K + 4) + K, 1);
    }
    for (size_t i = 0; i < N; i++) {
        std::fill_n(a_data.begin() + i * (K + 4), 4, 4096);
        std::fill_n(b_data.begin() + 4 + i * (K + 4), K, i + 1);
    }
    if (!is_f32) {
        gemm.copy_buffer_b(b_data.data() + 4, b_scratch.data());
    }
    auto m_block_size = gemm.get_mblk_size();
    auto m_blocks = (M + gemm.get_mblk_size() - 1) / m_block_size;
    void* b_ptr = !is_f32 ? static_cast<void*>(b_scratch.data()) : static_cast<void*>(b_data.data());
    ov::parallel_for2d(nthr, m_blocks, [&](size_t i, size_t m_blk) {
        auto m_start = m_blk * m_block_size;
        auto m_end = std::min(m_start + m_block_size, M);
        auto m_cnt = m_end - m_start;
        gemm.executeGemm(m_cnt < m_block_size,
                         a_data.data() + 4 + m_start * K,
                         b_ptr,
                         c_data.data() + i * M * N + m_start * N,
                         wsp.data() + i * 4 * 1024,
                         a_scratch.data());
    });
    ov::parallel_for(nthr, [&](size_t i) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                int32_t expected_value = (1 + K) * K / 2 * (n + 1);
                if (expected_value != c_data[i * M * N + m * N + n]) {
                    std::ostringstream out_stream;
                    out_stream << m << "|" << n << "|actual " << c_data[m * N + n] << "|expected|" << expected_value
                               << std::endl;
                    throw std::runtime_error(out_stream.str());
                }
            }
        }
    });
}

static void run_test_post_scales(ov::element::Type rtPrec, size_t M, size_t N, size_t K) {
    // size_t M = 32;
    // size_t N = 32;
    // size_t K = 80;
    ov::intel_cpu::BrgemmKernel gemm(M,
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
    bool is_f32 = (rtPrec == ov::element::f32);
    std::vector<int8_t> a_data(M * (K + 4));
    std::vector<int8_t> b_data(N * (K + 4), 0);
    std::vector<int32_t> c_data(nthr * M * N, 0.0f);
    std::vector<float> d_data(nthr * M * N, 0.0f);
    std::vector<float> b_scale(N, 2.0f);
    std::vector<size_t> wsp(nthr * 4 * 1024, 0.0f);
    std::vector<uint8_t> a_scratch(gemm.get_scratch_a_size(), 0.0f);
    std::vector<uint8_t> b_scratch(gemm.get_scratch_b_size(), 0.0f);
    for (size_t i = 0; i < M; i++) {
        std::fill_n(a_data.begin() + i * (K + 4), 4, 4096);
        std::iota(a_data.begin() + 4 + i * (K + 4), a_data.begin() + 4 + i * (K + 4) + K, 1);
    }
    for (size_t i = 0; i < N; i++) {
        std::fill_n(a_data.begin() + i * (K + 4), 4, 4096);
        std::fill_n(b_data.begin() + 4 + i * (K + 4), K, i + 1);
    }
    if (!is_f32) {
        gemm.copy_buffer_b(b_data.data() + 4, b_scratch.data());
    }
    auto m_block_size = gemm.get_mblk_size();
    auto m_blocks = (M + gemm.get_mblk_size() - 1) / m_block_size;
    void* b_ptr = !is_f32 ? static_cast<void*>(b_scratch.data()) : static_cast<void*>(b_data.data());
    ov::parallel_for2d(nthr, m_blocks, [&](size_t i, size_t m_blk) {
        auto m_start = m_blk * m_block_size;
        auto m_end = std::min(m_start + m_block_size, M);
        auto m_cnt = m_end - m_start;
        gemm.executeGemmWithScale(m_cnt < m_block_size,
                                  a_data.data() + 4 + m_start * K,
                                  b_ptr,
                                  c_data.data() + i * M * N + m_start * N,
                                  d_data.data() + i * M * N + m_start * N,
                                  b_scale.data(),
                                  wsp.data() + i * 4 * 1024,
                                  a_scratch.data());
    });

    ov::parallel_for(nthr, [&](size_t i) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float expected_value = (1 + K) * K / 2 * (n + 1) * 2.0f;
                if (expected_value != d_data[i * M * N + m * N + n]) {
                    std::ostringstream out_stream;
                    out_stream << m << "|" << n << "|actual " << d_data[m * N + n] << "|expected|" << expected_value
                               << std::endl;
                    throw std::runtime_error(out_stream.str());
                }
            }
        }
    });
}

TEST_P(BrgemmKernelTest, simpleGemmTest) {
    ov::element::Type rtPrec;
    bool postScale;
    size_t M, N, K;
    std::tie(rtPrec, M, N, K, postScale) = this->GetParam();
    if (rtPrec == ov::element::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();
    if (rtPrec == ov::element::f32 && !ov::with_cpu_x86_avx512_core())
        GTEST_SKIP();
    if (rtPrec == ov::element::f16 && !ov::with_cpu_x86_avx512_core_fp16())
        GTEST_SKIP();
    // TODO enable vnni2 if vnni2 flag available
    if (rtPrec == ov::element::i8 && !(ov::with_cpu_x86_avx512_core_amx_int8()))
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
                                                {ov::element::i8, 32, 32, 64, true}};

INSTANTIATE_TEST_SUITE_P(BrgemmKernelUnitTest,
                         BrgemmKernelTest,
                         ::testing::ValuesIn(params),
                         BrgemmKernelTest::getTestCaseName);
}  // namespace brgemmUnitTest