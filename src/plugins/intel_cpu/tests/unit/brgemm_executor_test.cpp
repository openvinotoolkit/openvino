// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>
#include "common_test_utils/test_common.hpp"
#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace brgemmUnitTest {
class BrgemmKernelTest : public ov::test::TestsCommon, public testing::WithParamInterface<ov::element::Type> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::element::Type>& obj) {
        ov::element::Type rtPrec;
        rtPrec = obj.param;
        std::ostringstream result;
        result << "Prec=" << rtPrec.to_string() << std::endl;
        return result.str();
    }
};

template <typename T>
void run_test(ov::element::Type rtPrec) {
    size_t M = 33;
    size_t N = 32;
    size_t K = 33;
    ov::intel_cpu::BrgemmKernel gemm(M, N, K, K, N, N, false, rtPrec);
    size_t nthr = 8;
    bool is_f32 = (rtPrec == ov::element::f32);
    std::vector<T> a_data(M * K, (1.0f/33));
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
    ov::parallel_for(nthr, [&](size_t i){
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

TEST_P(BrgemmKernelTest, simpleGemmTest) {
    ov::element::Type rtPrec = this->GetParam();
    if (rtPrec == ov::element::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();
    if (rtPrec == ov::element::f32 && !ov::with_cpu_x86_avx512_core())
        GTEST_SKIP();
    if (rtPrec == ov::element::f16 && !ov::with_cpu_x86_avx512_core_fp16())
        GTEST_SKIP();

    if (rtPrec == ov::element::bf16) {
        run_test<ov::bfloat16>(rtPrec);
    } else if (rtPrec == ov::element::f16) {
        run_test<ov::float16>(rtPrec);
    } else {
        run_test<float>(rtPrec);
    }
}

INSTANTIATE_TEST_SUITE_P(BrgemmKernelUnitTest,
                         BrgemmKernelTest,
                         ::testing::Values(ov::element::f32, ov::element::bf16, ov::element::f16),
                         BrgemmKernelTest::getTestCaseName);
} // namespace brgemmUnitTest
