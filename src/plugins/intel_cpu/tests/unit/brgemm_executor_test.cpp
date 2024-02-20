// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>
#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/system_conf.hpp"

TEST(BrgemmKernel, simple_gemm_test) {
    if (!ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();
    size_t M = 33;
    size_t N = 32;
    size_t K = 33;
    ov::intel_cpu::BrgemmKernel gemm(M, N, K, K, N, N, false);
    std::vector<ov::bfloat16> a_data(M * K, (1.0f/33));
    std::vector<ov::bfloat16> b_data(K * N, 4.0f);
    size_t nthr = 8;
    std::vector<float> c_data(nthr * M * N, 0.0f);
    std::vector<size_t> wsp(nthr * 4 * 1024, 0.0f);
    std::vector<ov::bfloat16> b_scracth(gemm.get_scratch_b_size(), 0.0f);
    std::vector<ov::bfloat16> a_scracth(gemm.get_scratch_a_size(), 0.0f);

    gemm.copy_buffer_b(b_data.data(), b_scracth.data());
    auto m_block_size = gemm.get_mblk_size();
    auto m_blocks = (M + gemm.get_mblk_size() - 1) / m_block_size;
    ov::parallel_for2d(nthr, m_blocks, [&](size_t i, size_t m_blk) {
        auto m_start = m_blk * m_block_size;
        auto m_end = std::min(m_start + m_block_size, M);
        auto m_cnt = m_end - m_start;
        gemm.executeGemmPackedB(m_cnt < m_block_size,
                                a_data.data() + m_start * K,
                                b_scracth.data(),
                                c_data.data() + i * M * N + m_start * N,
                                wsp.data() + i * 4 * 1024,
                                a_scracth.data());
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