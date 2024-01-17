// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>
#include "mlas/sgemm.hpp"

// This test is used to test whether mlas gemm lib compiles successfully
TEST(MLASGemmTests, getPackedSize) {
    int N = 51864;
    int K = 384;
    ASSERT_NO_THROW(ov::intel_cpu::mlas_sgemm_pack_get_size(N, K));
}
// Test mlas thread partition with even/odd threads
TEST(MLASGemmTests, simpleGemm) {
    size_t M = 33;
    size_t N = 32;
    size_t K = 33;
    std::vector<float> a_data(M * K, (1.0f/33));
    std::vector<float> b_data(K * N, 4.0f);
    std::vector<float> c_data(M * N, 0.0f);
    ASSERT_NO_THROW(
        ov::intel_cpu::
            mlas_sgemm("N", "T", M, N, K, 1.0f, a_data.data(), K, b_data.data(), N, 0.0f, c_data.data(), N, 3));
    ASSERT_NO_THROW(
        ov::intel_cpu::
            mlas_sgemm("N", "T", M, N, K, 1.0f, a_data.data(), K, b_data.data(), N, 0.0f, c_data.data(), N, 4));
}