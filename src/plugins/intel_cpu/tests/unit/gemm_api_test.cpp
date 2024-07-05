// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>
#include "mlas/sgemm.hpp"
#include "onednn/dnnl.h"
#include "cpu_memory.h"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "common_test_utils/test_assertions.hpp"

// This test is used to test whether mlas gemm lib compiles successfully
TEST(MLASGemmTests, getPackedSize) {
    int N = 51864;
    int K = 384;
    OV_ASSERT_NO_THROW(ov::intel_cpu::mlas_sgemm_pack_get_size(N, K));
}
// Test mlas thread partition with even/odd threads
TEST(MLASGemmTests, simpleGemm) {
    const auto L2cacheSize = dnnl::utils::get_cache_size(2, true);
    size_t M = 128;
    size_t K = 512;
    size_t N = L2cacheSize / sizeof(float) / (M);
    std::vector<float> aData(M * K, (1.0f/33));
    size_t bSize = ov::intel_cpu::mlas_sgemm_pack_get_size(N, K);
    size_t nthr = parallel_get_max_threads();
    auto alignedB = ov::AlignedBuffer(bSize, 64);
    float* bData = reinterpret_cast<float*>(alignedB.get_ptr());
    std::vector<float> cData(M * N, 0.0f);

    OV_ASSERT_NO_THROW(
        ov::intel_cpu::
            mlas_sgemm_compute("N", "T", M, N, K, 1.0f, aData.data(), K, bData, N, 0.0f, cData.data(), N, nullptr, nthr));

    OV_ASSERT_NO_THROW(
        ov::intel_cpu::
            mlas_sgemm_compute("N", "T", M, N, K, 1.0f, aData.data(), K, bData, N, 0.0f, cData.data(), N, nullptr, nthr - 1));
}