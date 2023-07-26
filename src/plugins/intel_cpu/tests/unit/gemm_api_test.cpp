// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>
#include "mlas/sgemm.hpp"

// This test is used to test whether mlas gemm lib compiles successfully
TEST(GemmTests, getPackedSize) {
    int N = 51864;
    int K = 384;
    ASSERT_NO_THROW(ov::intel_cpu::mlas_sgemm_pack_get_size(N, K));
}