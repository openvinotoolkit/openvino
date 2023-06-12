// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>
#include "gemm/ov_cpu_gemm.h"


TEST(GemmTests, getPackedSize) {
    int N = 51864;
    int K = 384;
    ASSERT_NO_THROW(ov_sgemm_pack_get_size("B", N, K));
}