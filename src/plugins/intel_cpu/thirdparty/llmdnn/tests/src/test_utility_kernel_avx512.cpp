// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <iostream>
#include "gtest/gtest.h"
#include "llm_mm.hpp"
#include "common/tensor2d.hpp"
#include "common/tensor2d_helper.hpp"
#include "utility_kernel_avx512.hpp"
#include "test_common.hpp"

using namespace std;
using namespace llmdnn;
using ::testing::Values;
using ::testing::ValuesIn;

TEST(smoke_Utility, muladd) {
    float normal_factor = 1.2f;
    for (size_t len = 1; len < 129; len++) {
        std::vector<float> x(len), x_out(len), bias(len), ref(len);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = -10.0f + i;
            bias[i] = -100.0f + i;
            ref[i] = x[i] * normal_factor + bias[i];
        }
        mul_add_f32_avx512(x_out.data(), x.data(), normal_factor, bias.data(), len);
        for (size_t i = 0; i < x.size(); i++) {
            ASSERT_TRUE(std::abs(x_out[i] - ref[i]) < 0.0001f) << " length: " << len << " pos: " << i << " cur: " << x[i] << " ref: " << ref[i];
        }
    }
}