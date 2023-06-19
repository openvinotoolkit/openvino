// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>
#include "gtest/gtest.h"
#include "llm_mm.hpp"
#include "common/tensor2d.hpp"
#include "common/tensor2d_helper.hpp"
#include "rotary_kernel_avx512.hpp"
#include "test_common.hpp"

using namespace std;
using namespace llmdnn;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

using RotaryTestParamSet = std::tuple<
        data_type_t                                // data type
        >;

class RotaryTest : public TestWithParam<RotaryTestParamSet> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RotaryTestParamSet>& obj) {
        data_type_t types;
        std::tie(types) = obj.param;

        std::ostringstream result;
        result << dtype_to_str(types);
        return result.str();
    }

protected:
    virtual void SetUp() override {
        std::tie(_types) = GetParam();
    };

    template<typename T>
    static void rotary_emb(size_t rotaryNdims, float* cos, float* sin, T* q_src, T* k_src, T* q_dst, T* k_dst) {
        auto halfRotaryNdims = rotaryNdims / 2;
        for (size_t i = 0; i < halfRotaryNdims; i++) {
            q_dst[i] = q_src[i] * cos[i] - q_src[i + halfRotaryNdims] * sin[i];
            k_dst[i] = k_src[i] * cos[i] - k_src[i + halfRotaryNdims] * sin[i];
        }
        for (size_t i = halfRotaryNdims; i < rotaryNdims; i++) {
            q_dst[i] = q_src[i] * cos[i] + q_src[i - halfRotaryNdims] * sin[i];
            k_dst[i] = k_src[i] * cos[i] + k_src[i - halfRotaryNdims] * sin[i];
        }
    }

    template<typename T>
    void test(float thresh) {
        for (int n = 6; n < 129; n += 2) {
            tensor2D<T> q_src(1, n, true);
            tensor2D<T> k_src(1, n, true);
            tensor2D<T> q_dst(1, n, true);
            tensor2D<T> k_dst(1, n, true);
            tensor2D<T> q_dst_ref(1, n, true);
            tensor2D<T> k_dst_ref(1, n, true);
            tensor2D<float> cos(1, n, true);
            tensor2D<float> sin(1, n, true);
            for (int i = 0; i < n; i++) {
                q_src[i] = i % 19 - 10;
                k_src[i] = i % 19 - 9;
                cos[i] = i % 19 - 8;
                sin[i] = i % 19 - 7;
            }
            rotary_emb(n, cos.data, sin.data, q_src.data, k_src.data, q_dst_ref.data, k_dst_ref.data);
            rotary_avx512(n, cos.data, sin.data, q_src.data, k_src.data, q_dst.data, k_dst.data);
            for (int i = 0; i < n; i++) {
                float q = q_dst[i];
                float q_ref = q_dst_ref[i];
                float k = k_dst[i];
                float k_ref = k_dst_ref[i];
                if (std::abs(q - q_ref) > thresh) {
                    FAIL() << " q is not equal, N: " << n << " pos: " << i << " opt: " << q << " ref: " << q_ref;
                }
                if (std::abs(k - k_ref) > thresh) {
                    FAIL() << " k is not equal, N: " << n << " pos: " << i << " opt: " << k << " ref: " << k_ref;
                }
            }
        }
    }

    data_type_t _types;
};

TEST_P(RotaryTest, rotary) {
    if (_types == dnnl_s8) {
        ASSERT_TRUE(false);        
    } else {
        test<ov::bfloat16>(0.01f);
    }
}

const std::vector<data_type_t> types = {
    dnnl_bf16
};

INSTANTIATE_TEST_SUITE_P(smoke_Rotary, RotaryTest,
    ::testing::Combine(ValuesIn(types)),
    RotaryTest::getTestCaseName);
