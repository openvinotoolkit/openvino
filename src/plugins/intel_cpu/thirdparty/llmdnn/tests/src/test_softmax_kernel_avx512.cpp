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
#include "tensor2d.hpp"
#include "tensor2d_helper.hpp"
#include "softmax_kernel_avx512.hpp"
#include "test_common.hpp"

using namespace std;
using namespace llmdnn;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

using SoftmaxTestParamSet = std::tuple<
        data_type_t                                // data type
        >;

class SoftmaxTest : public TestWithParam<SoftmaxTestParamSet> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SoftmaxTestParamSet>& obj) {
        data_type_t types;
        std::tie(types) = obj.param;

        std::ostringstream result;
        result << dtype_to_str(types);
        return result.str();
    }

protected:
    virtual void SetUp() {
        std::tie(_types) = GetParam();
    };

    template<typename T>
    static void gen_ref(tensor2D<float>& x, tensor2D<T>& out, tensor2D<float>& quant) {
        tensor2D<float> y = x.clone();
        float x_max = std::numeric_limits<float>::lowest();
        for(int i = 0; i < x.dims[1]; i++) {
            x_max = std::max(x_max, x[i]);
        }
        float sum = 0;
        for(int i = 0; i < x.dims[1]; i++) {
            y[i] = expf(x[i] - x_max);
            sum += y[i];
        }
        for(int i = 0; i < x.dims[1]; i++) {
            y[i] = y[i] / sum;
        }
        out.resize(x.dims[0], x.dims[1], true);
        if (std::is_same<T, float>::value) {
            memcpy(out.data, y.data, x.dims[0] * x.dims[1] * sizeof(float));
        }
        if (std::is_same<T, ov::bfloat16>::value) {
            for(int i = 0; i < x.dims[1]; i++) {
                out[i] = y[i];
            }
        }
#define CLIP(x, low, high) \
        (x < low ? low : (x > high ? high : x))
        if (std::is_same<T, int8_t>::value) {
            for(int i = 0; i < x.dims[1]; i++) {
                auto tmp = y[i] * quant[i];
                out[i] = static_cast<int8_t>(CLIP(tmp, -128, 127));
            }
        }
        if (std::is_same<T, uint8_t>::value) {
            for(int i = 0; i < x.dims[1]; i++) {
                auto tmp = y[i] * quant[i];
                out[i] = static_cast<uint8_t>(CLIP(tmp, 0, 255));
            }
        }
#undef CLIP
    }

    template<typename T>
    void test(float thresh) {
        for (int n = 1; n < 129; n++) {
            tensor2D<float> A(1, n, true);
            tensor2D<float> quant(1, n, true);
            tensor2D<T> out(1, n, true), out_ref;
            for (int i = 0; i < n; i++) {
                A[i] = static_cast<float>(i) - n / 2;
            }
            quant = 128.f;
            gen_ref(A, out_ref, quant);
            llmdnn::softmax<T>(out.data, A.data, n, nullptr, nullptr, quant.data);
            for (int i = 0; i < n; i++) {
                float a = out[i];
                float b = out_ref[i];
                if (std::abs(a - b) > thresh) {
                    FAIL() << " N: " << n << " pos: " << i << " opt: " << a << " ref: " << b;
                }
            }
        }
    }

    data_type_t _types;
};

TEST_P(SoftmaxTest, Func) {
    if (_types == dnnl_s8) {
        test<int8_t>(1.1f);
    } else if (_types == dnnl_u8) {
        test<uint8_t>(1.1f);
    } else if (_types == dnnl_f32) {
        test<float>(0.00001f);
    } else {
        test<ov::bfloat16>(0.01f);
    }
}

const std::vector<data_type_t> types = {
    dnnl_s8, dnnl_bf16, dnnl_u8, dnnl_f32
};

INSTANTIATE_TEST_SUITE_P(smoke_Softmax, SoftmaxTest,
    ::testing::Combine(ValuesIn(types)),
    SoftmaxTest::getTestCaseName);
