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
#include "transpose_kernel_avx512.hpp"
#include "test_common.hpp"

using namespace std;
using namespace llmdnn;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

using TransposeTestParamSet = std::tuple<
        data_type_t                                // data type
        >;

class TransposeTest : public TestWithParam<TransposeTestParamSet> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeTestParamSet>& obj) {
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
    static void gen_ref(T* dst, float* src, size_t height, size_t width, size_t src_stride, size_t dst_stride, float* quant) {
        for (size_t j = 0; j < height; j++) {
            if (std::is_same<T, float>::value) {
                memcpy(static_cast<void*>(dst), src, width * sizeof(float));
            }
            if (std::is_same<T, ov::bfloat16>::value) {
                for(size_t i = 0; i < width; i++) {
                    dst[i] = src[i];
                }
            }
    #define CLIP(x, low, high) \
            (x < low ? low : (x > high ? high : x))
            if (std::is_same<T, int8_t>::value) {
                for(size_t i = 0; i < width; i++) {
                    auto tmp = src[i] * quant[i];
                    dst[i] = static_cast<int8_t>(CLIP(tmp, -128, 127));
                }
            }
            if (std::is_same<T, uint8_t>::value) {
                for(size_t i = 0; i < width; i++) {
                    auto tmp = src[i] * quant[i];
                    dst[i] = static_cast<uint8_t>(CLIP(tmp, 0, 255));
                }
            }
    #undef CLIP
            src = reinterpret_cast<float*>(reinterpret_cast<int8_t*>(src) + src_stride);
            dst = reinterpret_cast<T*>(reinterpret_cast<int8_t*>(dst) + dst_stride);
        }
    }

    template<typename T>
    void test(float thresh) {
        // [num_heads, query_seq_len, head_size] => [query_seq_len, num_heads * head_size]
        int num_heads = 2, query_seq_len = 10;
        for (int head_size = 1; head_size < 129; head_size++) {
            tensor2D<float> src(num_heads, head_size * query_seq_len, true);
            tensor2D<float> quant(1, head_size, true);
            tensor2D<T> dst(query_seq_len, num_heads * head_size, true);
            tensor2D<T> dst_ref(query_seq_len, num_heads * head_size, true);
            for (int i = 0; i < num_heads * head_size * query_seq_len; i++) {
                src[i] = i % 253 - 127;
            }
            quant = 1.28f;
            auto* dst_p = dst.data;
            auto* dst_p_ref = dst_ref.data;
            for (int i = 0; i < num_heads; i++) {
                auto* src_p = &src(i, 0);
                llmdnn::memcpy2d_stride_avx512(dst_p, src_p, query_seq_len, head_size,
                    head_size * sizeof(float), num_heads * head_size * sizeof(T), quant.data);
                gen_ref(dst_p_ref, src_p, query_seq_len, head_size,
                    head_size * sizeof(float), num_heads * head_size * sizeof(T), quant.data);                    
                dst_p += head_size;
                dst_p_ref += head_size;
            }
            for (int i = 0; i < num_heads * head_size * query_seq_len; i++) {
                float a = dst[i];
                float b = dst_ref[i];
                if (std::abs(a - b) > thresh) {
                    FAIL() << " N: " << head_size << " pos: " << i << " opt: " << a << " ref: " << b;
                }
            }
        }
    }

    data_type_t _types;
};

TEST_P(TransposeTest, memcpy2d) {
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

INSTANTIATE_TEST_SUITE_P(smoke_Transpose, TransposeTest,
    ::testing::Combine(ValuesIn(types)),
    TransposeTest::getTestCaseName);
