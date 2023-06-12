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
#include "test_common.h"

using namespace std;
using namespace llmdnn;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

using MMKernelTestShape = std::tuple<size_t, size_t, size_t>;
using MMKernelTestParamSet = std::tuple<
        std::pair<data_type_t, data_type_t>,         // a, b data type
        bool,                                        // b needs transpose
        MMKernelTestShape                            // M, N, K
        >;

class GemmKernelTest : public TestWithParam<MMKernelTestParamSet> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MMKernelTestParamSet>& obj) {
        std::pair<data_type_t, data_type_t> types;
        bool is_transpose;
        MMKernelTestShape shape;
        int M, N, K;
        std::tie(types, is_transpose, shape) = obj.param;
        std::tie(M, N, K) = shape;

        std::ostringstream result;
        result << "A_" << dtype_to_str(types.first) << "_B_" << dtype_to_str(types.second) << "_"
               << (is_transpose ? "transpose_" : "")
               << "M_" << M << "_N_" << N << "_K_" << K;
        return result.str();
    }

protected:
    virtual void SetUp() {
        initXTILE();

        MMKernelTestShape shape;
        std::tie(_types, _is_transpose, shape) = GetParam();
        std::tie(_M, _N, _K) = shape;
    };

    template<typename TA, typename TB>
    void test() {
        if (_N == 1 && (_is_transpose || _types.first == dnnl_u8)) {
            GTEST_SKIP() << "gemv does not support transpose or u8s8.";
        }
        mm_kernel* mm;
        mm_create_param param = {
            _types.first, _types.second,
            _N == 1, _is_transpose
        };
        ASSERT_TRUE(mm_kernel_create(&mm, &param));
        auto gemm = std::shared_ptr<mm_kernel>(mm, [](mm_kernel* p) { mm_kernel_destroy(p); });

        tensor2D<TA> A(_M, _K, true);
        tensor2D<TB> B(_K, _N, true);
        tensor2D<float> C(_M, _N, true);
        tensor2D<float> C_Ref(_M, _N, true);

        fill_rnd(A);
        fill_rnd(B);
        tensor2D<TB> BT = B.Tr();
        TB* ptr_B;
        size_t ldb;
        if (_is_transpose) {
            ptr_B = BT.data;
            ldb = BT.stride;
        } else {
            ptr_B = B.data;
            ldb = B.stride;
        }
        mm_kernel_execute(gemm.get(), A.data, ptr_B, C.data, A.stride, ldb,
            C.stride, _M, _N, _K);
        C_Ref = 0;
        matmul(A, B, C_Ref);
        ASSERT_TRUE(C_Ref == C);
    }

    int _M, _N, _K;
    std::pair<data_type_t, data_type_t> _types;
    bool _is_transpose;
};

TEST_P(GemmKernelTest, Func) {
    if (_types.first == dnnl_u8 && _types.second == dnnl_s8) {
        test<uint8_t, int8_t>();
    } else if (_types.first == dnnl_s8 && _types.second == dnnl_s8) {
        test<int8_t, int8_t>();
    } else {
        test<ov::bfloat16, ov::bfloat16>();
    }
}

const std::vector<std::pair<data_type_t, data_type_t>> types = {
    { dnnl_u8, dnnl_s8 },
    { dnnl_s8, dnnl_s8 },
    { dnnl_bf16, dnnl_bf16 },
};

// M, N, K
const std::vector<MMKernelTestShape> shapes = {
    // normal
    {256, 48, 448},
    // k tail
    {256, 48, 449},
    // M tail == unroll 8
    {256 + 8, 48, 449},
    // M tail == unroll 8 + 2
    {256 + 10, 48, 449},
    // N tail
    {256, 40, 448},
    // all tail
    {256 + 9, 47, 449},
    // gemv, K <= 64(32)*6
    {256, 1, 160},
};

INSTANTIATE_TEST_SUITE_P(smoke_GemmKernel, GemmKernelTest,
    ::testing::Combine(ValuesIn(types),
                       Values(true, false),
                       ValuesIn(shapes)),
    GemmKernelTest::getTestCaseName);
