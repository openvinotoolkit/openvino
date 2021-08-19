// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <tuple>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ngraph;
using namespace InferenceEngine;

namespace {
struct ScatterElementsUpdateParams {
    ScatterElementsUpdateParams(const Tensor& paramData,
                                const Tensor& paramIndices,
                                const Tensor& paramUpdates,
                                const Tensor& paramAxis,
                                const Tensor& paramExpected)
        : input(paramData),
          indices(paramIndices),
          updates(paramUpdates),
          axis(paramAxis),
          expected(paramExpected) {}

    Tensor input;
    Tensor indices;
    Tensor updates;
    Tensor axis;
    Tensor expected;
};

class ReferenceScatterElementsUpdateLayerTest : public testing::TestWithParam<ScatterElementsUpdateParams>,
                                       public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.input.data,
                     params.indices.data,
                     params.updates.data,
                     params.axis.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ScatterElementsUpdateParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "data_sh=" << param.input.shape;
        result << "_data_pr=" << param.input.type;
        result << "_indx_sh=" << param.indices.shape;
        result << "_indx_pr=" << param.indices.type;
        result << "_updt_sh=" << param.updates.shape;
        result << "_updt_pr=" << param.updates.type;
        result << "_axis_sh=" << param.axis.shape;
        result << "_axis_pr=" << param.axis.type;
        return result.str();
    }
private:
    static std::shared_ptr<Function> CreateFunction(const ScatterElementsUpdateParams& params) {
        const auto A = std::make_shared<op::Parameter>(params.input.type, params.input.shape);
        const auto B = std::make_shared<op::Parameter>(params.indices.type, params.indices.shape);
        const auto C = std::make_shared<op::Parameter>(params.updates.type, params.updates.shape);
        const auto D = std::make_shared<op::Parameter>(params.axis.type, params.axis.shape);
        auto scatterElts = std::make_shared<op::ScatterElementsUpdate>(A, B, C, D);
        return std::make_shared<Function>(NodeVector{scatterElts}, ParameterVector{A, B, C, D});
    }
};

TEST_P(ReferenceScatterElementsUpdateLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_ScatterEltsUpdate_With_Hardcoded_Refs,
    ReferenceScatterElementsUpdateLayerTest,
    ::testing::Values(
        // f16, i16
        ScatterElementsUpdateParams(Tensor({2, 2}, element::f16, std::vector<float16>{1, 2, 3, 4}),       // input
                                    Tensor({2, 2}, element::i16, std::vector<int16_t>{1, 1, 0, 0}),       // indices
                                    Tensor({2, 2}, element::f16, std::vector<float16>{10, 20, 30, 40}),   // updates
                                    Tensor({1}, element::i16, std::vector<int16_t>{0}),                   // axis
                                    Tensor({2, 2}, element::f16, std::vector<float16>{30, 40, 10, 20})),  // expected
        // f16, i32
        ScatterElementsUpdateParams(Tensor({2, 2}, element::f16, std::vector<float16>{1, 2, 3, 4}),       // input
                                    Tensor({2, 2}, element::i32, std::vector<int32_t>{1, 1, 0, 0}),       // indices
                                    Tensor({2, 2}, element::f16, std::vector<float16>{10, 20, 30, 40}),   // updates
                                    Tensor({1}, element::i32, std::vector<int32_t>{0}),                   // axis
                                    Tensor({2, 2}, element::f16, std::vector<float16>{30, 40, 10, 20})),  // expected
        // f32, i32
        ScatterElementsUpdateParams(Tensor({2, 2}, element::f32, std::vector<float>{1, 2, 3, 4}),       // input
                                    Tensor({2, 2}, element::i32, std::vector<int32_t>{1, 1, 0, 0}),     // indices
                                    Tensor({2, 2}, element::f32, std::vector<float>{10, 20, 30, 40}),   // updates
                                    Tensor({1}, element::i32, std::vector<int32_t>{0}),                 // axis
                                    Tensor({2, 2}, element::f32, std::vector<float>{30, 40, 10, 20})),  // expected
        // f32, i64
        ScatterElementsUpdateParams(Tensor({2, 2}, element::f32, std::vector<float>{1, 2, 3, 4}),       // input
                                    Tensor({2, 2}, element::i64, std::vector<int64_t>{1, 1, 0, 0}),     // indices
                                    Tensor({2, 2}, element::f32, std::vector<float>{10, 20, 30, 40}),   // updates
                                    Tensor({1}, element::i64, std::vector<int64_t>{0}),                 // axis
                                    Tensor({2, 2}, element::f32, std::vector<float>{30, 40, 10, 20})),  // expected
        // i16
        ScatterElementsUpdateParams(Tensor({2, 2}, element::i16, std::vector<int16_t>{1, 2, 3, 4}),       // input
                                    Tensor({2, 2}, element::i16, std::vector<int16_t>{1, 1, 0, 0}),       // indices
                                    Tensor({2, 2}, element::i16, std::vector<int16_t>{10, 20, 30, 40}),   // updates
                                    Tensor({1}, element::i16, std::vector<int16_t>{0}),                   // axis
                                    Tensor({2, 2}, element::i16, std::vector<int16_t>{30, 40, 10, 20})),  // expected
        // i32, axis=1
        ScatterElementsUpdateParams(Tensor({2, 2}, element::i32, std::vector<int32_t>{1, 2, 3, 4}),       // input
                                    Tensor({2, 2}, element::i32, std::vector<int32_t>{1, 1, 0, 0}),       // indices
                                    Tensor({2, 2}, element::i32, std::vector<int32_t>{10, 20, 30, 40}),   // updates
                                    Tensor({1}, element::i32, std::vector<int32_t>{1}),                   // axis
                                    Tensor({2, 2}, element::i32, std::vector<int32_t>{1, 20, 40, 4})),  // expected
        // i64
        ScatterElementsUpdateParams(Tensor({2, 2}, element::i64, std::vector<int64_t>{1, 2, 3, 4}),        // input
                                    Tensor({2, 2}, element::i64, std::vector<int64_t>{1, 1, 0, 0}),        // indices
                                    Tensor({2, 2}, element::i64, std::vector<int64_t>{10, 20, 30, 40}),    // updates
                                    Tensor({1}, element::i64, std::vector<int64_t>{0}),                    // axis
                                    Tensor({2, 2}, element::i64, std::vector<int64_t>{30, 40, 10, 20}))),  // expected
    ReferenceScatterElementsUpdateLayerTest::getTestCaseName);
} // namespace
