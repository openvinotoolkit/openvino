// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vector>

#include "base_reference_test.hpp"

using namespace ngraph;

namespace reference_tests {
namespace {

struct SqrtParams {
    Tensor input;
    Tensor expected;
};

struct Builder : ParamsBuilder<SqrtParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceSqrtLayerTest : public testing::TestWithParam<SqrtParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SqrtParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::Parameter>(type, shape);
        const auto sqrt = std::make_shared<op::Sqrt>(in);
        return std::make_shared<Function>(NodeVector {sqrt}, ParameterVector {in});
    }
};

TEST_P(ReferenceSqrtLayerTest, SqrtWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Sqrt_Basic_With_Hardcoded_Refs, ReferenceSqrtLayerTest,
    ::testing::Values(Builder {}
                          .input({{6}, element::f16, std::vector<ngraph::float16> {16, 4, 81, 100, 10000, 0}})
                          .expected({{6}, element::f16, std::vector<ngraph::float16> {4, 2, 9, 10, 100, 0}}),
                      Builder {}
                          .input({{6}, element::f32, std::vector<float>{16, 4, 81, 100, 10000, 0}})
                          .expected({{6}, element::f32, std::vector<float>{4, 2, 9, 10, 100, 0}}),
                      Builder {}
                          .input({{6}, element::i32, std::vector<int32_t>{16, 4, 81, 100, 10000, 0}})
                          .expected({{6}, element::i32, std::vector<int32_t>{4, 2, 9, 10, 100, 0}}),
                      Builder {}
                          .input({{6}, element::i64, std::vector<int64_t>{16, 4, 81, 100, 10000, 0}})
                          .expected({{6}, element::i64, std::vector<int64_t>{4, 2, 9, 10, 100, 0}}),
                      Builder {}
                          .input({{6}, element::u32, std::vector<uint32_t>{16, 4, 81, 100, 10000, 0}})
                          .expected({{6}, element::u32, std::vector<uint32_t>{4, 2, 9, 10, 100, 0}}),
                      Builder {}
                          .input({{6}, element::u64, std::vector<uint64_t>{16, 4, 81, 100, 10000, 0}})
                          .expected({{6}, element::u64, std::vector<uint64_t>{4, 2, 9, 10, 100, 0}})),
    ReferenceSqrtLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Sqrt_Negative_With_Hardcoded_Refs,
    ReferenceSqrtLayerTest,
    ::testing::Values(
        Builder{}
            .input({{4}, element::f16, std::vector<ngraph::float16>{-1, 4, -81, 100}})
            .expected({{4}, element::f16, std::vector<ngraph::float16>{NAN, 2, NAN, 10}}),
        Builder{}
            .input({{4}, element::f32, std::vector<float>{-1, 4, -81, 100}})
            .expected({{4}, element::f32, std::vector<float>{NAN, 2, NAN, 10}}),
        Builder{}
            .input({{4}, element::i32, std::vector<int32_t>{-1, 4, -81, 100}})
            .expected({{4}, element::i32, std::vector<int32_t>{static_cast<int32_t>(NAN), 2, static_cast<int32_t>(NAN), 10}}),
        Builder{}
            .input({{4}, element::i64, std::vector<int64_t>{-1, 4, -81, 100}})
            .expected({{4}, element::i64, std::vector<int64_t>{static_cast<int64_t>(NAN), 2, static_cast<int64_t>(NAN), 10}})),
    ReferenceSqrtLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Sqrt_Integral_With_Hardcoded_Refs,
    ReferenceSqrtLayerTest,
    ::testing::Values(Builder{}
                          .input({{14}, element::f16, std::vector<ngraph::float16>{4, 7, 9, 10, 80, 55, 6.25, 0.9, 23.33, 233, 256, 473.7891, 1024, 111108.88}})
                          .expected({{14}, element::f16, std::vector<ngraph::float16>{2, 2.6457512, 3, 3.1622777, 8.944272, 7.4161983, 2.5, 0.94868326, 4.830114, 15.264338, 16., 21.766697, 32., 333.33}}),
                      Builder{}
                          .input({{14}, element::f32, std::vector<float>{4, 7, 9, 10, 80, 55, 6.25, 0.9, 23.33, 233, 256, 473.7891, 1024, 111108.88}})
                          .expected({{14}, element::f32, std::vector<float>{2, 2.6457512, 3, 3.1622777, 8.944272, 7.4161983, 2.5, 0.94868326, 4.830114, 15.264338, 16., 21.766697, 32., 333.33}}),
                      Builder{}
                          .input({{14}, element::i32, std::vector<int32_t>{4, 7, 9, 10, 80, 55, 6, 1, 23, 233, 256, 474, 1024, 110889}})
                          .expected({{14}, element::i32, std::vector<int32_t>{2, 3, 3, 3, 9, 7, 2, 1, 5, 15, 16, 22, 32, 333}}),
                      Builder{}
                          .input({{14}, element::i64, std::vector<int64_t>{4, 7, 9, 10, 80, 55, 6, 1, 23, 233, 256, 474, 1024, 110889}})
                          .expected({{14}, element::i64, std::vector<int64_t>{2, 3, 3, 3, 9, 7, 2, 1, 5, 15, 16, 22, 32, 333}}),
                      Builder{}
                          .input({{14}, element::u32, std::vector<uint32_t>{4, 7, 9, 10, 80, 55, 6, 1, 23, 233, 256, 474, 1024, 110889}})
                          .expected({{14}, element::u32, std::vector<uint32_t>{2, 3, 3, 3, 9, 7, 2, 1, 5, 15, 16, 22, 32, 333}}),
                      Builder{}
                          .input({{14}, element::u64, std::vector<uint64_t>{4, 7, 9, 10, 80, 55, 6, 1, 23, 233, 256, 474, 1024, 110889}})
                          .expected({{14}, element::u64, std::vector<uint64_t>{2, 3, 3, 3, 9, 7, 2, 1, 5, 15, 16, 22, 32, 333}})),
    ReferenceSqrtLayerTest::getTestCaseName);
}  // namespace reference_tests
