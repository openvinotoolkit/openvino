// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/atanh.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "base_reference_test.hpp"
#include "openvino/runtime/allocator.hpp"

using namespace ov;

namespace reference_tests {
namespace {

struct AtanhParams {
    Tensor input;
    Tensor expected;
};

struct Builder : ParamsBuilder<AtanhParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceAtanhLayerTest : public testing::TestWithParam<AtanhParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<AtanhParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;

        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::v0::Parameter>(type, shape);
        const auto out = std::make_shared<op::v3::Atanh>(in);
        return std::make_shared<ov::Function>(NodeVector{out}, ParameterVector{in});
    }
};

TEST_P(ReferenceAtanhLayerTest, AtanhWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Atanh_With_Hardcoded_Refs,
    ReferenceAtanhLayerTest,
    ::testing::Values(
        Builder{}
            .input({{8},
                    element::f16,
                    std::vector<ngraph::float16>{-INFINITY, -2.0f, -1.0f, -0.5f, 0.0f, 0.8f, 1.0f, INFINITY}})
            .expected(
                {{8},
                 element::f16,
                 std::vector<
                     ngraph::float16>{NAN, NAN, -INFINITY, -0.54930614f, 0.00000000f, 1.0986123f, INFINITY, NAN}}),
        Builder{}
            .input(
                {{2, 4}, element::f32, std::vector<float>{-INFINITY, -2.0f, -1.0f, -0.5f, 0.0f, 0.8f, 1.0f, INFINITY}})
            .expected({{2, 4},
                       element::f32,
                       std::vector<float>{NAN, NAN, -INFINITY, -0.54930614f, 0.00000000f, 1.0986123f, INFINITY, NAN}}),
        Builder{}
            .input({{6},
                    element::i32,
                    std::vector<int32_t>{std::numeric_limits<int32_t>::min(),
                                         -2,
                                         -1,
                                         1,
                                         2,
                                         std::numeric_limits<int32_t>::max()}})
            .expected({{6}, element::i32, std::vector<int32_t>{0, 0, -INFINITY, INFINITY, 0, 0}}),
        Builder{}
            .input({{2, 3},
                    element::u32,
                    std::vector<uint32_t>{std::numeric_limits<uint32_t>::min(),
                                          0,
                                          1,
                                          2,
                                          3,
                                          std::numeric_limits<uint32_t>::max()}})
            .expected({{2, 3}, element::u32, std::vector<uint32_t>{NAN, 0, INFINITY, NAN, NAN, NAN}}),
        Builder{}
            .input({{2, 3},
                    element::i64,
                    std::vector<int64_t>{std::numeric_limits<int64_t>::min(),
                                         -2,
                                         -1,
                                         1,
                                         2,
                                         std::numeric_limits<int64_t>::max()}})
            .expected({{2, 3}, element::i64, std::vector<int64_t>{NAN, NAN, -INFINITY, INFINITY, NAN, NAN}}),
        Builder{}
            .input({{2, 3},
                    element::u64,
                    std::vector<uint64_t>{std::numeric_limits<uint64_t>::min(),
                                          0,
                                          1,
                                          2,
                                          3,
                                          std::numeric_limits<uint64_t>::max()}})
            .expected({{2, 3}, element::u64, std::vector<uint64_t>{NAN, 0, INFINITY, NAN, NAN, NAN}})),
    ReferenceAtanhLayerTest::getTestCaseName);
}  // namespace reference_tests
