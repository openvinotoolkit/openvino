// Copyright (C) 2018-2024 Intel Corporation
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
    reference_tests::Tensor input;
    reference_tests::Tensor expected;
};

struct Builder : ParamsBuilder<AtanhParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceAtanhLayerTest : public testing::TestWithParam<AtanhParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        legacy_compare = true;
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
    static std::shared_ptr<Model> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::v0::Parameter>(type, shape);
        const auto out = std::make_shared<op::v3::Atanh>(in);
        return std::make_shared<ov::Model>(NodeVector{out}, ParameterVector{in});
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
            .input({{5}, element::f16, std::vector<ov::float16>{-1.0f, -0.5f, 0.0f, 0.8f, 1.0f}})
            .expected({{5},
                       element::f16,
                       std::vector<ov::float16>{-INFINITY, -0.54930614f, 0.00000000f, 1.0986123f, INFINITY}}),
        Builder{}
            .input({{5}, element::f32, std::vector<float>{-1.0f, -0.5f, 0.0f, 0.8f, 1.0f}})
            .expected(
                {{5}, element::f32, std::vector<float>{-INFINITY, -0.54930614f, 0.00000000f, 1.0986123f, INFINITY}}),
        Builder{}
            .input({{3}, element::i32, std::vector<int32_t>{-1, 0, 1}})
            .expected(
                {{3},
                 element::i32,
                 std::vector<int32_t>{std::numeric_limits<int32_t>::min(), 0, std::numeric_limits<int32_t>::max()}}),
        Builder{}
            .input({{2}, element::u32, std::vector<uint32_t>{0, 1}})
            .expected({{2}, element::u32, std::vector<uint32_t>{0, std::numeric_limits<uint32_t>::max()}}),
        Builder{}
            .input({{3}, element::i64, std::vector<int64_t>{-1, 0, 1}})
            .expected({{3},
                       element::i64,
                       std::vector<int64_t>{
                           std::numeric_limits<int64_t>::min(),
                           0,
                           std::numeric_limits<int64_t>::max(),
                       }}),
        Builder{}
            .input({{2}, element::u64, std::vector<uint64_t>{0, 1}})
            .expected({{2}, element::u64, std::vector<uint64_t>{0, std::numeric_limits<uint64_t>::max()}})),
    ReferenceAtanhLayerTest::getTestCaseName);
}  // namespace reference_tests
