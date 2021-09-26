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

struct RoundParams {
    Tensor input;
    Tensor expected;
};

struct Builder : ParamsBuilder<RoundParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceRoundHalfToEvenLayerTest : public testing::TestWithParam<RoundParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RoundParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::Parameter>(type, shape);
        const auto round = std::make_shared<op::v5::Round>(in, op::v5::Round::RoundMode::HALF_TO_EVEN);
        return std::make_shared<Function>(NodeVector{round}, ParameterVector{in});
    }
};

class ReferenceRoundHalfAwayFromZeroLayerTest : public testing::TestWithParam<RoundParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RoundParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::Parameter>(type, shape);
        const auto round = std::make_shared<op::v5::Round>(in, op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
        return std::make_shared<Function>(NodeVector{round}, ParameterVector{in});
    }
};



TEST_P(ReferenceRoundHalfToEvenLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceRoundHalfAwayFromZeroLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Round_Half_To_Even_With_Hardcoded_Refs, 
    ReferenceRoundHalfToEvenLayerTest,
    ::testing::Values(Builder{}
                          .input({{5}, element::f16, std::vector<ngraph::float16>{0.9f, 2.5f, 2.3f, 1.5f, -4.5f}})
                          .expected({{5}, element::f16, std::vector<ngraph::float16>{1.0f, 2.0f, 2.0f, 2.0f, -4.0f}}),
                      Builder{}
                          .input({{5}, element::f32, std::vector<float>{0.9f, 2.5f, 2.3f, 1.5f, -4.5f}})
                          .expected({{5}, element::f32, std::vector<float>{1.0f, 2.0f, 2.0f, 2.0f, -4.0f}})),
    ReferenceRoundHalfToEvenLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Round_2D_With_Hardcoded_Refs,
    ReferenceRoundHalfToEvenLayerTest,
    ::testing::Values(Builder{}
                          .input({{15}, element::f16, std::vector<ngraph::float16>{0.1f, 0.5f, 0.9f, 1.2f, 1.5f, 1.8f, 2.3f, 2.5f, 2.7f, -1.1f, -1.5f, -1.9f, -2.2f, -2.5f, -2.8f}})
                          .expected({{15}, element::f16, std::vector<ngraph::float16>{0.f, 0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, -1.f, -2.f, -2.f, -2.f, -2.f, -3.f}}),
                      Builder{}
                          .input({{15}, element::f32, std::vector<float>{0.1f, 0.5f, 0.9f, 1.2f, 1.5f, 1.8f, 2.3f, 2.5f, 2.7f, -1.1f, -1.5f, -1.9f, -2.2f, -2.5f, -2.8f}})
                          .expected({{15}, element::f32, std::vector<float>{0.f, 0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, -1.f, -2.f, -2.f, -2.f, -2.f, -3.f}})),
    ReferenceRoundHalfToEvenLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Round_Int64_With_Hardcoded_Refs,
    ReferenceRoundHalfToEvenLayerTest,
    ::testing::Values(Builder{}
                          .input({{3}, element::i64, std::vector<int64_t>{0, 1, 0x4000000000000001}})
                          .expected({{3}, element::i64, std::vector<int64_t>{0, 1, 0x4000000000000001}})),
    ReferenceRoundHalfToEvenLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Round_Away_From_Zero_With_Hardcoded_Refs,
    ReferenceRoundHalfAwayFromZeroLayerTest,
    ::testing::Values(Builder{}
                          .input({{5}, element::f16, std::vector<ngraph::float16>{0.9f, 2.5f, 2.3f, 1.5f, -4.5f}})
                          .expected({{5}, element::f16, std::vector<ngraph::float16>{1.0f, 3.0f, 2.0f, 2.0f, -5.0f}}),
                      Builder{}
                          .input({{5}, element::f32, std::vector<float>{0.9f, 2.5f, 2.3f, 1.5f, -4.5f}})
                          .expected({{5}, element::f32, std::vector<float>{1.0f, 3.0f, 2.0f, 2.0f, -5.0f}})),
    ReferenceRoundHalfAwayFromZeroLayerTest::getTestCaseName);

}  // namespace reference_tests