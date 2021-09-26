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

struct NegativeParams {
    Tensor input;
    Tensor expected;
};

struct Builder : ParamsBuilder<NegativeParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceNegativeLayerTest : public testing::TestWithParam<NegativeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<NegativeParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::Parameter>(type, shape);
        const auto negative = std::make_shared<op::Negative>(in);
        return std::make_shared<Function>(NodeVector {negative}, ParameterVector {in});
    }
};

TEST_P(ReferenceNegativeLayerTest, NegativeWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Negative_With_Hardcoded_Refs, ReferenceNegativeLayerTest,
    ::testing::Values(Builder {}
                          .input({{6}, element::f16, std::vector<ngraph::float16> {1, -2, 0, -4.75f, 8.75f, -8.75f}})
                          .expected({{6}, element::f16, std::vector<ngraph::float16> {-1, 2, 0, 4.75f, -8.75f, 8.75f}}),
                      Builder {}
                          .input({{6}, element::f32, std::vector<float> {1, -2, 0, -4.75f, 8.75f, -8.75f}})
                          .expected({{6}, element::f32, std::vector<float>{-1, 2, 0, 4.75f, -8.75f, 8.75f}}),
                      Builder {}
                          .input({{10}, element::i32, std::vector<int32_t>{1, 8, -8, 17, -2, 1, 8, -8, 17, -1}})
                          .expected({{10}, element::i32, std::vector<int32_t>{-1, -8, 8, -17, 2, -1, -8, 8, -17, 1}}),
                      Builder {}
                          .input({{10}, element::i64, std::vector<int64_t>{1, 8, -8, 17, -2, 1, 8, -8, 17, -1}})
                          .expected({{10}, element::i64, std::vector<int64_t>{-1, -8, 8, -17, 2, -1, -8, 8, -17, 1}})),
    ReferenceNegativeLayerTest::getTestCaseName);
}  // namespace reference_tests
