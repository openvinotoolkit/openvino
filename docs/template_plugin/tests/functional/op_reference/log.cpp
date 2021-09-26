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

struct LogParams {
    Tensor input;
    Tensor expected;
};

struct Builder : ParamsBuilder<LogParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceLogLayerTest : public testing::TestWithParam<LogParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<LogParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::Parameter>(type, shape);
        const auto log = std::make_shared<op::Log>(in);
        return std::make_shared<Function>(NodeVector {log}, ParameterVector {in});
    }
};

TEST_P(ReferenceLogLayerTest, LogWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Log_With_Hardcoded_Refs, ReferenceLogLayerTest,
    ::testing::Values(Builder {}
                          .input({{8}, element::f16, std::vector<ngraph::float16> {0.125f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f, 16.f}})
                          .expected({{8}, element::f16, std::vector<ngraph::float16> {-2.07944154f, -1.38629436f, -0.69314718f, 0.00000000f, 0.69314718f, 1.38629436f, 2.07944154f, 2.77258872f}}),
                      Builder {}
                          .input({{8}, element::f32, std::vector<float> {0.125f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f, 16.f}})
                          .expected({{8}, element::f32, std::vector<float> {-2.07944154f, -1.38629436f, -0.69314718f, 0.00000000f, 0.69314718f, 1.38629436f, 2.07944154f, 2.77258872f}})),
    ReferenceLogLayerTest::getTestCaseName);
}  // namespace reference_tests
