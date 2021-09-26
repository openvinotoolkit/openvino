// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include <single_layer_common.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/ops.hpp>
#include <ie_precision.hpp>
#include <legacy/ngraph_ops/power.hpp>
#include <debug.h>
#include "../gna_matcher.hpp"

typedef struct {
    std::string activationType;
    size_t input_shape;
    std::pair<float, float> range;
} ActivationCaseParam;

using ActivationCaseParam2 = std::tuple<InferenceEngine::Precision, ActivationCaseParam>;

class GNAActivationTest : public GNATest<>,
                          public testing::WithParamInterface<ActivationCaseParam2> {
 public:

    static std::string getTestName(const testing::TestParamInfo<ActivationCaseParam2>& params) {
        std::string test_name = std::string(std::get<0>(params.param).name()) + "_";
        test_name += std::get<1>(params.param).activationType + "_";
        test_name += std::to_string(std::get<1>(params.param).input_shape) + "_";
        test_name += std::to_string(std::get<1>(params.param).range.first) + "_";
        test_name += std::to_string(std::get<1>(params.param).range.second);
        return test_name;
    }

    std::shared_ptr<ngraph::Function> buildNgraphFunction(const ActivationCaseParam& param) {

        auto shape = ngraph::Shape{1, param.input_shape};
        auto inputN = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);

        auto absN = std::make_shared<ngraph::op::v0::Abs>(inputN);

        auto powerN = std::make_shared<ngraph::op::PowerIE>(absN, -1, 1, 1.0);

        auto eltwiseN = std::make_shared<ngraph::op::v1::Multiply>(powerN, inputN);

        auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{eltwiseN}, ngraph::ParameterVector{inputN});
        return function;
    }
};

TEST_P(GNAActivationTest, ActivationTest) {
    const auto precision = std::get<0>(GetParam());
    const auto param = std::get<1>(GetParam());

    if (precision == InferenceEngine::Precision::FP32) {
        auto input_data = generate_random_1d<float>(param.input_shape, param.range.first, param.range.second);
        std::vector<float> expected_result(param.input_shape);


        for (std::size_t i = 0; i < expected_result.size(); i++) {
            auto & x = input_data[i];
            if (param.activationType == "softsign") {
                expected_result[i] =  x  / (1 + fabs(x));
            } else {
                FAIL() << "Unsupported activation type: " << param.activationType;
            }
        }

        assert_that().onInferNgraphModel(buildNgraphFunction(param))
            .inNotCompactMode()
            .gna()
            .propagate_forward()
            .onCPU()
            .called_with_input(input_data)
            .equals_to(expected_result);
    } else {
        assert_that().onInferNgraphModel(buildNgraphFunction(param))
            .inNotCompactMode()
            .gna()
            .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
            .withGNAConfig(GNA_CONFIG_KEY(PRECISION), precision.name())
            .propagate_forward()
            .called_with().pwls_inserted_into_nnet({kActSigmoid});
    }
}

static const ActivationCaseParam gna_activation_test_params[] = {
    {"softsign", 200, {-10, 10}},
};

INSTANTIATE_TEST_SUITE_P(
    GNALayerTests, GNAActivationTest,
    ::testing::Combine(
        ::testing::Values(InferenceEngine::Precision::FP32, InferenceEngine::Precision::I16, InferenceEngine::Precision::I8),
        ::testing::ValuesIn(gna_activation_test_params)),
         GNAActivationTest::getTestName);
