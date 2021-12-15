// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <single_layer_common.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/ops.hpp>
#include <ie_precision.hpp>
#include "../gna_matcher.hpp"

using SplitToConcatTestParams  = std::tuple<InferenceEngine::Precision, std::size_t, std::size_t>;

class GNASplitToConcatTest : public GNATest<>,
                             public testing::WithParamInterface<SplitToConcatTestParams> {
public:

    static std::string getTestName(const testing::TestParamInfo<SplitToConcatTestParams>& params) {
        std::string test_name = "first_" + std::to_string(std::get<1>(params.param)) + "_second_";
        test_name += std::to_string(std::get<2>(params.param)) + "_";
        test_name += std::get<0>(params.param).name();
        return test_name;
    }

protected:

    InferenceEngine::Precision precision = InferenceEngine::Precision::FP32;
    std::size_t input_dim = 0;
    std::size_t first_split_output = 0;
    std::size_t second_split_output = 0;
    const float MUL_VALUE = -1.0f;

    void SetUp() override {
        std::tie(precision, first_split_output, second_split_output) = GetParam();
        input_dim = first_split_output + second_split_output;
    }

    std::shared_ptr<ngraph::Function> getNgraphModel() {
        std::vector<std::size_t> split_desc_vector({first_split_output, second_split_output});
        auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, input_dim});
        auto split = std::make_shared<ngraph::op::v1::VariadicSplit>(input,
                ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1}),
                ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{split_desc_vector.size()}, split_desc_vector));
        auto concat = std::make_shared<ngraph::op::Concat>(split->outputs(), 1);

        auto weights = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape {1}, std::vector<float> {MUL_VALUE});
        auto biases = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape {1}, std::vector<float> {0.0f});

        auto mul = std::make_shared<ngraph::op::v1::Multiply>(concat, weights);
        auto add = std::make_shared<ngraph::op::v1::Add>(mul, biases);

        auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input});
        return function;
    }
};

TEST_P(GNASplitToConcatTest, SplitToConcatWith2Inputs) {
    if (precision == InferenceEngine::Precision::FP32) {
        std::vector<float> input_data(input_dim);
        std::iota(input_data.begin(), input_data.end(), 1.0);

        std::vector<float> expected_result(input_dim);
        for (std::size_t i = 0; i < expected_result.size(); i++) {
            expected_result[i] = input_data[i] * MUL_VALUE;
        }

        assert_that().onInferNgraphModel(getNgraphModel())
                .inNotCompactMode()
                .gna()
                .propagate_forward()
                .onCPU()
                .called_with_input(input_data)
                .equals_to(expected_result);
    } else {
        assert_that().onInferNgraphModel(getNgraphModel())
                .inNotCompactMode()
                .gna()
                .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                .withGNAConfig(GNA_CONFIG_KEY(PRECISION), "I16")
                .propagate_forward()
                .called();
    }
}

INSTANTIATE_TEST_SUITE_P(
    GNALayerTests,
    GNASplitToConcatTest,
    testing::Combine(
        testing::Values(InferenceEngine::Precision::FP32, InferenceEngine::Precision::I16),
        // Size of first Split layer output
        testing::Values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 32, 42, 50, 64, 96, 100, 128, 132),
        // Size of second Split layer output
        testing::Values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 32, 42, 50, 64, 96, 100, 128, 132)),
    GNASplitToConcatTest::getTestName);
