// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <tuple>
#include <gtest/gtest.h>
#include <ngraph/op/parameter.hpp>
#include <ngraph/ops.hpp>
#include <ie_precision.hpp>
#include "../gna_matcher.hpp"

using MultiInputToConcatParams = std::tuple<InferenceEngine::Precision, std::size_t, std::size_t>;

class GNAMultiInputToConcatTest : public GNATest<>,
                                  public testing::WithParamInterface<MultiInputToConcatParams> {
public:

    static std::string getTestName(const testing::TestParamInfo<MultiInputToConcatParams>& params) {
        std::string test_name = std::to_string(std::get<1>(params.param)) + "_inputs_";
        test_name += std::to_string(std::get<2>(params.param)) + "_per_input_";
        test_name += std::get<0>(params.param).name();
        return test_name;
    }

protected:

    InferenceEngine::Precision precision = InferenceEngine::Precision::FP32;
    std::size_t number_of_inputs = 0;
    std::size_t dims_pre_input = 0;
    const float MUL_VALUE = -1.0f;

    void SetUp() override {
        std::tie(precision, number_of_inputs, dims_pre_input) = GetParam();
    }

    std::tuple<std::vector<std::string>, std::shared_ptr<ngraph::Function>> getNgraphModelWithIO() {
        ngraph::ParameterVector inputs;
        std::vector<std::string> inputs_names;
        ngraph::OutputVector outputs_from_inputs;

        for (std::size_t i = 0; i < number_of_inputs; i++) {
            auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, dims_pre_input });
            inputs.push_back(input);
            inputs_names.push_back(input->get_name());
            outputs_from_inputs.push_back(input->outputs()[0]);
        }

        auto concat = std::make_shared<ngraph::op::Concat>(outputs_from_inputs, 1);

        auto weights = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1 }, std::vector<float> {MUL_VALUE});
        auto biases = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1 }, std::vector<float> {0.0f});

        auto mul = std::make_shared<ngraph::op::v1::Multiply>(concat, weights);
        auto add = std::make_shared<ngraph::op::v1::Add>(mul, biases);

        auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, inputs);
        return std::make_tuple(inputs_names, function);
    }
};

TEST_P(GNAMultiInputToConcatTest, InputsToConcat) {
    if (precision == InferenceEngine::Precision::I16) {
        GTEST_SKIP();
    }
    float start_from = 1.0f;

    auto model_with_io_names = getNgraphModelWithIO();
    auto input_names = std::get<0>(model_with_io_names);

    std::vector<std::vector<float>> inputs;
    for (std::size_t i = 0; i < number_of_inputs; i++) {
        std::vector<float> input_data(dims_pre_input);
        std::iota(input_data.begin(), input_data.end(), start_from);
        start_from = input_data[input_data.size() - 1] + 1.0f;
        inputs.push_back(input_data);
    }

    if (precision == InferenceEngine::Precision::FP32) {
        auto test_object = assert_that().onInferNgraphModel(std::get<1>(model_with_io_names))
                .inNotCompactMode()
                .gna()
                .propagate_forward()
                .onCPU()
                .called_with();
        std::vector<float> expected_result;
        for (std::size_t i = 0; i < number_of_inputs; i++) {
            test_object.input(input_names[i], inputs[i]);
            expected_result.insert(expected_result.end(), inputs[i].begin(), inputs[i].end());
        }
        for (std::size_t i = 0; i < expected_result.size(); i++) {
            expected_result[i] *= MUL_VALUE;
        }
        test_object.equals_to(expected_result);
    } else {
        auto test_object = assert_that().onInferNgraphModel(std::get<1>(model_with_io_names))
                .inNotCompactMode()
                .gna()
                .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                .withGNAConfig(GNA_CONFIG_KEY(PRECISION), "I16")
                .propagate_forward()
                .called();
        for (std::size_t i = 0; i < number_of_inputs; i++) {
            test_object.input(input_names[i], inputs[i]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
        GNALayerTests,
        GNAMultiInputToConcatTest,
        ::testing::Combine(
                ::testing::Values(InferenceEngine::Precision::FP32, InferenceEngine::Precision::I16),
                // Number of inputs to Concat layer
                ::testing::Values(2, 3, 4, 5, 6, 7, 8, 9, 10, 32, 96),
                // Size of each input
                ::testing::Values(1, 2, 3, 8, 9, 10, 15, 16, 32, 42, 48, 50, 64, 96, 100, 128, 132)),
        GNAMultiInputToConcatTest::getTestName);
