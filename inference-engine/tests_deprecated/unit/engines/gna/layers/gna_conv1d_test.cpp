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

struct Point2D {
    std::size_t x = 0;
    std::size_t y = 0;
};

using Conv1DParams = std::tuple<InferenceEngine::Precision, std::size_t, std::size_t, std::size_t, std::size_t>;

class GNAConv1DTest : public GNATest<>,
                                  public testing::WithParamInterface<Conv1DParams> {
public:

    static std::string getTestName(const testing::TestParamInfo<Conv1DParams>& params) {
        std::string test_name = std::to_string(std::get<1>(params.param)) + "_kernel.x_";
        test_name += std::to_string(std::get<2>(params.param)) + "_pad.x_";
        test_name += std::to_string(std::get<3>(params.param)) + "_stride.x_";
        test_name += std::to_string(std::get<4>(params.param)) + "_output_channels_";
        test_name += std::get<0>(params.param).name();
        return test_name;
    }

protected:

    InferenceEngine::Precision precision = InferenceEngine::Precision::FP32;
    Point2D kernel;
    Point2D pad;
    Point2D stride;
    std::size_t output_channels = 0;
    std::size_t input_dim = 784;

    void SetUp() override {
        std::tie(precision, kernel.x, pad.x, stride.x, output_channels) = GetParam();
        kernel.y = 1;
        pad.y = 0;
        stride.y = 0;
    }

    std::shared_ptr<ngraph::Function> getNgraphModel() {
        auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, input_dim});

        std::vector<int> shape {1, 1};
        shape.push_back(input_dim);
        auto input_reshaped = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape {shape.size()}, shape);
        auto reshape = std::make_shared<ngraph::op::v1::Reshape>(input, input_reshaped, false);

        auto filters = std::make_shared<ngraph::op::Constant>(
                ngraph::element::f32,
                ngraph::Shape {output_channels, kernel.y, kernel.x},
                std::vector<float> {1.0f});
        auto strides = ngraph::Strides(stride.x);
        auto dilations = ngraph::Strides(stride.y);
        auto pads_begin = ngraph::CoordinateDiff(pad.x);
        auto pads_end = ngraph::CoordinateDiff(pad.y);
        auto convolution = std::make_shared<ngraph::op::v1::Convolution>(
                reshape,
                filters,
                strides,
                pads_begin,
                pads_end,
                dilations);

        auto weights = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape {}, std::vector<float> {2.0f});
        auto mul = std::make_shared<ngraph::op::v1::Multiply>(convolution, weights);

        auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});
        return function;
    }
};

TEST_P(GNAConv1DTest, SplitToConcatWith2Inputs) {
    if (precision == InferenceEngine::Precision::FP32) {
        std::vector<float> input_data(input_dim);
        std::iota(input_data.begin(), input_data.end(), 1.0);

        assert_that().onInferNgraphModel(getNgraphModel())
                .inNotCompactMode()
                .gna()
                .propagate_forward()
                .onCPU()
                .called_with_input(input_data);
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
        GNAConv1DTest,
        testing::Combine(
                testing::Values(InferenceEngine::Precision::FP32, InferenceEngine::Precision::I16),
                testing::Values(1, 3, 9, 16, 24, 32, 42, 64),
                testing::Values(0, 1),
                testing::Values(0),
                testing::Values(32, 128, 512)),
        GNAConv1DTest::getTestName);
