// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/split_conv_concat.hpp"

namespace SubgraphTestsDefinitions {

std::string SplitConvConcat::getTestCaseName(const testing::TestParamInfo<LayerTestsUtils::basicParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void SplitConvConcat::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

    std::vector<float> filterWeights1;
    std::vector<float> filterWeights2;
    if (targetDevice == CommonTestUtils::DEVICE_GNA) {
        filterWeights1 = CommonTestUtils::generate_float_numbers(8 * inputShape[1] / 2 * 3, -0.2f, 0.2f);
        filterWeights2 = CommonTestUtils::generate_float_numbers(8 * inputShape[1] / 2 * 3, -0.2f, 0.2f);
    }
    auto conv1 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {1, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::VALID, 8, false, filterWeights1);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1);

    auto conv2 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {1, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::VALID, 8, false, filterWeights2);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2);
    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), relu2->output(0)}, 1);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, params, "SplitConvConcat");
}

std::string ConvConcatD::getTestCaseName(const testing::TestParamInfo<LayerTestsUtils::basicParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ConvConcatD::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    const size_t kx = 3;
    const size_t ky = 3;
    const size_t outCh = 8;
    std::vector<float> filterWeights1;
    std::vector<float> filterWeights2;
    std::vector<float> filterWeights3;
    if (targetDevice == CommonTestUtils::DEVICE_GNA) {
        const auto fn = outCh * kx * ky * inputShape[1];
        filterWeights1 = CommonTestUtils::generate_float_numbers(fn, -0.2f, 0.2f);
        filterWeights2 = CommonTestUtils::generate_float_numbers(fn, -0.2f, 0.2f);
        filterWeights3 = CommonTestUtils::generate_float_numbers(9*8*16, -0.2f, 0.2f);
    }
    auto conv1 = ngraph::builder::makeConvolution(params[0],
                                                  ngPrc,
                                                  {kx, ky},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ngraph::op::PadType::VALID,
                                                  outCh,
                                                  false,
                                                  filterWeights1);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1);

    auto conv2 = ngraph::builder::makeConvolution(params[0],
                                                  ngPrc,
                                                  {kx, ky},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ngraph::op::PadType::VALID,
                                                  outCh,
                                                  false,
                                                  filterWeights2);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2);
    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), relu2->output(0)}, 1);

    auto conv3 = ngraph::builder::makeConvolution(concat,
                                                  ngPrc,
                                                  {kx, ky},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ngraph::op::PadType::VALID,
                                                  outCh,
                                                  false,
                                                  filterWeights3);
    auto relu3 = std::make_shared<ngraph::opset1::Relu>(conv3);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(relu3)};
    function = std::make_shared<ngraph::Function>(results, params, "ConvConcatD");
}

}  // namespace SubgraphTestsDefinitions
