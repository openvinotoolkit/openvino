// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/transpose_conv_transpose_squeeze.hpp"

namespace SubgraphTestsDefinitions {

std::string TransposeConvTest::getTestCaseName(const testing::TestParamInfo<TransposeConvTestParams>& obj) {
    ConvParams convParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(convParams, netPrecision, inputShapes, targetDevice, config) = obj.param;

    std::vector<float> inputArg;
    std::vector<size_t> kernelShape;
    std::vector<size_t> strides;
    size_t inputChannels;
    size_t outputChannels;
    std::tie(kernelShape, strides, inputChannels, outputChannels) = convParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "trgDev=" << targetDevice;
    for (auto const& configItem : config) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    result << "_KERNEL=" << ov::test::utils::vec2str(kernelShape) << "_";
    result << "STRIDES=" << ov::test::utils::vec2str(strides) << "_";
    result << "IC=" << inputChannels << "_";
    result << "OC=" << outputChannels;
    return result.str();
}

void TransposeConvTest::SetUp() {
    ConvParams conv_params;
    std::vector<size_t> input_shape;
    std::map<std::string, std::string> config;
    auto net_precision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(conv_params, net_precision, input_shape, targetDevice, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());

    std::vector<size_t> kernel_shape, strides;
    size_t input_channels, output_channels;
    std::tie(kernel_shape, strides, input_channels, output_channels) = conv_params;
    auto ng_prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(net_precision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ng_prc, ov::Shape(input_shape))};

    std::vector<size_t> nchw_order = { 0, 3, 1, 2 };
    std::vector<size_t> nhwc_order = { 0, 2, 3, 1 };
    std::vector<size_t> conv_input_shape = {1, 1, input_shape[0] * input_shape[1] / input_channels, input_channels};
    auto reshape_pattern = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{conv_input_shape.size()}, conv_input_shape);
    auto reshape = std::make_shared<ngraph::opset7::Reshape>(params[0], reshape_pattern, false);

    const auto input_order1 = std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64,
                                                                         ngraph::Shape({conv_input_shape.size()}),
                                                                         nchw_order);
    auto transpose1 = std::make_shared<ngraph::opset7::Transpose>(reshape, input_order1);

    float weight_val = 0.02;
    auto filter_weights_node = ngraph::builder::makeConstant<float>(ng_prc, {output_channels, input_channels, kernel_shape[0], kernel_shape[1]},
                                                                  { weight_val });

    auto conv = std::make_shared<ngraph::opset7::Convolution>(transpose1, filter_weights_node, strides, std::vector<ptrdiff_t>{ 0, 0 },
                                                              std::vector<ptrdiff_t>{ 0, 0 }, std::vector<size_t>{ 1, 1 },
                                                              ngraph::op::PadType::VALID);

    const auto input_order2 = std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64,
                                                                         ngraph::Shape({conv_input_shape.size()}),
                                                                         nhwc_order);
    auto transpose2 = std::make_shared<ngraph::opset7::Transpose>(conv, input_order2);

    auto constant_squeeze = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{1}, std::vector<size_t>{0});
    auto squeeze = std::make_shared<ngraph::op::Squeeze>(transpose2, constant_squeeze);

    function = std::make_shared<ngraph::Function>(squeeze, params, "transposeConv");
}

InferenceEngine::Blob::Ptr TransposeConvTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin, 1 / inputDataResolution,
                                            seed);
}
}  // namespace SubgraphTestsDefinitions
