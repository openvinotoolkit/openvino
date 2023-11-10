// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset1.hpp"
#include "test_utils/convolution_params.hpp"
#include "subgraph_tests/include/conv_with_zero_point_fuse.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

std::string ConvWithZeroPointFuseSubgraphTest::getTestCaseName(testing::TestParamInfo<convConcatCPUParams> obj) {
    std::ostringstream result;
    nodeType type;
    SizeVector inputShapes;
    std::tie(type, inputShapes) = obj.param;

    result << "Type=" << nodeType2str(type) << "_";
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";

    return result.str();
}

void ConvWithZeroPointFuseSubgraphTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;
    nodeType type;
    SizeVector inputShapes;
    std::tie(type, inputShapes) = this->GetParam();
    pluginTypeNode = nodeType2PluginType(type);

    const ngraph::op::PadType paddingType { ngraph::op::PadType::EXPLICIT };
    const size_t numOutChannels = 256;
    const SizeVector dilation { 1, 1 };
    const SizeVector kernelSize { 1, 1 };
    const SizeVector strides { 1, 1 };
    const std::vector<ptrdiff_t> padBegin { 0, 0 };
    const std::vector<ptrdiff_t> padEnd { 0, 0 };

    selectedType = ".*_I8";

    ov::ParameterVector inputParams {std::make_shared<ov::op::v0::Parameter>(ngraph::element::f32, ov::Shape(inputShapes))};
    const auto fq = ngraph::builder::makeFakeQuantize(
        inputParams[0],
        ov::element::f32,
        256,
        {1, 1, 1, 1},
        {-12.8f},
        {12.7f},
        {-12.8f},
        {12.7f});

    std::vector<std::shared_ptr<ngraph::Node>> branches(2);
    {
        ngraph::Strides strides{1, 1};
        ngraph::Shape pads_begin{0, 0}, pads_end{0, 0}, kernel{1, 1};
        branches[0] = std::make_shared<ngraph::opset1::MaxPool>(fq,
                                                                        strides,
                                                                        pads_begin,
                                                                        pads_end,
                                                                        kernel);
    }
    {
        const auto fq_conv_data = ngraph::builder::makeFakeQuantize(
            fq,
            ov::element::f32,
            256,
            {1, 1, 1, 1},
            {-12.8f},
            {12.7f},
            {-12.8f},
            {12.7f});

        const InferenceEngine::SizeVector weights_const_shape = {numOutChannels, inputShapes[1], kernelSize[0], kernelSize[1]};
        const auto weights_const_values = std::vector<int>(ngraph::shape_size(weights_const_shape), 1);
        const auto weights_const = ngraph::builder::makeConstant(ov::element::i8, weights_const_shape, weights_const_values);

        const auto weights_convert = ngraph::builder::makeConversion(
            weights_const,
            ov::element::f32,
            ngraph::helpers::ConversionTypes::CONVERT);

        const auto weights_multiply = std::make_shared<ov::opset10::Multiply>(
            weights_convert,
            ngraph::builder::makeConstant(ov::element::f32,
                                            {numOutChannels, 1, 1, 1},
                                            std::vector<float>(numOutChannels, 1.0)));

        switch (type) {
            case nodeType::convolution: {
                branches[1] = ngraph::builder::makeConvolution(fq_conv_data,
                                                               weights_multiply,
                                                               ngraph::element::f32,
                                                               kernelSize,
                                                               strides,
                                                               padBegin,
                                                               padEnd,
                                                               dilation,
                                                               paddingType,
                                                               numOutChannels);
                break;
            }
            case nodeType::groupConvolution: {
                branches[1] = ngraph::builder::makeGroupConvolution(
                    fq_conv_data,
                    std::make_shared<ov::opset10::Reshape>(
                        weights_multiply,
                        ngraph::builder::makeConstant(
                            ov::element::i32,
                            {5},
                            std::vector<size_t>{1, numOutChannels, inputShapes[1], kernelSize[0], kernelSize[1]}),
                        true),
                    ngraph::element::f32,
                    strides,
                    padBegin,
                    padEnd,
                    dilation,
                    paddingType);
                break;
            }
            default: {
                throw std::runtime_error("Subgraph concat test doesn't support this type of operation");
            }
        }
    }

    auto concat = ngraph::builder::makeConcat(ngraph::OutputVector{branches[0], branches[1]}, 1);

    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, inputParams, "ConvWithZeroPointFuseSubgraphTest");
}

TEST_P(ConvWithZeroPointFuseSubgraphTest, CompareWithRefs) {
    Run();

    CheckPluginRelatedResults(executableNetwork, pluginTypeNode);
};

TEST_P(ConvWithZeroPointFuseSubgraphTest, CompareWithRefs_FP16) {
    if (!(ov::with_cpu_x86_avx512_core_fp16() || ov::with_cpu_x86_avx512_core_amx_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    configuration.insert({ov::hint::inference_precision.name(), "f16"});

    Run();

    CheckPluginRelatedResults(executableNetwork, pluginTypeNode);
};


const SizeVector inputShapes2D = {1, 32, 136, 136};

const auto params2DConv = ::testing::Combine(::testing::ValuesIn({nodeType::convolution, nodeType::groupConvolution}),
                                             ::testing::Values(inputShapes2D));

INSTANTIATE_TEST_SUITE_P(smoke_ConvWithZeroPointFuse,
                         ConvWithZeroPointFuseSubgraphTest,
                         params2DConv,
                         ConvWithZeroPointFuseSubgraphTest::getTestCaseName);

}  // namespace SubgraphTestsDefinitions
