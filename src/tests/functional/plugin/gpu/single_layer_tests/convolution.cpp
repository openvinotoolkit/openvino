// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_layer/convolution.hpp"
#include "common_test_utils/test_constants.hpp"

// using namespace LayerTestsDefinitions;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

using LayerTestsDefinitions::convSpecificParams;

typedef std::tuple<
        convSpecificParams,
        ElementType,     // Net precision
        ElementType,     // Input precision
        ElementType,     // Output precision
        InputShape,      // Input shape
        LayerTestsUtils::TargetDevice   // Device name
> convLayerTestParamsSet;


class ConvolutionLayerGPUTest : public testing::WithParamInterface<convLayerTestParamsSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerTestParamsSet>& obj) {
        convSpecificParams convParams;
        ElementType netType;
        ElementType inType, outType;
        InputShape inputShape;
        std::string targetDevice;
        std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = obj.param;

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        std::ostringstream result;
        result << "IS=";
        result  << CommonTestUtils::partialShape2str({inputShape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << ")_";
        result << "K" << CommonTestUtils::vec2str(kernel) << "_";
        result << "S" << CommonTestUtils::vec2str(stride) << "_";
        result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
        result << "O=" << convOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << netType << "_";
        result << "inPRC=" << inType << "_";
        result << "outPRC=" << outType << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        convSpecificParams convParams;
        InputShape inputShape;
        auto netType = ElementType::undefined;
        std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = this->GetParam();

        init_input_shapes({inputShape});

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        auto inputParams = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParams));

        auto convolutionNode = ngraph::builder::makeConvolution(paramOuts.front(), netType, kernel, stride, padBegin,
                                                                padEnd, dilation, padType, convOutChannels);

        ngraph::ResultVector results;
        for (int i = 0; i < convolutionNode->get_output_size(); i++)
                results.push_back(std::make_shared<ngraph::opset1::Result>(convolutionNode->output(i)));

        function = std::make_shared<ngraph::Function>(results, inputParams, "Convolution");
    }
};

TEST_P(ConvolutionLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {
// Check 3D input tensor for convolution is handled properly and its output is correct comparing with ngraph runtime.
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_3D_tensor_basic, ConvolutionLayerGPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3}),
                        ::testing::Values(SizeVector{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(SizeVector{1}),
                        ::testing::Values(13),
                        ::testing::Values(ngraph::op::PadType::SAME_UPPER)),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::Values(InputShape{{}, {{1, 13, 30}}}),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTest::getTestCaseName);

const std::vector<ov::test::InputShape> dynInputShapes4D = {
    {
        {1, 10, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        {{1, 10, 20, 20}, {1, 10, 30, 30}, {1, 10, 40, 20}}
    },
};

const std::vector<ov::test::InputShape> dynInputShapes3D = {
    {
        {1, 10, ov::Dimension::dynamic()},
        {{1, 10, 20}, {1, 10, 30}, {1, 10, 50}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic3D, ConvolutionLayerGPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3}),
                        ::testing::Values(SizeVector{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1}),
                        ::testing::Values(SizeVector{1}),
                        ::testing::Values(10),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT)),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes3D),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic4D, ConvolutionLayerGPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(10),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT)),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes4D),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTest::getTestCaseName);

}  // namespace

class ManyConvolutionLayersGPUTest : public testing::WithParamInterface<convLayerTestParamsSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerTestParamsSet>& obj) {
        convSpecificParams convParams;
        ElementType netType;
        ElementType inType, outType;
        InputShape inputShape;
        std::string targetDevice;
        std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = obj.param;

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        std::ostringstream result;
        result << "IS=";
        result  << CommonTestUtils::partialShape2str({inputShape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << ")_";
        result << "K" << CommonTestUtils::vec2str(kernel) << "_";
        result << "S" << CommonTestUtils::vec2str(stride) << "_";
        result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
        result << "O=" << convOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << netType << "_";
        result << "inPRC=" << inType << "_";
        result << "outPRC=" << outType << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        convSpecificParams convParams;
        InputShape inputShape;
        auto netType = ElementType::undefined;
        std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = this->GetParam();

        init_input_shapes({inputShape});

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        auto inputParams = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParams));

        auto convolutionNode0 = ngraph::builder::makeConvolution(paramOuts.front(), netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        auto convolutionNode1 = ngraph::builder::makeConvolution(convolutionNode0, netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        auto convolutionNode2 = ngraph::builder::makeConvolution(convolutionNode1, netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        auto convolutionNode3 = ngraph::builder::makeConvolution(convolutionNode2, netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        auto convolutionNode4 = ngraph::builder::makeConvolution(convolutionNode3, netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        auto convolutionNode5 = ngraph::builder::makeConvolution(convolutionNode4, netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        auto convolutionNode6 = ngraph::builder::makeConvolution(convolutionNode5, netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        auto convolutionNode7 = ngraph::builder::makeConvolution(convolutionNode6, netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        auto convolutionNode8 = ngraph::builder::makeConvolution(convolutionNode7, netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        auto convolutionNode9 = ngraph::builder::makeConvolution(convolutionNode8, netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        auto convolutionNode10 = ngraph::builder::makeConvolution(convolutionNode9, netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        auto convolutionNode11 = ngraph::builder::makeConvolution(convolutionNode10, netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        auto convolutionNode12 = ngraph::builder::makeConvolution(convolutionNode11, netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);

        ngraph::ResultVector results;
        for (int i = 0; i < convolutionNode12->get_output_size(); i++)
                results.push_back(std::make_shared<ngraph::opset1::Result>(convolutionNode12->output(i)));

        function = std::make_shared<ngraph::Function>(results, inputParams, "Convolution");
    }
};

TEST_P(ManyConvolutionLayersGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {
const std::vector<ov::test::InputShape> staticInputShapes4D = {
    {
        {},
        {{1, 96, 24, 24}}
    },
};
const std::vector<ov::test::InputShape> dynamicInputShapes4D = {
    {
        {1, 96, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        //{{1, 96, 24, 24}, {1, 96, 48, 48}}
        {{1, 96, 24, 24}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_ManyConvolutionLayerGPUTest_static4D_SymPad, ManyConvolutionLayersGPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(96),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT)),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(staticInputShapes4D),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_ManyConvolutionLayerGPUTest_dynamic4D_SymPad, ManyConvolutionLayersGPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(96),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT)),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynamicInputShapes4D),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_ManyConvolutionLayerGPUTest_dynamic4D_AsymPad, ManyConvolutionLayersGPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{2, 1}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(96),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT)),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynamicInputShapes4D),
                ::testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConvolutionLayerGPUTest::getTestCaseName);

}  // namespace

} // namespace GPULayerTestsDefinitions
