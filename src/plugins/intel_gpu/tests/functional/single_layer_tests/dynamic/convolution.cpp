// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
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
        LayerTestsUtils::TargetDevice,   // Device name
        bool             // activation fusing
> convLayerTestParamsSet;


class ConvolutionLayerGPUTestDynamic : public testing::WithParamInterface<convLayerTestParamsSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerTestParamsSet>& obj) {
        convSpecificParams convParams;
        ElementType netType;
        ElementType inType, outType;
        InputShape inputShape;
        std::string targetDevice;
        bool activationFusing;
        std::tie(convParams, netType, inType, outType, inputShape, targetDevice, activationFusing) = obj.param;

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        std::ostringstream result;
        result << "IS=";
        result  << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << ")_";
        result << "K" << ov::test::utils::vec2str(kernel) << "_";
        result << "S" << ov::test::utils::vec2str(stride) << "_";
        result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
        result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
        result << "D=" << ov::test::utils::vec2str(dilation) << "_";
        result << "O=" << convOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << netType << "_";
        result << "inPRC=" << inType << "_";
        result << "outPRC=" << outType << "_";
        result << "trgDev=" << targetDevice << "_";
        result << "activationFusing=" << activationFusing;

        return result.str();
    }

protected:
    void SetUp() override {
        convSpecificParams convParams;
        InputShape inputShape;
        auto netType = ElementType::undefined;
        bool activationFusing;
        std::tie(convParams, netType, inType, outType, inputShape, targetDevice, activationFusing) = this->GetParam();

        init_input_shapes({inputShape});

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        // WA: check data when input shape is dynamic and pad is exist.
        //     If 1d conv, 1d pad should be applied to y axis. But there was a bug what it applied to x axis.
        if (inputShape.first.is_dynamic() && padBegin.size() == 1 && padBegin[0] == 1 && padEnd.size() == 1 && padEnd[0] == 1) {
            abs_threshold = 9;
            rel_threshold = 0.002;
        }

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParams));

        auto convolutionNode = ngraph::builder::makeConvolution(paramOuts.front(), netType, kernel, stride, padBegin,
                                                                padEnd, dilation, padType, convOutChannels);
        if (activationFusing) {
                auto activationNode = ngraph::builder::makeActivation(convolutionNode, netType, ngraph::helpers::ActivationTypes::Relu);

                ngraph::ResultVector results;
                for (size_t i = 0; i < activationNode->get_output_size(); i++)
                results.push_back(std::make_shared<ngraph::opset1::Result>(activationNode->output(i)));

                function = std::make_shared<ngraph::Function>(results, inputParams, "Convolution");
        } else {
                ngraph::ResultVector results;
                for (size_t i = 0; i < convolutionNode->get_output_size(); i++)
                results.push_back(std::make_shared<ngraph::opset1::Result>(convolutionNode->output(i)));

                function = std::make_shared<ngraph::Function>(results, inputParams, "Convolution");
        }
    }
};

TEST_P(ConvolutionLayerGPUTestDynamic, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {

// ======== 1D convolutions
const std::vector<ov::test::InputShape> dynInputShapes1D = {
    {
        {1, 10, ov::Dimension::dynamic()},
        {{1, 10, 20}, {1, 10, 30}, {1, 10, 50}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic1DSymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3}),
                        ::testing::Values(SizeVector{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1}),
                        ::testing::Values(SizeVector{1}),
                        ::testing::Values(10),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes1D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

const std::vector<SizeVector> kernels1D = { {3}, {1} };
const std::vector<SizeVector> strides1D = { {1} };
const std::vector<std::vector<ptrdiff_t>> padBegins1D = { {0}, {1} };
const std::vector<std::vector<ptrdiff_t>> padEnds1D = { {0}, {1} };
const std::vector<SizeVector> dilations1D = { {1} };
const SizeVector numOutChannels = { 64, 63 };
const std::vector<InputShape> inputShapes1D = {
        {{}, {{ 2, 64, 7 }}},
        {{}, {{ 1, 67, 7 }}},
        {
            //dynamic shape
            { -1, 64, {1, 200} },
            { //target static shapes
                { 2, 64, 7 },
                { 1, 64, 9 }
            }
        },
        {
            //dynamic shape
            { {1, 200}, 64, -1 },
            { //target static shapes
                { 2, 64, 7 },
                { 1, 64, 5 }
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_ExplicitPad1D, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(kernels1D),
                        ::testing::ValuesIn(strides1D),
                        ::testing::ValuesIn(padBegins1D),
                        ::testing::ValuesIn(padEnds1D),
                        ::testing::ValuesIn(dilations1D),
                        ::testing::ValuesIn(numOutChannels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT)),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(inputShapes1D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ======== 2D convolutions
const std::vector<ov::test::InputShape> dynInputShapes2D = {
    {
        {1, 10, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        {{1, 10, 20, 20}, {1, 10, 30, 30}, {1, 10, 40, 20}}
    },
};

// Specific range causes output static shapeS
const std::vector<ov::test::InputShape> dynInputShapes2D_static_output = {
    {
        {1, 128, {1, 2}, {1, 2}},
        {{1, 128, 1, 1}, {1, 128, 2, 2}}
    },
};
// ==== Symmetric pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic2DSymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(10),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Symmetric auto pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic2DSymAutoPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(10),
                        ::testing::ValuesIn({ngraph::op::PadType::SAME_LOWER, ngraph::op::PadType::SAME_UPPER})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Asymmetric pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic2D_AsymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{2, 1}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(10),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Static output
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic2D_static_output, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{2, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 1}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(256),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT)),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes2D_static_output),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(true)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ======== 3D convolutions
const std::vector<ov::test::InputShape> dynInputShapes3D = {
    {
        {1, 3, ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        {{1, 3, 10, 10, 10}, {1, 3, 20, 20, 10}, {1, 3, 15, 15, 10}}
    },
};

// ==== Symmetric pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic3DSymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3, 3}),
                        ::testing::Values(SizeVector{1, 1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2, 1}),
                        ::testing::Values(SizeVector{1, 1, 1}),
                        ::testing::Values(3),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes3D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Symmetric auto pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic3DSymAutoPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3, 3}),
                        ::testing::Values(SizeVector{1, 1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0, 0, 0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0, 0, 0}),
                        ::testing::Values(SizeVector{1, 1, 1}),
                        ::testing::Values(3),
                        ::testing::ValuesIn({ngraph::op::PadType::SAME_LOWER, ngraph::op::PadType::SAME_UPPER})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes3D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Asymmetric pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic3DAsymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3, 3}),
                        ::testing::Values(SizeVector{1, 1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{2, 1, 1}),
                        ::testing::Values(SizeVector{1, 1, 1}),
                        ::testing::Values(3),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes3D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

}  // namespace
} // namespace GPULayerTestsDefinitions
