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
#include "shared_test_classes/single_layer/group_convolution.hpp"
#include "common_test_utils/test_constants.hpp"

// using namespace LayerTestsDefinitions;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

using LayerTestsDefinitions::groupConvSpecificParams;
typedef std::tuple<
        groupConvSpecificParams,
        ElementType,     // Net precision
        ElementType,     // Input precision
        ElementType,     // Output precision
        InputShape,      // Input shape
        LayerTestsUtils::TargetDevice   // Device name
> groupConvLayerTestParamsSet;


class GroupConvolutionLayerGPUTestDynamic : public testing::WithParamInterface<groupConvLayerTestParamsSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<groupConvLayerTestParamsSet>& obj) {
        groupConvSpecificParams groupConvParams;
        ElementType netType;
        ElementType inType, outType;
        InputShape inputShape;
        std::string targetDevice;
        std::tie(groupConvParams, netType, inType, outType, inputShape, targetDevice) = obj.param;

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        size_t numGroups;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvParams;

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
        result << "G=" << numGroups << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << netType << "_";
        result << "inPRC=" << inType << "_";
        result << "outPRC=" << outType << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        groupConvSpecificParams groupConvParams;
        InputShape inputShape;
        auto netType = ElementType::undefined;
        std::tie(groupConvParams, netType, inType, outType, inputShape, targetDevice) = this->GetParam();

        init_input_shapes({inputShape});

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        size_t numGroups;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvParams;

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));

        auto groupConvolutionNode = ngraph::builder::makeGroupConvolution(inputParams.front(), netType, kernel, stride, padBegin,
                                                                padEnd, dilation, padType, convOutChannels, numGroups);

        ngraph::ResultVector results;
        for (size_t i = 0; i < groupConvolutionNode->get_output_size(); i++)
                results.push_back(std::make_shared<ngraph::opset1::Result>(groupConvolutionNode->output(i)));

        function = std::make_shared<ngraph::Function>(results, inputParams, "GroupConvolution");
    }
};

TEST_P(GroupConvolutionLayerGPUTestDynamic, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {
const std::vector<ov::test::InputShape> dynInputShapes1D = {
    {
        {1, 12, ov::Dimension::dynamic()},
        {{1, 12, 20}, {1, 12, 30}, {1, 12, 50}}
    },
};
INSTANTIATE_TEST_SUITE_P(smoke_DwGroupConvolutionLayerGPUTest_dynamic1DSymPad, GroupConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3}),
                        ::testing::Values(SizeVector{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(SizeVector{1}),
                        ::testing::Values(12),
                        ::testing::Values(12),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes1D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                GroupConvolutionLayerGPUTestDynamic::getTestCaseName);

// group convolution is not working for static case too
INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionLayerGPUTest_dynamic1DSymPad_Disabled, GroupConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3}),
                        ::testing::Values(SizeVector{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(SizeVector{1}),
                        ::testing::Values(4),
                        ::testing::Values(4),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes1D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                GroupConvolutionLayerGPUTestDynamic::getTestCaseName);


const std::vector<ov::test::InputShape> dynInputShapes2D = {
    {
        {1, 12, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        {{1, 12, 224, 224}, {1, 12, 48, 48}, {1, 12, 64, 16}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionLayerGPUTest_dynamic2DSymPad, GroupConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(4),
                        ::testing::Values(4),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                GroupConvolutionLayerGPUTestDynamic::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionLayerGPUTest_dynamic2D_AsymPad, GroupConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{2, 1}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(4),
                        ::testing::Values(4),
                        ::testing::ValuesIn({ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                GroupConvolutionLayerGPUTestDynamic::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionLayerGPUTest_dynamic2D_SymAutoPad, GroupConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(4),
                        ::testing::Values(4),
                        ::testing::ValuesIn({ngraph::op::PadType::SAME_LOWER, ngraph::op::PadType::SAME_UPPER})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                GroupConvolutionLayerGPUTestDynamic::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionLayerGPUTest_dynamic2D_AsymAutoPad, GroupConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(SizeVector{3, 3}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{2, 1}),
                        ::testing::Values(SizeVector{1, 1}),
                        ::testing::Values(4),
                        ::testing::Values(4),
                        ::testing::ValuesIn({ngraph::op::PadType::SAME_LOWER, ngraph::op::PadType::SAME_UPPER})),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::f16),
                ::testing::Values(ElementType::undefined),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                GroupConvolutionLayerGPUTestDynamic::getTestCaseName);
}  // namespace

} // namespace GPULayerTestsDefinitions
