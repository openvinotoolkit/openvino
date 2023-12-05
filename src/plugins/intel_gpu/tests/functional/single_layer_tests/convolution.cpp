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

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));

        auto convolutionNode = ngraph::builder::makeConvolution(inputParams.front(), netType, kernel, stride, padBegin,
                                                                padEnd, dilation, padType, convOutChannels);

        ngraph::ResultVector results;
        for (size_t i = 0; i < convolutionNode->get_output_size(); i++)
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
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                ConvolutionLayerGPUTest::getTestCaseName);
}  // namespace

} // namespace GPULayerTestsDefinitions
