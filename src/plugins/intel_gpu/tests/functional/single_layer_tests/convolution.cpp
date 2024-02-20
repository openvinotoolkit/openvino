// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/single_op/convolution.hpp"
#include "common_test_utils/node_builders/convolution.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {
using ov::test::InputShape;
using ov::test::convSpecificParams;

typedef std::tuple<
        convSpecificParams,
        ov::element::Type,     // Net precision
        ov::element::Type,     // Input precision
        ov::element::Type,     // Output precision
        InputShape,            // Input shape
        std::string            // Device name
> convLayerTestParamsSet;

class ConvolutionLayerGPUTest : public testing::WithParamInterface<convLayerTestParamsSet>,
                                virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerTestParamsSet>& obj) {
        convSpecificParams convParams;
        ov::element::Type netType;
        ov::element::Type inType, outType;
        InputShape inputShape;
        std::string targetDevice;
        std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = obj.param;

        ov::op::PadType padType;
        std::vector<size_t> kernel, stride, dilation;
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
        auto netType = ov::element::undefined;
        std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = this->GetParam();

        init_input_shapes({inputShape});

        ov::op::PadType padType;
        std::vector<size_t> kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));

        auto convolutionNode = ov::test::utils::make_convolution(inputParams.front(), netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);

        ov::ResultVector results;
        for (size_t i = 0; i < convolutionNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(convolutionNode->output(i)));

        function = std::make_shared<ov::Model>(results, inputParams, "Convolution");
    }
};

TEST_P(ConvolutionLayerGPUTest, Inference) {
    run();
}

// Check 3D input tensor for convolution is handled properly and its output is correct comparing with ov runtime.
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_3D_tensor_basic, ConvolutionLayerGPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(13),
                        ::testing::Values(ov::op::PadType::SAME_UPPER)),
                ::testing::Values(ov::element::f16),
                ::testing::Values(ov::element::f16),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(InputShape{{}, {{1, 13, 30}}}),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                ConvolutionLayerGPUTest::getTestCaseName);
}  // namespace
