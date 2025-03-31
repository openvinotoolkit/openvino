// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_op/group_convolution.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/group_conv.hpp"

namespace {
using ov::test::InputShape;
using ov::test::groupConvSpecificParams;

typedef std::tuple<
        groupConvSpecificParams,
        ov::element::Type,     // Model type
        InputShape,            // Input shape
        std::string            // Device name
> groupConvLayerTestParamsSet;

class GroupConvolutionLayerGPUTestDynamic : public testing::WithParamInterface<groupConvLayerTestParamsSet>,
                                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<groupConvLayerTestParamsSet>& obj) {
        groupConvSpecificParams groupConvParams;
        ov::element::Type model_type;
        InputShape inputShape;
        std::string targetDevice;
        std::tie(groupConvParams, model_type, inputShape, targetDevice) = obj.param;

        ov::op::PadType padType;
        std::vector<size_t> kernel, stride, dilation;
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
        result << "netPRC=" << model_type << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        groupConvSpecificParams groupConvParams;
        InputShape inputShape;
        auto model_type = ov::element::dynamic;
        std::tie(groupConvParams, model_type, inputShape, targetDevice) = this->GetParam();

        init_input_shapes({inputShape});

        ov::op::PadType padType;
        std::vector<size_t> kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        size_t numGroups;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvParams;

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        auto groupConvolutionNode = ov::test::utils::make_group_convolution(inputParams.front(), model_type, kernel, stride, padBegin,
                                                                            padEnd, dilation, padType, convOutChannels, numGroups);

        ov::ResultVector results;
        for (size_t i = 0; i < groupConvolutionNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(groupConvolutionNode->output(i)));

        function = std::make_shared<ov::Model>(results, inputParams, "GroupConvolution");
    }
};

TEST_P(GroupConvolutionLayerGPUTestDynamic, Inference) {
    run();
}

const std::vector<ov::test::InputShape> dynInputShapes1D = {
    {
        {1, 12, ov::Dimension::dynamic()},
        {{1, 12, 20}, {1, 12, 30}, {1, 12, 50}}
    },
};
INSTANTIATE_TEST_SUITE_P(smoke_DwGroupConvolutionLayerGPUTest_dynamic1DSymPad, GroupConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(12),
                        ::testing::Values(12),
                        ::testing::ValuesIn({ov::op::PadType::EXPLICIT, ov::op::PadType::VALID})),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(dynInputShapes1D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                GroupConvolutionLayerGPUTestDynamic::getTestCaseName);

// group convolution is not working for static case too
INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionLayerGPUTest_dynamic1DSymPad_Disabled, GroupConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(4),
                        ::testing::Values(4),
                        ::testing::ValuesIn({ov::op::PadType::EXPLICIT, ov::op::PadType::VALID})),
                ::testing::Values(ov::element::f16),
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
                        ::testing::Values(std::vector<size_t>{3, 3}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(4),
                        ::testing::Values(4),
                        ::testing::ValuesIn({ov::op::PadType::EXPLICIT, ov::op::PadType::VALID})),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                GroupConvolutionLayerGPUTestDynamic::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionLayerGPUTest_dynamic2D_AsymPad, GroupConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3, 3}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{2, 1}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(4),
                        ::testing::Values(4),
                        ::testing::ValuesIn({ov::op::PadType::EXPLICIT, ov::op::PadType::VALID})),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                GroupConvolutionLayerGPUTestDynamic::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionLayerGPUTest_dynamic2D_SymAutoPad, GroupConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3, 3}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(4),
                        ::testing::Values(4),
                        ::testing::ValuesIn({ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER})),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                GroupConvolutionLayerGPUTestDynamic::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionLayerGPUTest_dynamic2D_AsymAutoPad, GroupConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3, 3}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{2, 1}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(4),
                        ::testing::Values(4),
                        ::testing::ValuesIn({ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER})),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                GroupConvolutionLayerGPUTestDynamic::getTestCaseName);
}  // namespace
