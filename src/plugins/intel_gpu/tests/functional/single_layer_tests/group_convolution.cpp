// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/single_op/group_convolution.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/multiply.hpp"

namespace {
using ov::test::InputShape;
using ov::test::groupConvSpecificParams;

typedef std::tuple<
        groupConvSpecificParams,
        ov::element::Type,     // Net precision
        ov::element::Type,     // Input precision
        ov::element::Type,     // Output precision
        InputShape,            // Input shape
        bool,                  // Weights scaling
        std::string            // Device name
> groupConvLayerTestParamsSet;

class GroupConvolutionLayerGPUTest : public testing::WithParamInterface<groupConvLayerTestParamsSet>,
                                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<groupConvLayerTestParamsSet>& obj) {
        const auto& [convParams, netType, inType, outType, inputShape, weightsScaling, targetDevice] = obj.param;

        const auto& [kernel, stride, padBegin, padEnd, dilation, convOutChannels, group, padType] = convParams;

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
        result << "G=" << group << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << netType << "_";
        result << "inPRC=" << inType << "_";
        result << "outPRC=" << outType << "_";
        result << "weightsScaling=" << weightsScaling << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [groupConvParams, netType, _inType, _outType, inputShape, weightsScaling, _targetDevice] = this->GetParam();
        inType = _inType;
        outType = _outType;
        targetDevice = _targetDevice;

        init_input_shapes({inputShape});

        const auto& [_kernel, stride, padBegin, padEnd, dilation, convOutChannels, group, padType] = groupConvParams;
        auto kernel = _kernel;

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));



        std::shared_ptr<ov::Node> groupConvolutionNode;
        if (weightsScaling) {
            auto input_static_shape = inputDynamicShapes[0].to_shape();
            size_t convInChannels = static_cast<size_t>(input_static_shape.at(1) / group);
            ov::Shape filter_weights_shape = {group, convOutChannels, convInChannels};
            filter_weights_shape.insert(filter_weights_shape.end(), kernel.begin(), kernel.end());
            auto weights_tensor = ov::test::utils::create_and_fill_tensor(netType, filter_weights_shape);
            auto scaling_tensor = ov::test::utils::create_and_fill_tensor(netType, filter_weights_shape, ov::test::utils::InputGenerateData(0, 10, 1000, 1));
            auto filter_weights_node = std::make_shared<ov::op::v0::Constant>(weights_tensor);
            auto scaling_node = std::make_shared<ov::op::v0::Constant>(scaling_tensor);
            auto multiply_node = std::make_shared<ov::op::v1::Multiply>(filter_weights_node, scaling_node);
            groupConvolutionNode = ov::test::utils::make_group_convolution(inputParams.front(), multiply_node, netType, stride, padBegin,
                                                                           padEnd, dilation, padType);
        } else {
            groupConvolutionNode = ov::test::utils::make_group_convolution(inputParams.front(), netType, kernel, stride, padBegin,
                                                                           padEnd, dilation, padType, convOutChannels, group);
        }


        ov::ResultVector results;
        for (size_t i = 0; i < groupConvolutionNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(groupConvolutionNode->output(i)));

        function = std::make_shared<ov::Model>(results, inputParams, "GroupConvolution");
    }
};

TEST_P(GroupConvolutionLayerGPUTest, Inference) {
    run();
}

// Check 3D input tensor for convolution is handled properly and its output is correct comparing with ov runtime.
INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionLayerGPUTest_1D_basic,
                         GroupConvolutionLayerGPUTest,
                         ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>{3}),
                                                               ::testing::Values(std::vector<size_t>{1}),
                                                               ::testing::Values(std::vector<ptrdiff_t>{0}),
                                                               ::testing::Values(std::vector<ptrdiff_t>{0}),
                                                               ::testing::Values(std::vector<size_t>{1}),
                                                               ::testing::Values(4),
                                                               ::testing::Values(32),
                                                               ::testing::Values(ov::op::PadType::SAME_UPPER)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(InputShape{{}, {{10, 32, 3}}}),
                                            ::testing::Values(true),
                                            ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                         GroupConvolutionLayerGPUTest::getTestCaseName);
}  // namespace
