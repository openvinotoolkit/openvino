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
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"

namespace {
using ov::test::InputShape;
using ov::test::groupConvSpecificParams;

typedef std::tuple<
        groupConvSpecificParams,
        ov::element::Type,     // Net precision
        ov::element::Type,     // Input precision
        ov::element::Type,     // Output precision
        ov::element::Type,     // Weights precision
        std::vector<InputShape>,            // Input shape
        bool,                  // Weights scaling
        std::string            // Device name
> groupConvLayerTestParamsSet;

class GroupConvolutionLayerGPUTest : public testing::WithParamInterface<groupConvLayerTestParamsSet>,
                                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<groupConvLayerTestParamsSet>& obj) {
        const auto& [convParams, netType, inType, outType, weightsType, inputShapes, weightsScaling, targetDevice] = obj.param;

        const auto& [kernel, stride, padBegin, padEnd, dilation, convOutChannels, group, padType] = convParams;

        std::ostringstream result;
        result << "IS=";
        result  << ov::test::utils::partialShape2str({inputShapes[0].first}) << "_";
        result << "TS=(";
        for (const auto& shape : inputShapes[0].second) {
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
        result << "weightsPRC=" << weightsType << "_";
        result << "weightsScaling=" << weightsScaling << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [groupConvParams, netType, _inType, _outType, weightsType, inputShapes, weightsScaling, _targetDevice] = this->GetParam();
        inType = _inType;
        outType = _outType;
        targetDevice = _targetDevice;

        init_input_shapes(inputShapes);

        const auto& [_kernel, stride, padBegin, padEnd, dilation, convOutChannels, group, padType] = groupConvParams;
        auto kernel = _kernel;

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));

        std::shared_ptr<ov::Node> groupConvolutionNode;
        if (weightsScaling) {
            size_t convInChannels = static_cast<size_t>(targetStaticShapes.front()[0][1] / group);
            ov::Shape filter_weights_shape = {group, convOutChannels, convInChannels};
            filter_weights_shape.insert(filter_weights_shape.end(), kernel.begin(), kernel.end());
            ov::Shape scaling_shape = {group, convOutChannels, 1, 1};
            auto weights_tensor = ov::test::utils::create_and_fill_tensor(weightsType,
                 filter_weights_shape, ov::test::utils::InputGenerateData(-127, 256, 256, 1));
            auto scaling_tensor = ov::test::utils::create_and_fill_tensor(netType, scaling_shape, ov::test::utils::InputGenerateData(0, 1, 1000, 1));
            auto filter_weights_node = std::make_shared<ov::op::v0::Constant>(weights_tensor);
            auto convert_node = std::make_shared<ov::op::v0::Convert>(filter_weights_node, netType);
            auto scaling_node = std::make_shared<ov::op::v0::Constant>(scaling_tensor);
            auto multiply_node = std::make_shared<ov::op::v1::Multiply>(convert_node, scaling_node);
            groupConvolutionNode = ov::test::utils::make_group_convolution(inputParams[0], multiply_node, netType, stride, padBegin,
                                                                           padEnd, dilation, padType);
        } else {
            groupConvolutionNode = ov::test::utils::make_group_convolution(inputParams[0], netType, kernel, stride, padBegin,
                                                                           padEnd, dilation, padType, convOutChannels, group);
        }


        ov::ResultVector results;
        for (size_t i = 0; i < groupConvolutionNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(groupConvolutionNode->output(i)));

        function = std::make_shared<ov::Model>(results, inputParams, "GroupConvolution");

        if (netType == ov::element::f16) {
            abs_threshold = 0.1;
            rel_threshold = 0.1;
        } else {
            abs_threshold = 0.005;
            rel_threshold = 0.005;
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        for (size_t i = 0lu; i < funcInputs.size(); i++) {
            const auto& funcInput = funcInputs[i];
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -10;
            in_data.resolution = 1000;
            in_data.range = 20u;

            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

const std::vector<InputShape> input_shapes_1d = {
        {{100, 512, 3}, {{100, 512, 3}}},
    };

TEST_P(GroupConvolutionLayerGPUTest, Inference) {
    run();
}

// Check 3D input tensor for convolution is handled properly and its output is correct comparing with ov runtime.
INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionLayerGPUTest_1D_basic,
                         GroupConvolutionLayerGPUTest,
                         ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>{3}),
                                                               ::testing::Values(std::vector<size_t>{1}),
                                                               ::testing::Values(std::vector<ptrdiff_t>{2}),
                                                               ::testing::Values(std::vector<ptrdiff_t>{0}),
                                                               ::testing::Values(std::vector<size_t>{1}),
                                                               ::testing::Values(4),
                                                               ::testing::Values(512),
                                                               ::testing::Values(ov::op::PadType::EXPLICIT)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::element::i8),
                                            ::testing::Values(input_shapes_1d),
                                            ::testing::Values(true),
                                            ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                         GroupConvolutionLayerGPUTest::getTestCaseName);
}  // namespace
