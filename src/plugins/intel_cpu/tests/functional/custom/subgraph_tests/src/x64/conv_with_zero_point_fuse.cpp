// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/include/conv_with_zero_point_fuse.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "utils/convolution_params.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string ConvWithZeroPointFuseSubgraphTest::getTestCaseName(testing::TestParamInfo<convConcatCPUParams> obj) {
    std::ostringstream result;
    nodeType type;
    ov::Shape inputShapes;
    std::tie(type, inputShapes) = obj.param;

    result << "Type=" << nodeType2str(type) << "_";
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";

    return result.str();
}

void ConvWithZeroPointFuseSubgraphTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;
    nodeType type;
    ov::Shape inputShapes;
    std::tie(type, inputShapes) = this->GetParam();
    pluginTypeNode = nodeType2PluginType(type);

    const ov::op::PadType paddingType{ov::op::PadType::EXPLICIT};
    const size_t numOutChannels = 256;
    const std::vector<size_t> dilation{1, 1};
    const std::vector<size_t> kernelSize{1, 1};
    const std::vector<size_t> strides{1, 1};
    const std::vector<ptrdiff_t> padBegin{0, 0};
    const std::vector<ptrdiff_t> padEnd{0, 0};

    selectedType = ".*_i8";

    ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShapes)};
    const auto fq = ov::test::utils::make_fake_quantize(inputParams[0],
                                                      ov::element::f32,
                                                      256,
                                                      {1, 1, 1, 1},
                                                      {-12.8f},
                                                      {12.7f},
                                                      {-12.8f},
                                                      {12.7f});

    std::vector<std::shared_ptr<ov::Node>> branches(2);
    {
        ov::Strides strides{1, 1};
        ov::Shape pads_begin{0, 0}, pads_end{0, 0}, kernel{1, 1};
        branches[0] = std::make_shared<ov::op::v1::MaxPool>(fq, strides, pads_begin, pads_end, kernel);
    }
    {
        const auto fq_conv_data = ov::test::utils::make_fake_quantize(fq,
                                                                    ov::element::f32,
                                                                    256,
                                                                    {1, 1, 1, 1},
                                                                    {-12.8f},
                                                                    {12.7f},
                                                                    {-12.8f},
                                                                    {12.7f});

        const ov::Shape weights_const_shape = {numOutChannels, inputShapes[1], kernelSize[0], kernelSize[1]};
        const auto weights_const_values = std::vector<int>(ov::shape_size(weights_const_shape), 1);
        const auto weights_const = std::make_shared<ov::op::v0::Constant>(ov::element::i8, weights_const_shape, weights_const_values);

        const auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);

        const auto weights_multiply = std::make_shared<ov::opset10::Multiply>(
            weights_convert,
            std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                   ov::Shape{numOutChannels, 1, 1, 1},
                                                   std::vector<float>(numOutChannels, 1.0)));

        switch (type) {
        case nodeType::convolution: {
            branches[1] = ov::test::utils::make_convolution(fq_conv_data,
                                                            weights_multiply,
                                                            ov::element::f32,
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
            branches[1] = ov::test::utils::make_group_convolution(
                fq_conv_data,
                std::make_shared<ov::opset10::Reshape>(
                    weights_multiply,
                    std::make_shared<ov::op::v0::Constant>(
                        ov::element::i32,
                        ov::Shape{5},
                        std::vector<size_t>{1, numOutChannels, inputShapes[1], kernelSize[0], kernelSize[1]}),
                    true),
                ov::element::f32,
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

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{branches[0], branches[1]}, 1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};
    function = std::make_shared<ov::Model>(results, inputParams, "ConvWithZeroPointFuseSubgraphTest");
}

TEST_P(ConvWithZeroPointFuseSubgraphTest, CompareWithRefs) {
    run();

    CheckPluginRelatedResults(compiledModel, pluginTypeNode);
};

const ov::Shape inputShapes2D = {1, 32, 136, 136};

const auto params2DConv = ::testing::Combine(::testing::ValuesIn({nodeType::convolution, nodeType::groupConvolution}),
                                             ::testing::Values(inputShapes2D));

INSTANTIATE_TEST_SUITE_P(smoke_ConvWithZeroPointFuse,
                         ConvWithZeroPointFuseSubgraphTest,
                         params2DConv,
                         ConvWithZeroPointFuseSubgraphTest::getTestCaseName);

}  // namespace test
}  // namespace ov
