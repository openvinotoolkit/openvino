// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_opt/binary_convolution.hpp"

#include "ngraph_functions/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
std::string BinaryConvolutionLayerTest::getTestCaseName(const testing::TestParamInfo<binaryConvolutionTestParamsSet>& obj) {
    binConvSpecificParams binConvParams;
    ov::element::Type netType;
    ov::element::Type inPrc, outPrc;
    std::vector<InputShape> shapes;
    std::string targetDevice;

    std::tie(binConvParams, netType, inPrc, outPrc, shapes, targetDevice) = obj.param;

    ov::op::PadType padType;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    float padValue;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, padValue) = binConvParams;

    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "KS=" << ov::test::utils::vec2str(kernel) << "_";
    result << "S=" << ov::test::utils::vec2str(stride) << "_";
    result << "PB=" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE=" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "PV=" << padValue << "_";
    result << "netPRC=" << netType.get_type_name() << "_";
    result << "inPRC=" << inPrc.get_type_name() << "_";
    result << "outPRC=" << outPrc.get_type_name() << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void BinaryConvolutionLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto itTargetShape = targetInputStaticShapes.begin();
    for (auto&& param : function->get_parameters()) {
        auto tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(), *itTargetShape++, 1, 0, 1, 7235346);
        inputs.insert({param, tensor});
    }
}

void BinaryConvolutionLayerTest::SetUp() {
    binConvSpecificParams binConvParams;
    ov::element::Type net_type;
    std::vector<InputShape> shapes;

    std::tie(binConvParams, net_type, inType, outType, shapes, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    ov::op::PadType padType;
    std::vector<size_t> kernelSize, strides, dilations;
    std::vector<ptrdiff_t> padsBegin, padsEnd;
    size_t numOutChannels;
    float padValue;
    std::tie(kernelSize, strides, padsBegin, padsEnd, dilations, numOutChannels, padType, padValue) = binConvParams;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(net_type, inputDynamicShapes.front())};
    params[0]->set_friendly_name("a_data_batch");

    // TODO: refactor build BinaryConvolution op to accept filters input as Parameter
    auto binConv = ngraph::builder::makeBinaryConvolution(params[0], kernelSize, strides, padsBegin, padsEnd, dilations, padType, numOutChannels, padValue);
    auto result = std::make_shared<ov::op::v0::Result>(binConv);
    function = std::make_shared<ov::Model>(ov::OutputVector{result}, params, "BinaryConvolution");
}

} // namespace test
} // namespace ov
