// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/binary_convolution.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/binary_convolution.hpp"

namespace ov {
namespace test {
std::string BinaryConvolutionLayerTest::getTestCaseName(const testing::TestParamInfo<binaryConvolutionTestParamsSet>& obj) {
    const auto& [bin_conv_params, model_type, shapes, target_device] = obj.param;

    const auto& [kernel, stride, pad_begin, padEnd, dilation, conv_out_channels, pad_type, pad_value] = bin_conv_params;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "KS=" << ov::test::utils::vec2str(kernel) << "_";
    result << "S=" << ov::test::utils::vec2str(stride) << "_";
    result << "PB=" << ov::test::utils::vec2str(pad_begin) << "_";
    result << "PE=" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "O=" << conv_out_channels << "_";
    result << "AP=" << pad_type << "_";
    result << "PV=" << pad_value << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void BinaryConvolutionLayerTest::SetUp() {
    const auto& [bin_conv_params, model_type, shapes, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;
    init_input_shapes(shapes);

    const auto& [kernel_size, strides, pads_begin, pads_end, dilations, num_out_channels, pad_type, pad_value] =
        bin_conv_params;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front())};
    params[0]->set_friendly_name("a_data_batch");

    // TODO: refactor build BinaryConvolution op to accept filters input as Parameter
    auto bin_conv =
        ov::test::utils::make_binary_convolution(params[0], kernel_size, strides, pads_begin, pads_end, dilations, pad_type, num_out_channels, pad_value);
    auto result = std::make_shared<ov::op::v0::Result>(bin_conv);
    function = std::make_shared<ov::Model>(ov::OutputVector{result}, params, "BinaryConvolution");
}

} // namespace test
} // namespace ov
