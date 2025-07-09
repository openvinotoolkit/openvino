// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/group_convolution_backprop_data.hpp"

#include "common_test_utils/node_builders/group_convolution_backprop_data.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"

namespace ov {
namespace test {
std::string GroupConvBackpropLayerTest::getTestCaseName(testing::TestParamInfo<groupConvBackpropLayerTestParamsSet> obj) {
    groupConvBackpropSpecificParams group_conv_backprop_data_params;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    ov::Shape output_shape;
    std::string target_device;
    std::tie(group_conv_backprop_data_params, model_type, shapes, output_shape, target_device) = obj.param;
    ov::op::PadType pad_type;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> pad_begin, pad_end, out_padding;
    size_t conv_out_channels, num_groups;
    std::tie(kernel, stride, pad_begin, pad_end, dilation, conv_out_channels, num_groups, pad_type, out_padding) = group_conv_backprop_data_params;

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
    result << "OS=" << ov::test::utils::vec2str(output_shape) << "_";
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(pad_begin) << "_";
    result << "PE" << ov::test::utils::vec2str(pad_end) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "OP=" << ov::test::utils::vec2str(out_padding) << "_";
    result << "O=" << conv_out_channels << "_";
    result << "G=" << num_groups << "_";
    result << "AP=" << pad_type << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void GroupConvBackpropLayerTest::SetUp() {
    groupConvBackpropSpecificParams group_conv_backprop_data_params;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    ov::Shape output_shape;
    std::tie(group_conv_backprop_data_params, model_type, shapes, output_shape, targetDevice) = this->GetParam();
    ov::op::PadType pad_type;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> pad_begin, pad_end, out_padding;
    size_t conv_out_channels, num_groups;
    std::tie(kernel, stride, pad_begin, pad_end, dilation, conv_out_channels, num_groups, pad_type, out_padding) = group_conv_backprop_data_params;
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    std::shared_ptr<ov::Node> group_conv;
    if (!output_shape.empty()) {
        auto outShape = ov::op::v0::Constant::create(ov::element::i64, {output_shape.size()}, output_shape);
        group_conv = ov::test::utils::make_group_convolution_backprop_data(
            param, outShape, model_type, kernel, stride, pad_begin, pad_end, dilation, pad_type, conv_out_channels, num_groups, false, out_padding);
    } else {
        group_conv = ov::test::utils::make_group_convolution_backprop_data(
            param, model_type, kernel, stride, pad_begin, pad_end, dilation, pad_type, conv_out_channels, num_groups, false, out_padding);
    }

    auto result = std::make_shared<ov::op::v0::Result>(group_conv);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "GroupConvolutionBackpropData");
}
}  // namespace test
}  // namespace ov
