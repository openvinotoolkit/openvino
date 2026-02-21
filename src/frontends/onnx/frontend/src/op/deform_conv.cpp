// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/deformable_convolution.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/util/op_types.hpp"
#include "utils/common.hpp"
#include "utils/convpool.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_19 {

ov::OutputVector deform_conv(const ov::frontend::onnx::Node& node) {
    // ONNX DeformConv inputs: X(0), W(1), offset(2), B(3, optional), mask(4, optional)
    // OpenVINO v8::DeformableConvolution inputs: data, offsets, filters, [mask]
    common::default_op_checks(node, 3);

    const auto inputs = node.get_ov_inputs();
    const auto& X = inputs[0];       // data
    const auto& W = inputs[1];       // filters/weights
    const auto& offset = inputs[2];  // offsets

    const auto strides = convpool::get_strides(node);
    const auto dilations = convpool::get_dilations(node);
    const auto paddings = convpool::get_pads(node);

    const auto group = node.get_attribute_value<int64_t>("group", 1);
    const auto offset_group = node.get_attribute_value<int64_t>("offset_group", 1);

    const bool has_bias = common::is_input_valid(node, 3);
    const bool has_mask = common::is_input_valid(node, 4);

    ov::Output<ov::Node> deform_conv_node;
    if (has_mask) {
        deform_conv_node = std::make_shared<v8::DeformableConvolution>(X,
                                                                       offset,
                                                                       W,
                                                                       inputs[4],
                                                                       strides,
                                                                       paddings.first,
                                                                       paddings.second,
                                                                       dilations,
                                                                       ov::op::PadType::EXPLICIT,
                                                                       group,
                                                                       offset_group);
    } else {
        deform_conv_node = std::make_shared<v8::DeformableConvolution>(X,
                                                                       offset,
                                                                       W,
                                                                       strides,
                                                                       paddings.first,
                                                                       paddings.second,
                                                                       dilations,
                                                                       ov::op::PadType::EXPLICIT,
                                                                       group,
                                                                       offset_group);
    }

    if (has_bias) {
        const auto& bias = inputs[3];
        const auto conv_shape = std::make_shared<v3::ShapeOf>(deform_conv_node);
        const auto conv_rank = std::make_shared<v3::ShapeOf>(conv_shape);

        const std::string onnx_name = !node.get_name().empty() ? node.get_name() : node.output(0);
        deform_conv_node.get_node_shared_ptr()->set_friendly_name(onnx_name + "/WithoutBiases");
        return {std::make_shared<v1::Add>(deform_conv_node,
                                          reshape::reshape_channel_shaped_node_to_nchw(bias, conv_rank))};
    }

    return {deform_conv_node};
}

ONNX_OP("DeformConv", OPSET_SINCE(19), ai_onnx::opset_19::deform_conv);

}  // namespace opset_19
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
