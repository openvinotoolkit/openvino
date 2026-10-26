// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/deformable_convolution.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils/convpool.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector deform_conv(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector& inputs = node.get_ov_inputs();
    FRONT_END_GENERAL_CHECK(inputs.size() >= 3 && inputs.size() <= 5,
                            "DeformConv expects between 3 and 5 inputs, got: ",
                            inputs.size());

    // ONNX DeformConv inputs: X, W, offset, [B], [mask]
    const auto& data = inputs.at(0);
    const auto& filters = inputs.at(1);
    const auto& offsets = inputs.at(2);
    const auto bias = inputs.size() >= 4 ? inputs.at(3) : ov::Output<ov::Node>{std::make_shared<NullNode>()};
    const auto mask = inputs.size() == 5 ? inputs.at(4) : ov::Output<ov::Node>{std::make_shared<NullNode>()};

    const auto strides = convpool::get_strides(node);
    const auto dilations = convpool::get_dilations(node);
    const auto paddings = convpool::get_pads(node);
    const auto auto_pad_type = convpool::get_auto_pad(node);

    const auto group = node.get_attribute_value<int64_t>("group", 1);
    const auto offset_group = node.get_attribute_value<int64_t>("offset_group", 1);

    FRONT_END_GENERAL_CHECK(offset_group > 0, "DeformConv offset_group must be greater than zero");
    const auto& data_shape = data.get_partial_shape();
    if (data_shape.rank().is_static() && data_shape.rank().get_length() > 1 && data_shape[1].is_static()) {
        FRONT_END_GENERAL_CHECK(data_shape[1].get_length() % offset_group == 0,
                                "DeformConv input channels (",
                                data_shape[1].get_length(),
                                ") must be evenly divisible by offset_group (",
                                offset_group,
                                ")");
    }

    // ONNX samples zero when the location lies outside of the padded input tensor,
    // which corresponds to bilinear interpolation on the zero-padded feature map.
    constexpr bool bilinear_interpolation_pad = true;

    std::shared_ptr<ov::Node> deformable_conv;
    if (ov::op::util::is_null(mask)) {
        deformable_conv = std::make_shared<v8::DeformableConvolution>(data,
                                                                      offsets,
                                                                      filters,
                                                                      strides,
                                                                      paddings.first,
                                                                      paddings.second,
                                                                      dilations,
                                                                      auto_pad_type,
                                                                      group,
                                                                      offset_group,
                                                                      bilinear_interpolation_pad);
    } else {
        deformable_conv = std::make_shared<v8::DeformableConvolution>(data,
                                                                      offsets,
                                                                      filters,
                                                                      mask,
                                                                      strides,
                                                                      paddings.first,
                                                                      paddings.second,
                                                                      dilations,
                                                                      auto_pad_type,
                                                                      group,
                                                                      offset_group,
                                                                      bilinear_interpolation_pad);
    }

    if (ov::op::util::is_null(bias)) {
        return {deformable_conv};
    }

    const auto conv_rank = std::make_shared<v3::ShapeOf>(std::make_shared<v3::ShapeOf>(deformable_conv));
    const auto reshaped_bias = reshape::reshape_channel_shaped_node_to_nchw(bias, conv_rank);
    return {std::make_shared<v1::Add>(deformable_conv, reshaped_bias)};
}

ov::OutputVector deform_conv_before_opset_19(const ov::frontend::onnx::Node&) {
    FRONT_END_GENERAL_CHECK(false, "DeformConv is only defined starting with ONNX opset 19");
    return {};
}

// Keep older ONNX operator sets usable; DeformConv was introduced in opset 19.
static bool onnx_op_reg_before_opset_19 =
    ONNX_OP_M("DeformConv", OPSET_RANGE(1, 18), ai_onnx::opset_1::deform_conv_before_opset_19);
ONNX_OP("DeformConv", OPSET_SINCE(19), ai_onnx::opset_1::deform_conv);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1
