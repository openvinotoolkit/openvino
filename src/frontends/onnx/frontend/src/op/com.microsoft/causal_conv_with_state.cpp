// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {
// y = DepthwiseConv(concat(past_state, x)), present_state = trailing (k - 1) * dilation positions of that concat.
ov::OutputVector causal_conv_with_state(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2, 4);
    const auto inputs = node.get_ov_inputs();

    CHECK_VALID_NODE(node,
                     node.get_attribute_value<int64_t>("ndim", 1) == 1,
                     "CausalConvWithState: only ndim=1 is supported");
    CHECK_VALID_NODE(node,
                     node.get_attribute_value<int64_t>("channels_last", 0) == 0,
                     "CausalConvWithState: channels_last=1 is not supported");
    CHECK_VALID_NODE(node,
                     node.get_attribute_value<int64_t>("state_window", 0) == 0,
                     "CausalConvWithState: state_window>0 is not supported");
    const auto state_channels_last = node.get_attribute_value<int64_t>("state_channels_last", -1);
    CHECK_VALID_NODE(node,
                     node.get_attribute_value<int64_t>("input_channel_offset", 0) == 0 &&
                         node.get_attribute_value<int64_t>("input_channels", 0) == 0 &&
                         node.get_attribute_value<int64_t>("state_update_capacity", 0) == 0 &&
                         (state_channels_last == -1 || state_channels_last == 0),
                     "CausalConvWithState: CUDA-only extended attributes are not supported");
    CHECK_VALID_NODE(node, node.get_outputs_size() <= 2, "CausalConvWithState: state_update output is not supported");

    const auto dilation = node.get_attribute_value<int64_t>("dilation", 1);
    CHECK_VALID_NODE(node, dilation >= 1, "CausalConvWithState: dilation must be >= 1");
    const auto activation = node.get_attribute_value<std::string>("activation", "none");
    CHECK_VALID_NODE(node,
                     activation == "none" || activation == "silu" || activation == "swish",
                     "CausalConvWithState: unsupported activation '",
                     activation,
                     "'");

    const auto& x = inputs[0];
    CHECK_VALID_NODE(node,
                     x.get_partial_shape().rank().compatible(3),
                     "CausalConvWithState: input must be [batch, channels, length] for ndim=1");
    const auto& weight = inputs[1];
    const auto& weight_shape = weight.get_partial_shape();
    CHECK_VALID_NODE(
        node,
        weight_shape.rank().is_static() && weight_shape.rank().get_length() == 3 && weight_shape[2].is_static(),
        "CausalConvWithState: weight must be [channels, 1, kernel_size] with a static kernel_size");
    const auto kernel_size = weight_shape[2].get_length();
    CHECK_VALID_NODE(node,
                     kernel_size >= 1 && kernel_size - 1 <= std::numeric_limits<int64_t>::max() / dilation,
                     "CausalConvWithState: kernel_size must be positive and state length must fit in int64_t");
    const int64_t state_len = (kernel_size - 1) * dilation;
    const auto state_length_attr = node.get_attribute_value<int64_t>("state_length", 0);
    CHECK_VALID_NODE(node,
                     state_length_attr == 0 || state_length_attr == state_len,
                     "CausalConvWithState: state_length different from (kernel_size - 1) * dilation is not supported");

    const auto axis_2 = v0::Constant::create(element::i64, Shape{1}, {2});
    ov::Output<ov::Node> past;
    if (inputs.size() > 3 && !ov::op::util::is_null(inputs[3])) {
        past = inputs[3];
    } else {
        const auto batch_channels = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(x),
                                                                 v0::Constant::create(element::i64, Shape{2}, {0, 1}),
                                                                 v0::Constant::create(element::i64, Shape{}, {0}));
        const auto zeros_shape = std::make_shared<v0::Concat>(
            ov::OutputVector{batch_channels, v0::Constant::create(element::i64, Shape{1}, {state_len})},
            0);
        const auto zero = std::make_shared<v1::ConvertLike>(v0::Constant::create(element::f32, Shape{}, {0}), x);
        past = std::make_shared<v3::Broadcast>(zero, zeros_shape);
    }

    const auto full = std::make_shared<v0::Concat>(ov::OutputVector{past, x}, 2);

    // [C, 1, K] -> [C, 1, 1, K] (groups = C, one input and one output channel per group)
    const auto filters = std::make_shared<v0::Unsqueeze>(weight, v0::Constant::create(element::i64, Shape{1}, {1}));
    ov::Output<ov::Node> y = std::make_shared<v1::GroupConvolution>(full,
                                                                    filters,
                                                                    Strides{1},
                                                                    CoordinateDiff{0},
                                                                    CoordinateDiff{0},
                                                                    Strides{static_cast<size_t>(dilation)});

    if (inputs.size() > 2 && !ov::op::util::is_null(inputs[2])) {
        const auto bias = std::make_shared<v0::Unsqueeze>(inputs[2], v0::Constant::create(element::i64, Shape{1}, {1}));
        y = std::make_shared<v1::Add>(y, bias);
    }
    if (activation != "none") {
        y = std::make_shared<v4::Swish>(y);
    }

    const auto present = std::make_shared<v8::Slice>(
        full,
        v0::Constant::create(element::i64,
                             Shape{1},
                             {state_len > 0 ? -state_len : std::numeric_limits<int64_t>::max()}),
        v0::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()}),
        v0::Constant::create(element::i64, Shape{1}, {1}),
        axis_2);

    return {y, present};
}

ONNX_OP("CausalConvWithState", OPSET_SINCE(1), com_microsoft::opset_1::causal_conv_with_state, MICROSOFT_DOMAIN);
}  // namespace ov::frontend::onnx::com_microsoft::opset_1
