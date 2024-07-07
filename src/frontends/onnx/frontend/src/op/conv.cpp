// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils/conv_factory.hpp"
#include "utils/convpool.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
namespace detail {

std::shared_ptr<ov::Node> add_bias(const ov::Output<ov::Node>& ng_conv, const ov::Output<ov::Node>& bias) {
    const auto conv_shape = std::make_shared<v3::ShapeOf>(ng_conv);
    const auto conv_rank = std::make_shared<v3::ShapeOf>(conv_shape);

    return {std::make_shared<v1::Add>(ng_conv, reshape::reshape_channel_shaped_node_to_nchw(bias, conv_rank))};
}

ov::OutputVector conv(const ov::frontend::onnx::Node& node,
                      ov::Output<ov::Node> data,
                      ov::Output<ov::Node> filters,
                      ov::Output<ov::Node> bias) {
    // in the current implementation we assume that the data input rank is static
    // and only the 'batch' dimension can be dynamic
    const auto groups = node.get_attribute_value<int64_t>("group", 1);

    FRONT_END_GENERAL_CHECK(data.get_partial_shape().rank().is_static(),
                            "The input data tensor's rank has to be known (static)");

    const auto strides = convpool::get_strides(node);
    const auto dilations = convpool::get_dilations(node);
    const auto paddings = convpool::get_pads(node);
    const ov::op::PadType auto_pad_type = convpool::get_auto_pad(node);
    const auto& padding_below = paddings.first;
    const auto& padding_above = paddings.second;

    const auto conv_node = conv_factory::make_ng_convolution(data,
                                                             filters,
                                                             strides,
                                                             dilations,
                                                             padding_below,
                                                             padding_above,
                                                             groups,
                                                             auto_pad_type);

    // no bias param
    if (ov::op::util::is_null(bias)) {
        return {conv_node};
    } else {
        const auto& bias_ps = bias.get_partial_shape();

        FRONT_END_GENERAL_CHECK(bias_ps.rank().is_static() && bias_ps.rank().get_length() == 1,
                                "The bias input needs to be 1D vector");

        const std::string onnx_name = !node.get_name().empty() ? node.get_name() : node.output(0);
        conv_node->set_friendly_name(onnx_name + "/WithoutBiases");
        return {add_bias(conv_node, bias)};
    }
}
}  // namespace detail

ov::OutputVector conv(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector& inputs = node.get_ov_inputs();
    return detail::conv(node, inputs[0], inputs[1], inputs.size() < 3 ? std::make_shared<NullNode>() : inputs[2]);
}
ONNX_OP("Conv", OPSET_SINCE(1), ai_onnx::opset_1::conv);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
