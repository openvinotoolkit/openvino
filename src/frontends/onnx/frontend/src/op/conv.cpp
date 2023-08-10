// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/conv.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "onnx_import/core/null_node.hpp"
#include "utils/conv_factory.hpp"
#include "utils/convpool.hpp"
#include "utils/reshape.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
namespace detail {

std::shared_ptr<ngraph::Node> add_bias(const Output<ngraph::Node>& ng_conv, const Output<ngraph::Node>& bias) {
    const auto conv_shape = std::make_shared<default_opset::ShapeOf>(ng_conv);
    const auto conv_rank = std::make_shared<default_opset::ShapeOf>(conv_shape);

    return {
        std::make_shared<default_opset::Add>(ng_conv, reshape::reshape_channel_shaped_node_to_nchw(bias, conv_rank))};
}

OutputVector conv(const Node& node,
                  Output<ngraph::Node> data,
                  Output<ngraph::Node> filters,
                  Output<ngraph::Node> bias) {
    // in the current implementation we assume that the data input rank is static
    // and only the 'batch' dimension can be dynamic
    const auto groups = node.get_attribute_value<int64_t>("group", 1);

    NGRAPH_CHECK(data.get_partial_shape().rank().is_static(), "The input data tensor's rank has to be known (static)");

    const auto strides = convpool::get_strides(node);
    const auto dilations = convpool::get_dilations(node);
    const auto paddings = convpool::get_pads(node);
    const ngraph::op::PadType auto_pad_type = convpool::get_auto_pad(node);
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
    if (ngraph::op::is_null(bias)) {
        return {conv_node};
    } else {
        const auto& bias_ps = bias.get_partial_shape();

        NGRAPH_CHECK(bias_ps.rank().is_static() && bias_ps.rank().get_length() == 1,
                     "The bias input needs to be 1D vector");

        const std::string onnx_name = !node.get_name().empty() ? node.get_name() : node.output(0);
        conv_node->set_friendly_name(onnx_name + "/WithoutBiases");
        return {add_bias(conv_node, bias)};
    }
}
}  // namespace detail

OutputVector conv(const Node& node) {
    const OutputVector& inputs = node.get_ng_inputs();
    return detail::conv(node, inputs[0], inputs[1], inputs.size() < 3 ? std::make_shared<NullNode>() : inputs[2]);
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
