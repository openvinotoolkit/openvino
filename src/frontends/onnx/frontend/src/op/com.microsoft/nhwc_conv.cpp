// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx::opset_1::detail {
extern ov::OutputVector conv(const ov::frontend::onnx::Node& node,
                             ov::Output<ov::Node> data,
                             ov::Output<ov::Node> filters,
                             ov::Output<ov::Node> bias);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1::detail

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector nhwc_conv(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftnhwcconv
    // Same weights layout (M x C/group x k1 x ... x kn) and attributes as the standard Conv, but the data input
    // and the output are in the channels-last N x D1 x ... x Dn x C layout. The input is transposed to
    // N x C x D1 x ... x Dn, the standard Conv translation is reused and the result is transposed back.
    common::default_op_checks(node, 2, 3);

    const auto inputs = node.get_ov_inputs();
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const ov::Output<ov::Node> bias = inputs.size() > 2 ? inputs[2] : std::make_shared<NullNode>()->output(0);

    const auto x_rank = x.get_partial_shape().rank();
    CHECK_VALID_NODE(node, x_rank.is_static(), "The input data tensor's rank has to be known (static)");
    const auto rank = x_rank.get_length();
    CHECK_VALID_NODE(node, rank >= 3, "NhwcConv expects an input of rank 3 or more. Got: ", rank);

    // N x D1 x ... x Dn x C -> N x C x D1 x ... x Dn
    std::vector<int64_t> to_channels_first{0, rank - 1};
    // N x C x D1 x ... x Dn -> N x D1 x ... x Dn x C
    std::vector<int64_t> to_channels_last{0};
    for (int64_t axis = 1; axis < rank - 1; ++axis) {
        to_channels_first.push_back(axis);
        to_channels_last.push_back(axis + 1);
    }
    to_channels_last.push_back(1);

    const auto x_channels_first = std::make_shared<v1::Transpose>(
        x,
        v0::Constant::create(ov::element::i64, {to_channels_first.size()}, to_channels_first));
    const auto conv = ai_onnx::opset_1::detail::conv(node, x_channels_first, w, bias).at(0);
    return {std::make_shared<v1::Transpose>(
        conv,
        v0::Constant::create(ov::element::i64, {to_channels_last.size()}, to_channels_last))};
}

ONNX_OP("NhwcConv", OPSET_SINCE(1), com_microsoft::opset_1::nhwc_conv, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
