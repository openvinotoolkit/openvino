// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_embedding(const NodeContext& context) {
    // aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool
    // sparse=False)
    num_inputs_check(context, 2, 5);
    auto data = context.get_input(0);
    auto indices = context.get_input(1);
    indices = context.mark_node(std::make_shared<ov::op::v0::Convert>(indices, element::i32));
    // skip parameters 2, 3, 4 used only during training:
    // padding_idx - if specified, the entries at padding_idx do not contribute to the gradient
    // scale_grad_by_freq - if given, this will scale gradients by the inverse of frequency of
    //                      the words in the mini-batch.
    // sparse - if True, gradient will be represented as sparse tensor
    auto axis_0 = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{}, {0}));
    return {context.mark_node(std::make_shared<ov::op::v8::Gather>(data, indices, axis_0))};
};

OutputVector translate_embedding_ext(const NodeContext& context) {
    // used in 16-bit patching
    num_inputs_check(context, 2, 5);
    auto data = context.get_input(0);
    if (data.get_element_type() != element::f32) {
        data = context.mark_node(std::make_shared<ov::op::v0::Convert>(data, element::f32));
    }
    auto indices = context.get_input(1);
    indices = context.mark_node(std::make_shared<ov::op::v0::Convert>(indices, element::i32));
    auto axis_0 = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{}, {0}));
    return {context.mark_node(std::make_shared<ov::op::v8::Gather>(data, indices, axis_0))};
};

OutputVector translate_embedding_ct(const NodeContext& context) {
    // ov_ext::ct_embedding(weight_packed, weight_scale, group_size, sym,
    //                      indices, weight_zero_point?)
    // compressed-tensors pack-quantized int4 embedding lookup.
    num_inputs_check(context, 5, 6);
    auto weight_packed = context.get_input(0);
    auto scales = context.get_input(1);
    const auto group_size = context.const_input<int64_t>(2);
    const auto sym = context.const_input<bool>(3);
    auto indices = context.get_input(4);

    Output<Node> zp;
    if (!sym) {
        FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(5), "CT asymmetric embedding requires weight_zero_point.");
        zp = context.get_input(5);
    }
    // Dequantized as [embedding_dim, num_embeddings]; scales drive the output type (f32).
    auto table = dequantize_ct_weight(context, weight_packed, scales, sym, group_size, scales, zp);
    // Transpose to the [num_embeddings, embedding_dim] lookup table and gather rows.
    auto perm = ov::op::v0::Constant::create(element::i32, {2}, std::vector<int32_t>{1, 0});
    table = context.mark_node(std::make_shared<ov::op::v1::Transpose>(table, perm));
    indices = context.mark_node(std::make_shared<ov::op::v0::Convert>(indices, element::i32));
    auto axis_0 = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
    return {context.mark_node(std::make_shared<ov::op::v8::Gather>(table, indices, axis_0))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
