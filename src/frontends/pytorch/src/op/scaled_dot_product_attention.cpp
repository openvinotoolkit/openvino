// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/scaled_dot_product_attention.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

std::shared_ptr<ov::Node> translate_scaled_dot_product_attention_common(const NodeContext& context) {
    auto query = context.get_input(0);
    auto key = context.get_input(1);
    auto value = context.get_input(2);

    auto is_causal = context.input_is_none(5) ? false : context.const_input<bool>(5);
    OutputVector inputs = {query, key, value};  // mandatory inputs

    if (!context.input_is_none(3))
        inputs.push_back(context.get_input(3));
    else if (!context.input_is_none(6)) {
        // need to fill a gap in inputs with scalar 0 to be able to pass one extra input after that
        auto zero = op::v0::Constant::create(element::f32, Shape{}, {0});
        inputs.push_back(context.mark_node(std::make_shared<v1::ConvertLike>(zero, query)));
    }
    if (!context.input_is_none(6))
        inputs.push_back(context.get_input(6));

    return context.mark_node(std::make_shared<v13::ScaledDotProductAttention>(inputs, is_causal));
}

OutputVector translate_scaled_dot_product_attention(const NodeContext& context) {
    // aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float
    // dropout_p=0., bool is_causal=False, float scale=None)
    num_inputs_check(context, 6, 7);
    return {translate_scaled_dot_product_attention_common(context)};
};

OutputVector translate_scaled_dot_product_attention_fx(const NodeContext& context) {
    // aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float
    // dropout_p=0., bool is_causal=False)  TODO: Scale parameter?
    num_inputs_check(context, 3, 6);  // TODO: Set 7 instead of 6 if `scale` argument supported in FX.
    auto output = translate_scaled_dot_product_attention_common(context);
    // TODO: scaled_dot_product_flash_attention has 9 outputs but for most cases only
    // the first input is used. Rest of the outputs should be returned properly as
    // needed.
    ov::OutputVector out_vec;
    out_vec.push_back(output);
    return {context.mark_node(make_list_construct(out_vec))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
