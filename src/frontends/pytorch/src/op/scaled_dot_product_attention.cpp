// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/scaled_dot_product_attention.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/unsqueeze.hpp"
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
    if (!context.input_is_none(6)) {
        if (inputs.size() < 4) {
            // need to fill a gap in inputs with scalar 0 to be able to pass one extra input after that
            auto zero = op::v0::Constant::create(element::f32, Shape{}, {0});
            inputs.push_back(context.mark_node(std::make_shared<v1::ConvertLike>(zero, query)));
        }
        inputs.push_back(context.mark_node(std::make_shared<v1::ConvertLike>(context.get_input(6), query)));
    } else if (context.has_attribute("scale")) {
        const auto scale = context.get_input("scale");
        if (inputs.size() < 4) {
            auto zero = op::v0::Constant::create(element::f32, Shape{}, {0});
            inputs.push_back(context.mark_node(std::make_shared<v1::ConvertLike>(zero, query)));
        }
        inputs.push_back(context.mark_node(std::make_shared<v1::ConvertLike>(scale, query)));
    }
    if (!context.input_is_none(7)) {
        auto enable_gqa = context.const_input<bool>(7);
        PYTORCH_OP_CONVERSION_CHECK(enable_gqa == false,
                                    "Grouped Query Attention is not supported for SDPA operation.");
    }

    return context.mark_node(std::make_shared<v13::ScaledDotProductAttention>(inputs, is_causal));
}

OutputVector translate_scaled_dot_product_attention(const NodeContext& context) {
    // aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float
    // dropout_p=0., bool is_causal=False, float scale=None, bool enable_gqa=False)
    num_inputs_check(context, 3, 8);
    return {translate_scaled_dot_product_attention_common(context)};
};

OutputVector translate_scaled_dot_product_attention_fx(const NodeContext& context) {
    // torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(arg1_1, arg2_1, arg3_1, 0.0, True, attn_mask =
    // arg0_1, scale = 5.0)
    // aten._scaled_dot_product_flash_attention.default(arg0_1, arg1_1, arg2_1, 0.0, True, scale = 5.0)
    num_inputs_check(context, 3, 5);
    const auto query = context.get_input(0);
    const auto key = context.get_input(1);
    const auto value = context.get_input(2);
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    OutputVector inputs{query, key, value};
    // Index 3 is dropout
    auto causal = false;
    if (context.has_attribute("is_causal")) {
        causal = context.get_attribute<bool>("scale");
    } else if (!context.input_is_none(4)) {
        causal = context.const_input<bool>(4);
    }
    if (context.has_attribute("attn_mask")) {
        const auto attn_mask = context.get_input("attn_mask");
        inputs.push_back(attn_mask);
    } else if (context.has_attribute("scale")) {
        // if scale exist but attn_mask no, add zero input to fill in gap
        inputs.push_back(context.mark_node(std::make_shared<v1::ConvertLike>(zero, query)));
    }
    if (context.has_attribute("scale")) {
        const auto scale = context.get_input("scale");
        inputs.push_back(context.mark_node(std::make_shared<v1::ConvertLike>(scale, query)));
    }
    auto sdpa = context.mark_node(std::make_shared<v13::ScaledDotProductAttention>(inputs, causal));
    return {context.mark_node(make_list_construct({sdpa}))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
