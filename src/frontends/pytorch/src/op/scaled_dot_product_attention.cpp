// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/scaled_dot_product_attention.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
ov::OutputVector prepare_inputs_to_sdpa(const NodeContext& context,
                                        const Output<Node>& query,  // [B, Hq, T, D]
                                        const Output<Node>& key,    // [B, Hk, T, D]
                                        const Output<Node>& value,  // [B, Hk, T, D]
                                        const Output<Node>& scale,  // optional
                                        const Output<Node>& attn_mask) {
    OutputVector inputs = {query, key, value};
    if (attn_mask.get_node()) {
        inputs.push_back(attn_mask);
    }
    if (scale.get_node()) {
        if (!attn_mask.get_node()) {
            auto zero = op::v0::Constant::create(element::f32, Shape{}, {0});
            auto attn_mask_default = context.mark_node(std::make_shared<v1::ConvertLike>(zero, query));
            inputs.push_back(attn_mask_default);
        }
        inputs.push_back(scale);
    }

    return inputs;
}

ov::Output<ov::Node> get_dim(const NodeContext& context, const Output<Node>& shape, int64_t idx) {
    auto begin = v0::Constant::create(element::i64, Shape{1}, {idx});
    auto end = v0::Constant::create(element::i64, Shape{1}, {idx + 1});
    auto stride = v0::Constant::create(element::i64, Shape{1}, {1});

    auto res_dim = context.mark_node(std::make_shared<v1::StridedSlice>(shape,
                                                                        begin,
                                                                        end,
                                                                        stride,
                                                                        std::vector<int64_t>{0},
                                                                        std::vector<int64_t>{0},
                                                                        std::vector<int64_t>{0},
                                                                        std::vector<int64_t>{0}));

    return res_dim;
}

std::shared_ptr<Node> decompose_gqa(const NodeContext& context,
                                    const Output<Node>& query,      // [B, Hq, T, D]
                                    const Output<Node>& key,        // [B, Hk, T, D]
                                    const Output<Node>& value,      // [B, Hk, T, D]
                                    const Output<Node>& scale,      // optional
                                    const Output<Node>& attn_mask,  // optional
                                    bool is_casual) {
    auto q_shape = context.mark_node(std::make_shared<v3::ShapeOf>(query, element::i64));
    auto k_shape = context.mark_node(std::make_shared<v3::ShapeOf>(key, element::i64));

    auto B = get_dim(context, q_shape, 0);
    auto Hq = get_dim(context, q_shape, 1);
    auto T = get_dim(context, q_shape, 2);
    auto D = get_dim(context, q_shape, 3);
    auto Hk = get_dim(context, k_shape, 1);

    // Compute group_size = Hq / Hk
    auto group_size = context.mark_node(std::make_shared<v1::Divide>(Hq, Hk));

    // Reshape query: [B, Hq, T, D] -> [B, Hk, group_size, T, D]
    auto reshape_q_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{B, Hk, group_size, T, D}, 0));
    auto q_reshaped = context.mark_node(std::make_shared<v1::Reshape>(query, reshape_q_shape, true));

    // Unsqueeze key, value: [B, Hk, T, D] -> [B, Hk, 1, T, D]
    auto axis = v0::Constant::create(element::i64, Shape{1}, {2});
    auto k_unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(key, axis));
    auto v_unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(value, axis));

    auto inputs = prepare_inputs_to_sdpa(context, q_reshaped, k_unsqueezed, v_unsqueezed, scale, attn_mask);
    // ScaledDotProductAttention (v13)
    auto attn = context.mark_node(std::make_shared<v13::ScaledDotProductAttention>(inputs, is_casual));

    // Reshape back: [B, Hk, group_size, T, D] -> [B, Hq, T, D]
    auto reshape_out_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{B, Hq, T, D}, 0));
    return context.mark_node(std::make_shared<v1::Reshape>(attn, reshape_out_shape, true));
}

}  // namespace

std::shared_ptr<ov::Node> translate_scaled_dot_product_attention_common(const NodeContext& context) {
    auto query = context.get_input(0);
    auto key = context.get_input(1);
    auto value = context.get_input(2);

    auto is_causal = context.input_is_none(5) ? false : context.const_input<bool>(5);

    ov::Output<ov::Node> attn_mask{};
    ov::Output<ov::Node> scale{};
    if (!context.input_is_none(3)) {
        attn_mask = context.get_input(3);
    }
    if (!context.input_is_none(6)) {
        scale = context.mark_node(std::make_shared<v1::ConvertLike>(context.get_input(6), query));
    } else if (context.has_attribute("scale")) {
        scale = context.get_input("scale");
        scale = context.mark_node(std::make_shared<v1::ConvertLike>(scale, query));
    }
    if (!context.input_is_none(7) && context.const_input<bool>(7)) {
        return decompose_gqa(context, query, key, value, scale, attn_mask, is_causal);
    }

    auto inputs = prepare_inputs_to_sdpa(context, query, key, value, scale, attn_mask);
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
