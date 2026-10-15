// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/relu.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> layer_norm(const NodeContext& context,
                        const Output<Node>& input,
                        const Output<Node>& weight,
                        const Output<Node>& bias,
                        float eps) {
    // The input is always normalized over the last dimension, which is embed_dim.
    const auto axes = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    Output<Node> res = context.mark_node(std::make_shared<v6::MVN>(input, axes, true, eps, MVNEpsMode::INSIDE_SQRT));
    const auto weight_conv = context.mark_node(std::make_shared<v1::ConvertLike>(weight, res));
    res = context.mark_node(std::make_shared<v1::Multiply>(res, weight_conv));
    const auto bias_conv = context.mark_node(std::make_shared<v1::ConvertLike>(bias, res));
    return context.mark_node(std::make_shared<v1::Add>(res, bias_conv));
}

Output<Node> linear(const NodeContext& context,
                    const Output<Node>& input,
                    const Output<Node>& weight,
                    const Output<Node>& bias) {
    const auto matmul = context.mark_node(std::make_shared<v0::MatMul>(input, weight, false, true));
    return context.mark_node(std::make_shared<v1::Add>(matmul, bias));
}
}  // namespace

OutputVector translate_transformer_encoder_layer_fwd(const NodeContext& context) {
    /*
    aten::_transformer_encoder_layer_fwd(
        Tensor src,
        int embed_dim,
        int num_heads,
        Tensor qkv_weight,
        Tensor qkv_bias,
        Tensor proj_weight,
        Tensor proj_bias,
        bool use_gelu,
        bool norm_first,
        float eps,
        Tensor norm_weight_1,
        Tensor norm_bias_1,
        Tensor norm_weight_2,
        Tensor norm_bias_2,
        Tensor ffn_weight_1,
        Tensor ffn_bias_1,
        Tensor ffn_weight_2,
        Tensor ffn_bias_2,
        Optional[Tensor] mask = None,
        Optional[int64] mask_type = None
    )
    */
    // The mask and the mask type are missing in the graphs produced by torch.export.
    num_inputs_check(context, 18, 20);
    const auto src = context.get_input(0);
    const auto embed_dim = context.get_input(1);
    const auto num_heads = context.get_input(2);
    const auto qkv_weight = context.get_input(3);
    const auto qkv_bias = context.get_input(4);
    const auto proj_weight = context.get_input(5);
    const auto proj_bias = context.get_input(6);
    const auto use_gelu = context.const_input<bool>(7);
    const auto norm_first = context.const_input<bool>(8);
    const auto eps = context.const_input<float>(9);
    const auto norm_weight_1 = context.get_input(10);
    const auto norm_bias_1 = context.get_input(11);
    const auto norm_weight_2 = context.get_input(12);
    const auto norm_bias_2 = context.get_input(13);
    const auto ffn_weight_1 = context.get_input(14);
    const auto ffn_bias_1 = context.get_input(15);
    const auto ffn_weight_2 = context.get_input(16);
    const auto ffn_bias_2 = context.get_input(17);

    Output<Node> mask;
    // Mask type 0 is a source mask of shape [sequence, sequence], which is broadcastable to the
    // attention weights, so it is a safe default when the type is not provided.
    int64_t mask_type = 0;
    if (context.get_input_size() > 18 && !context.input_is_none(18)) {
        mask = context.get_input(18);
        if (context.get_input_size() > 19 && !context.input_is_none(19)) {
            mask_type = context.const_input<int64_t>(19);
        }
    }

    Output<Node> x = src;
    if (norm_first) {
        x = layer_norm(context, x, norm_weight_1, norm_bias_1, eps);
    }
    x = build_multi_head_attention(context,
                                   x,
                                   x,
                                   x,
                                   embed_dim,
                                   num_heads,
                                   qkv_weight,
                                   qkv_bias,
                                   proj_weight,
                                   proj_bias,
                                   mask,
                                   mask_type,
                                   /* need_weights */ false,
                                   /* average_weights */ false)
            .first;
    x = context.mark_node(std::make_shared<v1::Add>(x, src));
    if (!norm_first) {
        x = layer_norm(context, x, norm_weight_1, norm_bias_1, eps);
    }

    const auto pre_ffn_res = x;
    if (norm_first) {
        x = layer_norm(context, x, norm_weight_2, norm_bias_2, eps);
    }
    x = linear(context, x, ffn_weight_1, ffn_bias_1);
    if (use_gelu) {
        x = context.mark_node(std::make_shared<v7::Gelu>(x, GeluApproximationMode::ERF));
    } else {
        x = context.mark_node(std::make_shared<v0::Relu>(x));
    }
    x = linear(context, x, ffn_weight_2, ffn_bias_2);
    x = context.mark_node(std::make_shared<v1::Add>(x, pre_ffn_res));
    if (!norm_first) {
        x = layer_norm(context, x, norm_weight_2, norm_bias_2, eps);
    }
    return {x};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
