// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Translates openvino.gdn_attention (vLLM Gated DeltaNet); state and metadata come
// from "__pa__gdn__" side-channel Parameters bound in torchdynamo/vllm/side_channel.py.

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {

Output<Node> to_type(const Output<Node>& t, const element::Type& et) {
    if (t.get_element_type() == et) {
        return t;
    }
    return std::make_shared<v0::Convert>(t, et);
}

}  // namespace

OutputVector translate_openvino_gdn_attention(const NodeContext& context) {
    // Args: (mixed_qkv, b, a, conv_weight, conv_bias, A_log, dt_bias,
    //        layer_name, num_k_heads, num_v_heads, head_k_dim, head_v_dim).
    num_inputs_check(context, 12, 12);
    auto mixed_qkv = context.get_input(0);    // [T, conv_dim]
    auto b = context.get_input(1);            // [T, Hv]
    auto a = context.get_input(2);            // [T, Hv]
    auto conv_weight = context.get_input(3);  // [conv_dim, K]
    auto a_log = context.get_input(5);        // [Hv]
    auto dt_bias = context.get_input(6);      // [Hv]
    const auto layer_name = context.const_input<std::string>(7);
    const auto num_k_heads = context.const_input<int64_t>(8);
    const auto num_v_heads = context.const_input<int64_t>(9);
    const auto head_k_dim = context.const_input<int64_t>(10);
    const auto head_v_dim = context.const_input<int64_t>(11);
    PYTORCH_OP_CONVERSION_CHECK(!layer_name.empty(), "gdn_attention: layer_name must be a constant string");

    const auto data_et = mixed_qkv.get_element_type();
    // Both kernels run in f32 with f32 states, independent of the activation type the graph arrives in.
    const auto rec_et = element::f32;
    const std::string prefix = "__pa__gdn__" + layer_name + "__";
    const std::string sprefix = "__pa__gdn__shared__";

    const auto& w_ps = conv_weight.get_partial_shape();
    PYTORCH_OP_CONVERSION_CHECK(w_ps.rank().is_static() && w_ps.rank().get_length() == 2 && w_ps.is_static(),
                                "gdn_attention: conv_weight must be a static [conv_dim, kernel] tensor");
    const auto conv_dim = w_ps[0].get_length();
    const auto kernel = w_ps[1].get_length();

    auto subsequence_begins =
        get_or_make_shared_pa_param(context, sprefix + "gdn_subsequence_begins", element::i32, PartialShape{-1});
    auto block_indices =
        get_or_make_shared_pa_param(context, sprefix + "gdn_block_indices", element::i32, PartialShape{-1});
    auto block_indices_begins =
        get_or_make_shared_pa_param(context, sprefix + "gdn_block_indices_begins", element::i32, PartialShape{-1});
    auto processed_tokens =
        get_or_make_shared_pa_param(context, sprefix + "gdn_processed_tokens", element::i32, PartialShape{-1});
    auto cache_interval =
        get_or_make_shared_pa_param(context, sprefix + "gdn_cache_interval", element::i32, PartialShape{-1});

    // Causal conv1d (depthwise) + SiLU.
    auto conv_state =
        make_tagged_parameter(context, prefix + "gdn_conv_state", rec_et, PartialShape{-1, conv_dim, kernel});
    auto w3 = std::make_shared<v1::Reshape>(
        to_type(conv_weight, rec_et),
        v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{conv_dim, 1, kernel}),
        false);
    Output<Node> conv_bias;
    if (context.input_is_none(4)) {
        conv_bias = v0::Constant::create(rec_et, Shape{0}, std::vector<float>{});
    } else {
        conv_bias = to_type(context.get_input(4), rec_et);
    }
    auto conv = context.mark_node(std::make_shared<ov::op::internal::PagedCausalConv1D>(to_type(mixed_qkv, rec_et),
                                                                                        conv_state,
                                                                                        w3,
                                                                                        conv_bias,
                                                                                        subsequence_begins,
                                                                                        block_indices,
                                                                                        block_indices_begins,
                                                                                        processed_tokens,
                                                                                        cache_interval));
    auto act = std::make_shared<v4::Swish>(conv);

    // Split [q | k | v] and give each its head layout.
    const int64_t key_dim = num_k_heads * head_k_dim;
    const int64_t value_dim = num_v_heads * head_v_dim;
    auto split = std::make_shared<v1::VariadicSplit>(
        act,
        v0::Constant::create(element::i64, Shape{}, {-1}),
        v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{key_dim, key_dim, value_dim}));
    auto heads = [&](const Output<Node>& t, int64_t h, int64_t d) {
        auto shp = v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{-1, h, d});
        return to_type(std::make_shared<v1::Reshape>(t, shp, false), rec_et);
    };
    auto q = heads(split->output(0), num_k_heads, head_k_dim);
    auto k = heads(split->output(1), num_k_heads, head_k_dim);
    auto v = heads(split->output(2), num_v_heads, head_v_dim);

    // As vLLM's fused_gdn_gating; the kernel applies exp(gate) itself.
    auto softplus =
        std::make_shared<v4::SoftPlus>(std::make_shared<v1::Add>(to_type(a, rec_et), to_type(dt_bias, rec_et)));
    auto decay = std::make_shared<v0::Negative>(std::make_shared<v0::Exp>(to_type(a_log, rec_et)));
    auto gate = std::make_shared<v1::Multiply>(decay, softplus);
    auto beta = std::make_shared<v0::Sigmoid>(to_type(b, rec_et));

    auto recurrent_state = make_tagged_parameter(context,
                                                 prefix + "gdn_recurrent_state",
                                                 rec_et,
                                                 PartialShape{-1, num_v_heads, head_v_dim, head_k_dim});
    auto gdn = context.mark_node(std::make_shared<ov::op::internal::PagedGatedDeltaNet>(q,
                                                                                        k,
                                                                                        v,
                                                                                        recurrent_state,
                                                                                        gate,
                                                                                        beta,
                                                                                        subsequence_begins,
                                                                                        block_indices,
                                                                                        block_indices_begins,
                                                                                        processed_tokens,
                                                                                        cache_interval,
                                                                                        /*use_qk_l2norm=*/true,
                                                                                        1e-6F,
                                                                                        1e-6F));
    return {context.mark_node(to_type(gdn, data_et).get_node_shared_ptr())};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
