// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Translator for torch.ops.openvino.paged_attention.default -> PagedAttentionExtension.
//
// vLLM's attention comes into the FX graph as:
//   auto_functionalized_v2(vllm.unified_attention_with_output, query, key, value,
//                          output, layer_name=..., ...)
//
// A Python FX pre-pass rewrites this to a direct call of
//   torch.ops.openvino.paged_attention(query, key, value, layer_name)
// which reaches this translator. The KV cache, block tables, past_lens etc.
// are not in the FX graph - they live in vLLM's ForwardContext sidechannel.
// We create extra v0::Parameter nodes for those side-channel tensors, tag
// them with friendly names like "__pa__<layer_name>__<field>", and emit a
// PagedAttentionExtension. The execute-time backend inspects the compiled
// model for these tagged parameters and binds them from ForwardContext.

#include "openvino/op/paged_attention.hpp"

#include <cstdlib>
#include <iostream>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {

std::shared_ptr<v0::Parameter> make_tagged_parameter(const NodeContext& context,
                                                     const std::string& tag,
                                                     const element::Type& et,
                                                     const PartialShape& ps) {
    auto param = std::make_shared<v0::Parameter>(et, ps);
    param->set_friendly_name(tag);
    param->output(0).set_names({tag});
    // Register so the final Model::check_all_parameters_registered passes.
    context.add_external_parameter(param);
    return param;
}

}  // namespace

OutputVector translate_openvino_paged_attention(const NodeContext& context) {
    // Args: (query, key, value, layer_name)
    num_inputs_check(context, 4, 4);

    auto query = context.get_input(0);
    auto key = context.get_input(1);
    auto value = context.get_input(2);

    // Extract layer_name string attribute. The FX decoder may expose it as a
    // Constant string, a Python str value, or elsewhere.
    std::string layer_name;
    try {
        layer_name = context.const_input<std::string>(3);
    } catch (const std::exception&) {
        // Try the input as a Constant of u8 bytes, or a string tensor
        try {
            auto vals = context.get_values_from_const_input(3);
            if (vals.is<std::string>()) {
                layer_name = vals.as<std::string>();
            }
        } catch (const std::exception&) {
            // Fall through
        }
    }
    if (layer_name.empty()) {
        layer_name = "unknown_layer";
    }
    if (std::getenv("OV_DBG_PA_TRANS")) {
        std::cerr << "[PA_TRANS_IN] layer='" << layer_name
                  << "' q_ps=" << query.get_partial_shape()
                  << " k_ps=" << key.get_partial_shape()
                  << " v_ps=" << value.get_partial_shape() << std::endl;
    }

    const std::string prefix = "__pa__" + layer_name + "__";

    // PagedAttentionExtension requires q/k/v to be rank 2 [num_tokens,
    // num_heads*head_dim]. vLLM passes them as rank-2 already in the FX graph
    // for CPU (output shape from unified_attention is [num_tokens, hidden]).
    // Emit a Reshape(-1, H) to guarantee rank-2 even if upstream is dynamic.
    // For Llama-3.2-1B the hidden dims are:
    //   q hidden = 32 * 64 = 2048
    //   k hidden = 8 * 64 = 512
    //   v hidden = 8 * 64 = 512
    // We don't know them statically here - trust that upstream reshape has
    // already produced rank-2 tensors. If not rank 2, wrap in Reshape(-1, C)
    // with C taken from the last known static dim.
    // Flatten q/k/v to 2D [N, H*D] by collapsing all trailing dims. The
    // leading dim (num_tokens) is kept dynamic; we build target shape [N, -1]
    // at runtime via ShapeOf + Gather(0) + Concat([dim0, -1]).
    auto force_rank2 = [&](Output<Node>& t) {
        const auto& ps = t.get_partial_shape();
        if (ps.rank().is_static() && ps.rank().get_length() == 2) {
            return;
        }
        auto shp = std::make_shared<v3::ShapeOf>(t, element::i64);
        auto zero_i = v0::Constant::create(element::i64, Shape{1}, {0});
        auto axis0 = v0::Constant::create(element::i64, Shape{}, {0});
        auto dim0 = std::make_shared<v8::Gather>(shp, zero_i, axis0);
        auto neg1 = v0::Constant::create(element::i64, Shape{1}, {-1});
        auto target = std::make_shared<v0::Concat>(OutputVector{dim0, neg1}, 0);
        t = std::make_shared<v1::Reshape>(t, target, false);
    };
    force_rank2(query);
    force_rank2(key);
    force_rank2(value);

    // Side-channel Parameters bound at infer time from ForwardContext.
    // Shapes/types here mirror PagedAttentionExtension::validate_and_infer_types().
    // key_cache/value_cache must have rank 2-5 per PA validator. vLLM CPU uses
    // [num_blocks, num_kv_heads, block_size, head_size] (rank 4).
    auto key_cache = make_tagged_parameter(context, prefix + "key_cache", query.get_element_type(),
                                           PartialShape{-1, -1, -1, -1});
    auto value_cache = make_tagged_parameter(context, prefix + "value_cache", query.get_element_type(),
                                             PartialShape{-1, -1, -1, -1});
    auto past_lens = make_tagged_parameter(context, prefix + "past_lens", element::i32, PartialShape{-1});
    auto subsequence_begins = make_tagged_parameter(context, prefix + "subsequence_begins", element::i32, PartialShape{-1});
    auto block_indices = make_tagged_parameter(context, prefix + "block_indices", element::i32, PartialShape{-1});
    auto block_indices_begins = make_tagged_parameter(context, prefix + "block_indices_begins", element::i32, PartialShape{-1});
    auto max_context_len = make_tagged_parameter(context, prefix + "max_context_len", element::i32, PartialShape{});

    // Default scalar/empty constants for unused PA inputs.
    // Element type of scale follows q; for fp32 graphs this is f32.
    auto scale_et = query.get_element_type();
    if (scale_et == element::dynamic) scale_et = element::f32;
    // scale is attention 1/sqrt(head_dim); 1/8 = 0.125 for head_dim=64
    auto scale = v0::Constant::create(scale_et, Shape{}, {0.125f});

    auto sliding_window = v0::Constant::create(element::i32, Shape{}, {0});
    auto alibi_slopes = v0::Constant::create(scale_et, Shape{0}, std::vector<float>{});
    auto score_aggr_window = v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto rotated_block_indices = v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto rotation_deltas = v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto rotation_trig_lut = v0::Constant::create(scale_et, Shape{0}, std::vector<float>{});
    auto xattention_threshold = v0::Constant::create(scale_et, Shape{0}, std::vector<float>{});
    auto xattention_block_size = v0::Constant::create(element::i32, Shape{}, {0});
    auto xattention_stride = v0::Constant::create(element::i32, Shape{}, {0});
    auto sinks = v0::Constant::create(scale_et, Shape{0}, std::vector<float>{});
    auto adaptive_rkv_start_size = v0::Constant::create(element::i32, Shape{}, {0});
    auto adaptive_rkv_evictable_sizes = v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto adaptive_rkv_diversity_block_set_indices = v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto adaptive_rkv_diversity_block_set_indices_begins = v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto token_type_ids = v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto qq_bias = v0::Constant::create(element::u8, Shape{0}, std::vector<uint8_t>{});
    auto qq_bias_begins = v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});

    OutputVector pa_inputs = {
        query,                                                  // 0
        key,                                                    // 1
        value,                                                  // 2
        key_cache,                                              // 3
        value_cache,                                            // 4
        past_lens,                                              // 5
        subsequence_begins,                                     // 6
        block_indices,                                          // 7
        block_indices_begins,                                   // 8
        scale,                                                  // 9
        sliding_window,                                         // 10
        alibi_slopes,                                           // 11
        max_context_len,                                        // 12
        score_aggr_window,                                      // 13
        rotated_block_indices,                                  // 14
        rotation_deltas,                                        // 15
        rotation_trig_lut,                                      // 16
        xattention_threshold,                                   // 17
        xattention_block_size,                                  // 18
        xattention_stride,                                      // 19
        sinks,                                                  // 20
        adaptive_rkv_start_size,                                // 21
        adaptive_rkv_evictable_sizes,                           // 22
        adaptive_rkv_diversity_block_set_indices,               // 23
        adaptive_rkv_diversity_block_set_indices_begins,        // 24
        token_type_ids,                                         // 25
        qq_bias,                                                // 26
        qq_bias_begins,                                         // 27
    };

    auto pa = context.mark_node(std::make_shared<PagedAttentionExtension>(pa_inputs));
    if (std::getenv("OV_DBG_PA_TRANS")) {
        std::cerr << "[PA_TRANS] emitted PagedAttentionExtension for layer " << layer_name
                  << ", output ps=" << pa->output(0).get_partial_shape() << std::endl;
    }
    // PagedAttentionExtension has multiple outputs; the FX op returns just the
    // attention output (output 0).
    return {pa->output(0)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
