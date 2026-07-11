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

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "translate_session.hpp"
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

// Get-or-create a shared PA side-channel Parameter, scoped to the current
// TranslateSession so all PA layers reuse the same Parameter for per-sequence
// metadata (past_lens, subsequence_begins, etc.) rather than each emitting its
// own copy.
std::shared_ptr<v0::Parameter> get_or_make_shared_pa_param(const NodeContext& context,
                                                           const std::string& tag,
                                                           const element::Type& et,
                                                           const PartialShape& ps) {
    auto* session = context.get_session();
    if (session) {
        auto it = session->m_shared_pa_params.find(tag);
        if (it != session->m_shared_pa_params.end()) {
            return it->second;
        }
    }
    auto param = make_tagged_parameter(context, tag, et, ps);
    if (session) {
        session->m_shared_pa_params[tag] = param;
    }
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
    // Extract head_dim from q's pre-flattening shape for dynamic scale.
    // q arrives as rank-3 [num_tokens, num_heads, head_dim]; head_dim is last dim.
    // If q is already rank-2, we can't recover head_dim here and fall back to 0.125.
    Output<Node> scale_from_q;
    {
        const auto& q_ps = query.get_partial_shape();
        if (q_ps.rank().is_static() && q_ps.rank().get_length() >= 3 &&
            q_ps[q_ps.rank().get_length() - 1].is_static()) {
            double head_dim = static_cast<double>(q_ps[q_ps.rank().get_length() - 1].get_length());
            double scale_val = 1.0 / std::sqrt(head_dim);
            // scale must be f16 or f32 per PA validator; bf16 rejected
            auto scale_et_tmp = query.get_element_type();
            if (scale_et_tmp == element::dynamic || scale_et_tmp == element::bf16)
                scale_et_tmp = element::f32;
            scale_from_q = v0::Constant::create(scale_et_tmp, Shape{}, {static_cast<float>(scale_val)});
        }
    }

    // Capture per-layer K/V head geometry BEFORE force_rank2 flattens them.
    // K/V arrive rank-3 [num_tokens, num_kv_heads, head_dim]. This is what the
    // CPU plugin's ConvertPagedAttnInputs pass reads via rt_info to size each
    // layer's key_cache/value_cache Parameter independently — required for
    // models like Gemma-4 with heterogeneous head sizes across layers.
    auto capture_kv_geom = [](const Output<Node>& t, size_t& num_heads_out, size_t& head_size_out) {
        const auto& ps = t.get_partial_shape();
        if (ps.rank().is_static() && ps.rank().get_length() >= 3 &&
            ps[ps.rank().get_length() - 1].is_static() &&
            ps[ps.rank().get_length() - 2].is_static()) {
            head_size_out = static_cast<size_t>(ps[ps.rank().get_length() - 1].get_length());
            num_heads_out = static_cast<size_t>(ps[ps.rank().get_length() - 2].get_length());
        }
    };
    size_t k_num_heads = 0, k_head_size = 0, v_num_heads = 0, v_head_size = 0;
    capture_kv_geom(key, k_num_heads, k_head_size);
    capture_kv_geom(value, v_num_heads, v_head_size);

    // Flatten q/k/v to rank-2. Prefer a static Reshape([-1, H*D]) when the
    // trailing dims are known: that's a single Reshape op vs the dynamic
    // path's ShapeOf+Gather+Concat+Reshape chain (4 ops × 3 tensors × 16
    // layers = 192 extra ops of pure shape plumbing).
    auto force_rank2 = [&](Output<Node>& t) {
        const auto& ps = t.get_partial_shape();
        const auto r = ps.rank();
        if (r.is_static() && r.get_length() == 2) {
            return;
        }
        // Try static path: all dims after the leading num_tokens dim are
        // known, so target shape is [-1, product(rest)].
        if (r.is_static()) {
            bool trailing_static = true;
            int64_t trailing = 1;
            for (int i = 1; i < r.get_length(); ++i) {
                if (ps[i].is_static()) {
                    trailing *= ps[i].get_length();
                } else {
                    trailing_static = false;
                    break;
                }
            }
            if (trailing_static) {
                auto target = v0::Constant::create(element::i64, Shape{2},
                                                   std::vector<int64_t>{-1, trailing});
                t = std::make_shared<v1::Reshape>(t, target, false);
                return;
            }
        }
        // Fallback: build [-1, *] at runtime via ShapeOf + trailing-dims.
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

    // Q/K/V flow at their native dtype (typically bf16 from vLLM bf16 models).
    // KV cache Parameters use the same dtype as a placeholder; the CPU plugin's
    // KV_CACHE_PRECISION config and INFERENCE_PRECISION_HINT drive the actual
    // compute and cache precisions at compile time. This matches how OV GenAI
    // handles KV cache quant via runtime_options.kv_cache_precision.
    auto original_q_et = query.get_element_type();
    element::Type pa_dtype = original_q_et;

    // Side-channel Parameters bound at infer time from ForwardContext.
    // Shapes/types here mirror PagedAttentionExtension::validate_and_infer_types().
    // key_cache/value_cache must have rank 2-5 per PA validator. vLLM CPU uses
    // [num_blocks, num_kv_heads, block_size, head_size] (rank 4).
    // KV caches are per-layer (different storage per attention layer).
    // KV cache Parameter dtype: use pa_dtype as placeholder. Plugin's
    // KV_CACHE_PRECISION config overrides at compile time (matches genai).
    auto kv_et = pa_dtype;
    auto key_cache = make_tagged_parameter(context, prefix + "key_cache", kv_et,
                                           PartialShape{-1, -1, -1, -1});
    auto value_cache = make_tagged_parameter(context, prefix + "value_cache", kv_et,
                                             PartialShape{-1, -1, -1, -1});
    // Per-sequence metadata is identical across layers, so share a single
    // Parameter set across all PA ops in the model. Tagged "__pa__shared__*"
    // so the execute-time binding can recognize and populate them once.
    //
    // past_lens, subsequence_begins, and max_context_len are *derived* from
    // seq_lens + query_start_loc (vLLM's native attn_metadata format) via
    // graph ops, so Python binding only has to populate the two source
    // Parameters. block_indices / block_indices_begins still computed in
    // Python (CSR trim is awkward in graph ops with dynamic rows).
    const std::string sprefix = "__pa__shared__";
    auto seq_lens = get_or_make_shared_pa_param(context, sprefix + "seq_lens", element::i32, PartialShape{-1});
    auto query_start_loc = get_or_make_shared_pa_param(context, sprefix + "query_start_loc", element::i32, PartialShape{-1});
    // block_indices and block_indices_begins are per-layer, not shared: models
    // with multiple KV-cache groups (e.g. Gemma-4 hybrid: sliding block_size=64
    // + global block_size=32) have a distinct block_table per KV-cache group.
    // Using a single shared Parameter would silently overwrite indices from one
    // group with those of another and produce wrong results.
    auto block_indices = make_tagged_parameter(context, prefix + "block_indices", element::i32, PartialShape{-1});
    auto block_indices_begins = make_tagged_parameter(context, prefix + "block_indices_begins", element::i32, PartialShape{-1});

    auto* session = context.get_session();
    auto derive_or_cache = [&](const std::string& key,
                               std::function<Output<Node>()> mk) -> Output<Node> {
        if (session) {
            auto it = session->m_shared_pa_outputs.find(key);
            if (it != session->m_shared_pa_outputs.end()) return it->second;
        }
        auto out = mk();
        if (session) session->m_shared_pa_outputs[key] = out;
        return out;
    };

    // past_lens = seq_lens - (qsl[1:] - qsl[:-1])
    Output<Node> past_lens = derive_or_cache("past_lens", [&]() -> Output<Node> {
        auto one = v0::Constant::create(element::i32, Shape{1}, {1});
        auto zero = v0::Constant::create(element::i32, Shape{1}, {0});
        auto neg_one = v0::Constant::create(element::i32, Shape{1}, {-1});
        auto big = v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()});
        auto axis0 = v0::Constant::create(element::i32, Shape{1}, {0});
        auto qsl_tail = std::make_shared<v8::Slice>(query_start_loc, one, big, one, axis0);
        auto qsl_head = std::make_shared<v8::Slice>(query_start_loc, zero, neg_one, one, axis0);
        auto q_lens = std::make_shared<v1::Subtract>(qsl_tail, qsl_head);
        return std::make_shared<v1::Subtract>(seq_lens, q_lens);
    });

    // subsequence_begins = query_start_loc (same semantics).
    Output<Node> subsequence_begins = query_start_loc;

    // max_context_len = ReduceMax(seq_lens) along axis 0, kept scalar.
    Output<Node> max_context_len = derive_or_cache("max_context_len", [&]() -> Output<Node> {
        auto axis0 = v0::Constant::create(element::i32, Shape{1}, {0});
        // keep_dims=false -> scalar output, which matches PA's expected shape.
        return std::make_shared<v1::ReduceMax>(seq_lens, axis0, false);
    });

    // Default scalar/empty constants for unused PA inputs.
    // Element type for real-valued PA inputs. PA validator only accepts
    // f16/f32 for rotation_trig_lut/xattention_threshold; bf16 is rejected.
    // Use f32 for all these LUTs regardless of PA compute dtype.
    auto scale_et = (pa_dtype == element::bf16) ? element::f32 : pa_dtype;
    // scale is attention 1/sqrt(head_dim); extracted from q's pre-flatten shape
    // above. Falls back to 0.125 (head_dim=64) if q's rank/last-dim was dynamic.
    Output<Node> scale = scale_from_q.get_node_shared_ptr()
        ? scale_from_q
        : v0::Constant::create(scale_et, Shape{}, {0.125f});

    // sliding_window is per-layer: hybrid models like Gemma-4 mix
    // sliding-attention layers (window=512) with full-attention layers
    // (window=0). Emit as a side-channel Parameter so bind time can pass in
    // each layer's actual window value from vLLM's layer_obj.impl.sliding_window.
    auto sliding_window = make_tagged_parameter(context, prefix + "sliding_window",
                                                element::i32, PartialShape{});
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
    // Attach per-layer KV head geometry as rt_info so the CPU plugin's
    // ConvertPagedAttnInputs pass can size each layer's key_cache / value_cache
    // Parameter to its actual head dims (e.g. Gemma-4 has some layers with
    // head_size=256 and others with head_size=512). Without this, all layers
    // fall back to the plugin's default (uniform) shape and PA fails at
    // runtime with dim mismatches on the heterogeneous-head layers.
    if (k_num_heads && k_head_size && v_num_heads && v_head_size) {
        pa->get_rt_info()["num_k_heads"] = k_num_heads;
        pa->get_rt_info()["k_head_size"] = k_head_size;
        pa->get_rt_info()["num_v_heads"] = v_num_heads;
        pa->get_rt_info()["v_head_size"] = v_head_size;
    }
    if (std::getenv("OV_DBG_PA_TRANS")) {
        std::cerr << "[PA_TRANS] emitted PagedAttentionExtension for layer " << layer_name
                  << ", output ps=" << pa->output(0).get_partial_shape()
                  << ", k=(" << k_num_heads << "," << k_head_size << ")"
                  << ", v=(" << v_num_heads << "," << v_head_size << ")" << std::endl;
    }
    // PagedAttentionExtension has multiple outputs; the FX op returns just the
    // attention output (output 0). Convert back to query's original dtype
    // (typically f16) so downstream MatMul weight dtypes match.
    Output<Node> pa_out = pa->output(0);
    if (original_q_et != element::f32 && !original_q_et.is_dynamic()) {
        pa_out = std::make_shared<ov::op::v0::Convert>(pa_out, original_q_et);
    }
    return {pa_out};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
