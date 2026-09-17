// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Translator for torch.ops.openvino.paged_attention.default. KV cache/block
// tables aren't in the FX graph; creates side-channel Parameters instead.

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

// Get-or-create a shared PA side-channel Parameter, scoped to the session
// so all PA layers reuse one Parameter for per-sequence metadata.
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

    // PagedAttentionExtension requires rank-2 q/k/v; flatten any higher-rank
    // input, and read head_dim off q pre-flattening for the scale.
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

    // Capture per-layer K/V head geometry before flattening -- the CPU
    // plugin reads it via rt_info to size each layer's cache independently.
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

    // Flatten q/k/v to rank-2. Prefer a static Reshape([-1, H*D]) when
    // trailing dims are known -- cheaper than the dynamic ShapeOf chain.
    auto force_rank2 = [&](Output<Node>& t, bool safe_to_fuse_upstream = false) {
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
                // If this input is a single-consumer Reshape, retarget it
                // to [-1, trailing] instead of adding a second Reshape.
                static const bool _pa_fuse_upstream =
                    std::getenv("OV_PA_FUSE_UPSTREAM_RESHAPE") == nullptr ||
                    std::string(std::getenv("OV_PA_FUSE_UPSTREAM_RESHAPE")) != "0";
                if (_pa_fuse_upstream && safe_to_fuse_upstream) {
                    auto up = std::dynamic_pointer_cast<v1::Reshape>(t.get_node_shared_ptr());
                    if (up && up->get_output_target_inputs(0).size() <= 1) {
                        auto target = v0::Constant::create(element::i64, Shape{2},
                                                            std::vector<int64_t>{-1, trailing});
                        up->input(1).replace_source_output(target->output(0));
                        up->validate_and_infer_types();
                        // t already points at up's output; refresh partial shape
                        t = up->output(0);
                        return;
                    }
                }
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
    // Only Q's upstream Reshape can be fused: K/V's has a second consumer
    // that needs the original rank-3 shape.
    force_rank2(query, /*safe_to_fuse_upstream=*/true);
    force_rank2(key);
    force_rank2(value);

    // Q/K/V flow at their native dtype; KV cache Parameters use it as a
    // placeholder, with the plugin's config driving actual precision.
    auto original_q_et = query.get_element_type();
    element::Type pa_dtype = original_q_et;

    // Side-channel Parameters bound at infer time from ForwardContext, one
    // key_cache/value_cache pair per layer (rank 2-5 per the PA validator).
    auto kv_et = pa_dtype;
    auto key_cache = make_tagged_parameter(context, prefix + "key_cache", kv_et,
                                           PartialShape{-1, -1, -1, -1});
    auto value_cache = make_tagged_parameter(context, prefix + "value_cache", kv_et,
                                             PartialShape{-1, -1, -1, -1});
    // Per-sequence metadata is identical across layers, so share one
    // Parameter set, tagged "__pa__shared__*", across all PA ops.
    const std::string sprefix = "__pa__shared__";
    auto seq_lens = get_or_make_shared_pa_param(context, sprefix + "seq_lens", element::i32, PartialShape{-1});
    auto query_start_loc = get_or_make_shared_pa_param(context, sprefix + "query_start_loc", element::i32, PartialShape{-1});
    // block_indices/block_indices_begins are per-layer, not shared: models
    // with multiple KV-cache groups have a distinct block_table per group.
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

    // Default scalar/empty constants for unused PA inputs. f32 for LUTs
    // regardless of pa_dtype: the PA validator rejects bf16 here.
    auto scale_et = (pa_dtype == element::bf16) ? element::f32 : pa_dtype;
    // scale is attention 1/sqrt(head_dim); extracted from q's pre-flatten shape
    // above. Falls back to 0.125 (head_dim=64) if q's rank/last-dim was dynamic.
    Output<Node> scale = scale_from_q.get_node_shared_ptr()
        ? scale_from_q
        : v0::Constant::create(scale_et, Shape{}, {0.125f});

    // sliding_window is per-layer (hybrid models mix sliding/full attention);
    // emit as a side-channel Parameter bound to each layer's real value.
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
    // Attach per-layer KV head geometry as rt_info so ConvertPagedAttnInputs
    // can size each layer's cache Parameter to its actual head dims.
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
    // The FX op returns only output 0; convert back to query's original
    // dtype so downstream MatMul weight dtypes match.
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
