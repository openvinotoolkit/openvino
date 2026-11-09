// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Translator for torch.ops.ov_ext.paged_attention.default; KV cache and block tables come in as side-channel
// Parameters.

#include "openvino/op/paged_attention.hpp"

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

namespace ov::frontend::pytorch::op::ext {

using namespace ov::op;

namespace {
// A string argument as the FX decoder exposes it: a string Constant or a Python str value. Empty if neither.
std::string optional_const_string(const NodeContext& context, size_t idx) {
    try {
        return context.const_input<std::string>(idx);
    } catch (const std::exception&) {
    }
    try {
        auto vals = context.get_values_from_const_input(idx);
        if (vals.is<std::string>()) {
            return vals.as<std::string>();
        }
    } catch (const std::exception&) {
    }
    return {};
}
}  // namespace

OutputVector translate_paged_attention(const NodeContext& context) {
    // Args: (query, key, value, layer_name[, scale, kv_sharing_target]); the last two are optional.
    num_inputs_check(context, 4, 6);

    auto query = context.get_input(0);
    auto key = context.get_input(1);
    auto value = context.get_input(2);

    std::string layer_name = optional_const_string(context, 3);
    if (layer_name.empty()) {
        // A distinct tag per occurrence: a shared fallback tag would alias one Parameter across layers whose
        // KV-cache shapes differ.
        auto* session = context.get_session();
        size_t idx = session ? session->m_unknown_pa_layer_count++ : 0;
        layer_name = idx == 0 ? "unknown_layer" : "unknown_layer_" + std::to_string(idx);
    }
    // vLLM's layer.impl.scale; 0 if the op carries none, then 1/sqrt(head_dim) below.
    double scale_arg = 0.0;
    if (context.get_input_size() >= 5) {
        try {
            scale_arg = context.const_input<double>(4);
        } catch (const std::exception&) {
            try {
                scale_arg = static_cast<double>(context.const_input<float>(4));
            } catch (const std::exception&) {
                scale_arg = 0.0;
            }
        }
    }

    std::string kv_sharing_target = context.get_input_size() >= 6 ? optional_const_string(context, 5) : std::string{};

    // vLLM hands a KV-sharing layer raw k/v and skips its cache write, but PagedAttentionExtension always
    // writes. Using the target's normed k/v makes that write idempotent. The query stays the layer's own.
    auto* kv_session = context.get_session();
    if (!kv_sharing_target.empty() && kv_session) {
        auto it = kv_session->m_pa_layer_kv.find(kv_sharing_target);
        if (it != kv_session->m_pa_layer_kv.end()) {
            key = it->second.first;
            value = it->second.second;
        } else {
            // Target not translated yet (vLLM always points at an earlier layer): keep the raw tensors rather
            // than fail.
            std::cerr << "[PA_TRANS] warning: layer '" << layer_name << "' shares KV with '" << kv_sharing_target
                      << "', which has not been translated yet; using its own key/value" << std::endl;
        }
    } else if (kv_session) {
        kv_session->m_pa_layer_kv[layer_name] = {key, value};
    }

    if (std::getenv("OV_DBG_PA_TRANS")) {
        std::cerr << "[PA_TRANS_IN] layer='" << layer_name << "' q_ps=" << query.get_partial_shape()
                  << " k_ps=" << key.get_partial_shape() << " v_ps=" << value.get_partial_shape()
                  << " scale_arg=" << scale_arg << " kv_share='" << kv_sharing_target << "'" << std::endl;
    }

    const std::string prefix = "__pa__" + layer_name + "__";

    // PagedAttentionExtension requires rank-2 q/k/v; read head_dim off q before flattening.
    Output<Node> scale_from_q;
    {
        // The PA validator accepts an f16 or f32 scale only.
        auto scale_et_tmp = query.get_element_type();
        if (scale_et_tmp == element::dynamic || scale_et_tmp == element::bf16)
            scale_et_tmp = element::f32;

        bool have_scale = false;
        double scale_val = 0.0;
        if (scale_arg > 0.0) {
            // Authoritative: not always 1/sqrt(head_dim) (Gemma-4 uses 1.0 and folds the normalization into
            // its Q/K norms).
            scale_val = scale_arg;
            have_scale = true;
        } else {
            const auto& q_ps = query.get_partial_shape();
            if (q_ps.rank().is_static() && q_ps.rank().get_length() >= 3 &&
                q_ps[q_ps.rank().get_length() - 1].is_static()) {
                double head_dim = static_cast<double>(q_ps[q_ps.rank().get_length() - 1].get_length());
                scale_val = 1.0 / std::sqrt(head_dim);
                have_scale = true;
            }
        }
        if (have_scale) {
            scale_from_q = v0::Constant::create(scale_et_tmp, Shape{}, {static_cast<float>(scale_val)});
        }
    }

    // The CPU plugin reads the per-layer K/V head geometry from rt_info to size each cache.
    auto capture_kv_geom = [](const Output<Node>& t, size_t& num_heads_out, size_t& head_size_out) {
        const auto& ps = t.get_partial_shape();
        if (ps.rank().is_static() && ps.rank().get_length() >= 3 && ps[ps.rank().get_length() - 1].is_static() &&
            ps[ps.rank().get_length() - 2].is_static()) {
            head_size_out = static_cast<size_t>(ps[ps.rank().get_length() - 1].get_length());
            num_heads_out = static_cast<size_t>(ps[ps.rank().get_length() - 2].get_length());
        }
    };
    size_t k_num_heads = 0, k_head_size = 0, v_num_heads = 0, v_head_size = 0;
    capture_kv_geom(key, k_num_heads, k_head_size);
    capture_kv_geom(value, v_num_heads, v_head_size);

    // Prefer a static Reshape([-1, H*D]) over a ShapeOf chain when the trailing dims are known.
    auto force_rank2 = [&](Output<Node>& t, bool safe_to_fuse_upstream = false) {
        const auto& ps = t.get_partial_shape();
        const auto r = ps.rank();
        if (r.is_static() && r.get_length() == 2) {
            return;
        }
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
                // Retarget a single-consumer Reshape instead of adding a second one.
                static const bool _pa_fuse_upstream = std::getenv("OV_PA_FUSE_UPSTREAM_RESHAPE") == nullptr ||
                                                      std::string(std::getenv("OV_PA_FUSE_UPSTREAM_RESHAPE")) != "0";
                if (_pa_fuse_upstream && safe_to_fuse_upstream) {
                    auto up = std::dynamic_pointer_cast<v1::Reshape>(t.get_node_shared_ptr());
                    if (up && up->get_output_target_inputs(0).size() <= 1) {
                        auto target = v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, trailing});
                        up->input(1).replace_source_output(target->output(0));
                        up->validate_and_infer_types();
                        t = up->output(0);
                        return;
                    }
                }
                auto target = v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, trailing});
                t = std::make_shared<v1::Reshape>(t, target, false);
                return;
            }
        }
        auto shp = std::make_shared<v3::ShapeOf>(t, element::i64);
        auto zero_i = v0::Constant::create(element::i64, Shape{1}, {0});
        auto axis0 = v0::Constant::create(element::i64, Shape{}, {0});
        auto dim0 = std::make_shared<v8::Gather>(shp, zero_i, axis0);
        auto neg1 = v0::Constant::create(element::i64, Shape{1}, {-1});
        auto target = std::make_shared<v0::Concat>(OutputVector{dim0, neg1}, 0);
        t = std::make_shared<v1::Reshape>(t, target, false);
    };
    // Only Q's upstream Reshape can be fused: K/V's has a consumer that needs the rank-3 shape.
    force_rank2(query, /*safe_to_fuse_upstream=*/true);
    force_rank2(key);
    force_rank2(value);

    // The KV cache Parameters take Q/K/V's dtype as a placeholder; the plugin config sets the precision.
    auto original_q_et = query.get_element_type();
    element::Type pa_dtype = original_q_et;

    // Side-channel Parameters bound at infer time: one key_cache/value_cache pair per layer.
    auto kv_et = pa_dtype;
    auto key_cache = make_tagged_parameter(context, prefix + "key_cache", kv_et, PartialShape{-1, -1, -1, -1});
    auto value_cache = make_tagged_parameter(context, prefix + "value_cache", kv_et, PartialShape{-1, -1, -1, -1});
    // Per-sequence metadata is shared by all PA ops ("__pa__shared__*").
    const std::string sprefix = "__pa__shared__";
    auto seq_lens = get_or_make_shared_pa_param(context, sprefix + "seq_lens", element::i32, PartialShape{-1});
    auto query_start_loc =
        get_or_make_shared_pa_param(context, sprefix + "query_start_loc", element::i32, PartialShape{-1});
    // Per layer: models with several KV-cache groups have a block_table per group.
    auto block_indices = make_tagged_parameter(context, prefix + "block_indices", element::i32, PartialShape{-1});
    auto block_indices_begins =
        make_tagged_parameter(context, prefix + "block_indices_begins", element::i32, PartialShape{-1});

    auto* session = context.get_session();
    auto derive_or_cache = [&](const std::string& key, std::function<Output<Node>()> mk) -> Output<Node> {
        if (session) {
            auto it = session->m_shared_pa_outputs.find(key);
            if (it != session->m_shared_pa_outputs.end())
                return it->second;
        }
        auto out = mk();
        if (session)
            session->m_shared_pa_outputs[key] = out;
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

    Output<Node> subsequence_begins = query_start_loc;

    Output<Node> max_context_len = derive_or_cache("max_context_len", [&]() -> Output<Node> {
        auto axis0 = v0::Constant::create(element::i32, Shape{1}, {0});
        // keep_dims=false: PA expects a scalar.
        return std::make_shared<v1::ReduceMax>(seq_lens, axis0, false);
    });

    // f32 regardless of pa_dtype: the PA validator rejects bf16 here.
    auto scale_et = (pa_dtype == element::bf16) ? element::f32 : pa_dtype;
    // 0.125 (head_dim 64) only when the scale is neither given nor derivable.
    Output<Node> scale =
        scale_from_q.get_node_shared_ptr() ? scale_from_q : v0::Constant::create(scale_et, Shape{}, {0.125f});

    // Per layer: hybrid models mix sliding and full attention.
    auto sliding_window = make_tagged_parameter(context, prefix + "sliding_window", element::i32, PartialShape{});
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
    auto adaptive_rkv_diversity_block_set_indices =
        v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto adaptive_rkv_diversity_block_set_indices_begins =
        v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto token_type_ids = v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto qq_bias = v0::Constant::create(element::u8, Shape{0}, std::vector<uint8_t>{});
    auto qq_bias_begins = v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});

    OutputVector pa_inputs = {
        query,                                            // 0
        key,                                              // 1
        value,                                            // 2
        key_cache,                                        // 3
        value_cache,                                      // 4
        past_lens,                                        // 5
        subsequence_begins,                               // 6
        block_indices,                                    // 7
        block_indices_begins,                             // 8
        scale,                                            // 9
        sliding_window,                                   // 10
        alibi_slopes,                                     // 11
        max_context_len,                                  // 12
        score_aggr_window,                                // 13
        rotated_block_indices,                            // 14
        rotation_deltas,                                  // 15
        rotation_trig_lut,                                // 16
        xattention_threshold,                             // 17
        xattention_block_size,                            // 18
        xattention_stride,                                // 19
        sinks,                                            // 20
        adaptive_rkv_start_size,                          // 21
        adaptive_rkv_evictable_sizes,                     // 22
        adaptive_rkv_diversity_block_set_indices,         // 23
        adaptive_rkv_diversity_block_set_indices_begins,  // 24
        token_type_ids,                                   // 25
        qq_bias,                                          // 26
        qq_bias_begins,                                   // 27
    };

    auto pa = context.mark_node(std::make_shared<PagedAttentionExtension>(pa_inputs));
    if (k_num_heads && k_head_size && v_num_heads && v_head_size) {
        pa->get_rt_info()["num_k_heads"] = k_num_heads;
        pa->get_rt_info()["k_head_size"] = k_head_size;
        pa->get_rt_info()["num_v_heads"] = v_num_heads;
        pa->get_rt_info()["v_head_size"] = v_head_size;
    }
    if (std::getenv("OV_DBG_PA_TRANS")) {
        std::cerr << "[PA_TRANS] emitted PagedAttentionExtension for layer " << layer_name
                  << ", output ps=" << pa->output(0).get_partial_shape() << ", k=(" << k_num_heads << "," << k_head_size
                  << ")"
                  << ", v=(" << v_num_heads << "," << v_head_size << ")" << std::endl;
    }
    // The FX op returns only output 0; convert it back to query's dtype.
    Output<Node> pa_out = pa->output(0);
    if (original_q_et != element::f32 && !original_q_et.is_dynamic()) {
        pa_out = std::make_shared<ov::op::v0::Convert>(pa_out, original_q_et);
    }
    return {pa_out};
}

}  // namespace ov::frontend::pytorch::op::ext
