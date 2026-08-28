// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "patch_sliding_window_kv_layout.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <utility>

#include "../llm_compiled_model_utils.hpp"
#include "../logging.hpp"
#include "../util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace {

constexpr size_t kMaskInputIdx = 3;
constexpr size_t kTargetShapeInputIdx = 1;  // Reshape/Broadcast target-shape input.

// For each sliding past_kv Parameter, if its output has multiple users then all
// additional users (besides the single Concat consumer) must be ShapeOf nodes.
// Such ShapeOf outputs are frozen to constants so subsequent Step 1 Parameter reshapes
// do not leak into shared dynamic shape computations.
//
// If there is only one user, no action is needed.
// If a multi-user topology contains any non-ShapeOf/non-Concat consumer (or multiple
// Concat consumers), this function fails fast via OPENVINO_ASSERT.
void freeze_shared_shapeof_from_sliding_past_kv(const std::shared_ptr<ov::Model>& model,
                                                const std::vector<bool>& layer_is_sliding) {
    size_t num_shapeof_frozen = 0;
    std::unordered_set<const ov::Node*> global_frozen_shapeof;

    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        const auto maybe_layer_idx = ov::npuw::util::isPastKeyValuesContiguous(name);
        if (!maybe_layer_idx) {
            continue;
        }
        const size_t layer_idx = static_cast<size_t>(*maybe_layer_idx);
        if (layer_idx >= layer_is_sliding.size() || !layer_is_sliding[layer_idx]) {
            continue;
        }

        auto param = ov::as_type_ptr<ov::op::v0::Parameter>(input.get_node_shared_ptr());
        OPENVINO_ASSERT(param,
                        "[SWA] Sliding past_kv input '",
                        name,
                        "' must be a Parameter node.");

        std::vector<std::shared_ptr<ov::Node>> consumers;
        std::unordered_set<const ov::Node*> seen;
        for (const auto& target_input : param->output(0).get_target_inputs()) {
            auto consumer = target_input.get_node()->shared_from_this();
            if (seen.insert(consumer.get()).second) {
                consumers.push_back(consumer);
            }
        }

        if (consumers.size() <= 1) {
            continue;
        }

        bool seen_concat = false;
        for (const auto& consumer : consumers) {
            if (ov::is_type<ov::op::v0::Concat>(consumer)) {
                OPENVINO_ASSERT(!seen_concat,
                                "[SWA] Sliding past_kv '",
                                name,
                                "' has multiple Concat consumers; unsupported topology.");
                seen_concat = true;
                continue;
            }

            auto shapeof = ov::as_type_ptr<ov::op::v3::ShapeOf>(consumer);
            OPENVINO_ASSERT(shapeof,
                            "[SWA] Sliding past_kv '",
                            name,
                            "' has unsupported extra consumer '",
                            consumer->get_type_name(),
                            "' (only ShapeOf is allowed when fan-out > 1).");

            if (!global_frozen_shapeof.insert(shapeof.get()).second) {
                continue;
            }

            auto folded = ov::util::get_constant_from_source(shapeof->output(0));
            OPENVINO_ASSERT(folded,
                            "[SWA] Failed to fold ShapeOf output for '",
                            shapeof->get_friendly_name(),
                            "'.");
            auto vals = folded->cast_vector<int64_t>();
            auto frozen = std::make_shared<ov::op::v0::Constant>(
                shapeof->output(0).get_element_type(), ov::Shape{vals.size()}, vals);
            frozen->set_friendly_name(shapeof->get_friendly_name() + "/swa_shapeof_frozen");

            const auto shapeof_users = shapeof->output(0).get_target_inputs();
            for (const auto& user_input : shapeof_users) {
                user_input.replace_source_output(frozen);
            }
            ++num_shapeof_frozen;
            LOG_INFO("[SWA] Froze shared ShapeOf '" << shapeof->get_friendly_name()
                      << "' from sliding past_kv '" << name << "' (layer=" << layer_idx
                      << ")");
        }

        OPENVINO_ASSERT(seen_concat,
                        "[SWA] Sliding past_kv '",
                        name,
                        "' has fan-out > 1 but no Concat consumer; unsupported topology.");
    }

    if (num_shapeof_frozen > 0) {
        LOG_INFO("[SWA] freeze_shared_shapeof_from_sliding_past_kv: froze "
                  << num_shapeof_frozen << " shared ShapeOf node(s).");
    }
}



// Walk backward from a start node via port-0 chains to find the nearest Parameter.
// Returns nullptr if the walk exceeds the depth limit.
static std::shared_ptr<ov::op::v0::Parameter> find_upstream_parameter(const std::shared_ptr<ov::Node>& start,
                                                                       int max_depth = 8) {
    auto node = start;
    for (int depth = 0; node && depth < max_depth; ++depth) {
        if (auto param = ov::as_type_ptr<ov::op::v0::Parameter>(node)) {
            return param;
        }
        if (node->get_input_size() == 0) {
            break;
        }
        node = node->input_value(0).get_node_shared_ptr();
    }
    return nullptr;
}

// Classify an SDPA as sliding or non-sliding by walking the K (port 1) data path
// backward to find the first Concat boundary, then looking up the upstream past_kv
// Parameter name in layer_is_sliding[].
// Returns {is_sliding, layer_idx}; layer_idx is -1 if the layer cannot be identified.
// Uses layer_is_sliding[] as the sole authoritative source — does NOT use rt_info.
static std::pair<bool, int> classify_sdpa_layer(
    const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& sdpa,
    const std::vector<bool>& layer_is_sliding) {
    if (sdpa->get_input_size() < 2) {
        return {false, -1};
    }
    auto node = sdpa->input_value(1).get_node_shared_ptr();
    for (int depth = 0; depth < 16 && node; ++depth) {
        if (auto concat = ov::as_type_ptr<ov::op::v0::Concat>(node)) {
            auto past_param = find_upstream_parameter(concat->input_value(0).get_node_shared_ptr());
            if (past_param) {
                const auto maybe_idx =
                    ov::npuw::util::isPastKeyValuesContiguous(past_param->get_friendly_name());
                if (maybe_idx) {
                    const size_t idx = static_cast<size_t>(*maybe_idx);
                    const bool is_sl = idx < layer_is_sliding.size() && layer_is_sliding[idx];
                    return {is_sl, static_cast<int>(idx)};
                }
            }
            return {false, -1};
        }
        if (ov::is_type<ov::op::v0::Parameter>(node) || node->get_input_size() == 0) {
            break;
        }
        node = node->input_value(0).get_node_shared_ptr();
    }
    return {false, -1};
}

// When walking backward from `cur` to its port-0 producer, compute the kv_seq_axis
// in the producer's output. Searches the port-0 data input's partial shape for the
// unique axis that carries kvcache_size; if exactly one is found it is returned,
// otherwise kv_seq_axis is returned unchanged as a conservative fallback.
static size_t update_kv_axis_backward(const std::shared_ptr<ov::Node>& cur,
                                      size_t kv_seq_axis,
                                      int64_t kvcache_size) {
    if (cur->get_input_size() == 0) {
        return kv_seq_axis;
    }
    const auto& pshape = cur->input(0).get_partial_shape();
    if (!pshape.rank().is_static()) {
        return kv_seq_axis;
    }
    const size_t rank = pshape.rank().get_length();
    int found = -1;
    for (size_t i = 0; i < rank; ++i) {
        if (pshape[i].is_static() && pshape[i].get_length() == kvcache_size) {
            if (found >= 0) {
                return kv_seq_axis;  // multiple matches — fall back to unchanged axis
            }
            found = static_cast<int>(i);
        }
    }
    return found >= 0 ? static_cast<size_t>(found) : kv_seq_axis;
}

// Replaces the mask input of every sliding-layer SDPA with a single shared Parameter
// ('sliding_window_attention_mask'). The runtime supplies a window-aligned mask buffer
// at each decode step; this pass wires all sliding SDPAs to the same input port so only
// one external tensor needs to be provided.
void externalize_sliding_sdpa_masks(const std::shared_ptr<ov::Model>& model,
                                    const std::vector<bool>& layer_is_sliding) {
    std::shared_ptr<ov::op::v0::Parameter> mask_param;
    size_t num_externalized = 0;

    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
        if (!sdpa || !classify_sdpa_layer(sdpa, layer_is_sliding).first) {
            continue;
        }
        OPENVINO_ASSERT(sdpa->get_input_size() > kMaskInputIdx,
                        "[SWA] Sliding SDPA '",
                        sdpa->get_friendly_name(),
                        "' must have at least ",
                        kMaskInputIdx + 1,
                        " inputs (explicit mask required).");
        const ov::PartialShape mask_shape = sdpa->input(kMaskInputIdx).get_partial_shape();
        if (!mask_param) {
            OPENVINO_ASSERT(mask_shape.is_static(),
                            "[SWA] Sliding SDPA '",
                            sdpa->get_friendly_name(),
                            "' mask input must be fully static before PatchSlidingWindowKVLayout.");
            mask_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, mask_shape);
            mask_param->set_friendly_name(ov::npuw::util::kSlidingWindowAttentionMaskParamName);
            mask_param->get_output_tensor(0).set_names(
                {ov::npuw::util::kSlidingWindowAttentionMaskParamName});
        } else {
            OPENVINO_ASSERT(mask_shape == mask_param->get_partial_shape(),
                            "[SWA] All sliding SDPA layers must share the same mask shape at '",
                            sdpa->get_friendly_name(),
                            "' — shape mismatch with previously seen mask shape.");
        }
        sdpa->input(kMaskInputIdx).replace_source_output(mask_param->output(0));
        ++num_externalized;
    }

    if (mask_param) {
        model->add_parameters({mask_param});
        LOG_INFO("[SWA] Externalized " << num_externalized
                                       << " sliding SDPA mask input(s) as '"
                                       << ov::npuw::util::kSlidingWindowAttentionMaskParamName
                                       << "' in '" << model->get_friendly_name() << "'.");
    }
}

// Walk backward from the K (port 1) and V (port 2) inputs of each sliding SDPA
// along the strict KV path (only Reshape, Broadcast, and Unsqueeze are permitted
// between SDPA and the past_kv Concat boundary) and patch target-shape constants
// of Reshape/Broadcast nodes (kvcache_size -> new_kv_total at kv_seq_axis).
//
// The strict chain invariant is enforced via OPENVINO_ASSERT. kv_seq_axis is tracked
// via update_kv_axis_backward() for accurate per-dimension matching.
//
// Must be called BEFORE the past_kv Parameter reshape (Step 1).
void privatize_sliding_sdpa_shapes(const std::shared_ptr<ov::Model>& model,
                                   const std::vector<bool>& layer_is_sliding,
                                   size_t kv_seq_axis_initial,
                                   int64_t kvcache_size,
                                   int64_t new_kv_total) {
    size_t num_broadcast = 0, num_reshape = 0;

    for (const auto& model_node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(model_node);
        if (!sdpa) {
            continue;
        }
        const auto classification = classify_sdpa_layer(sdpa, layer_is_sliding);
        const bool is_sliding = classification.first;
        const int layer_idx = classification.second;
        if (!is_sliding || layer_idx < 0) {
            continue;  // skip non-sliding and unclassified SDPAs
        }
        LOG_INFO("[SWA] privatize: sliding SDPA '" << sdpa->get_friendly_name()
                  << "' (layer=" << layer_idx << ")");

        for (const size_t kv_port : {size_t{1}, size_t{2}}) {
            if (kv_port >= sdpa->get_input_size()) {
                continue;
            }
            auto cur = sdpa->input_value(kv_port).get_node_shared_ptr();
            size_t kv_axis = kv_seq_axis_initial;

            while (cur && !ov::is_type<ov::op::v0::Concat>(cur)) {
                OPENVINO_ASSERT(ov::is_type<ov::op::v1::Reshape>(cur) ||
                                    ov::is_type<ov::op::v3::Broadcast>(cur) ||
                                    ov::is_type<ov::op::v0::Unsqueeze>(cur),
                                "[SWA] Unexpected op '",
                                cur->get_type_name(),
                                "' in sliding SDPA '",
                                sdpa->get_friendly_name(),
                                "' KV path (port ",
                                kv_port,
                                "). Only Reshape, Broadcast, and Unsqueeze are allowed between SDPA and past_kv Concat.");

                if (ov::is_type<ov::op::v1::Reshape>(cur) || ov::is_type<ov::op::v3::Broadcast>(cur)) {
                    if (kTargetShapeInputIdx < cur->get_input_size()) {
                        const auto& src = cur->input_value(kTargetShapeInputIdx);
                        auto folded = ov::util::get_constant_from_source(src);
                        if (folded) {
                            auto vals = folded->cast_vector<int64_t>();
                            // Sliding: axis-aware patch kvcache_size -> new_kv_total.
                            if (kv_axis < vals.size() &&
                                (vals[kv_axis] == kvcache_size || vals[kv_axis] == -kvcache_size)) {
                                const int64_t old_val = vals[kv_axis];
                                vals[kv_axis] = (old_val == kvcache_size) ? new_kv_total : -new_kv_total;
                                auto priv = std::make_shared<ov::op::v0::Constant>(
                                    src.get_element_type(), ov::Shape{vals.size()}, vals);
                                priv->set_friendly_name(cur->get_friendly_name() + "/swa_kv_patched");
                                cur->input(kTargetShapeInputIdx).replace_source_output(priv);
                                LOG_INFO("[SWA]   Patched " << cur->get_type_name() << " '"
                                          << cur->get_friendly_name()
                                          << "' kv_axis=" << kv_axis
                                          << ": " << kvcache_size << " -> " << new_kv_total);
                                ov::is_type<ov::op::v1::Reshape>(cur) ? ++num_reshape : ++num_broadcast;
                            }
                        }
                    }
                }

                kv_axis = update_kv_axis_backward(cur, kv_axis, kvcache_size);
                cur = cur->input_value(0).get_node_shared_ptr();
            }

            OPENVINO_ASSERT(ov::is_type<ov::op::v0::Concat>(cur),
                            "[SWA] Sliding SDPA '",
                            sdpa->get_friendly_name(),
                            "' KV path (port ",
                            kv_port,
                            ") did not reach the expected past_kv Concat boundary.");
        }
    }

    if (num_reshape + num_broadcast > 0) {
        LOG_INFO("[SWA] privatize_sliding_sdpa_shapes: patched "
                  << num_reshape << " Reshape, " << num_broadcast << " Broadcast.");
    }
}

}  // namespace

namespace ov::npuw {

PatchSlidingWindowKVLayout::PatchSlidingWindowKVLayout(ov::npuw::util::SwaLayout swa_layout,
                                                       uint32_t kvcache_size,
                                                       uint32_t input_size,
                                                       const KVAxesPosition& kv_axes_position)
    : m_swa_layout(std::move(swa_layout)),
      m_kvcache_size(kvcache_size),
      m_input_size(input_size),
      m_kv_axes_position(kv_axes_position) {}

bool PatchSlidingWindowKVLayout::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (!m_swa_layout.enabled() || m_swa_layout.layer_is_sliding.empty()) {
        LOG_INFO("[SWA] Sliding Window Attention is not configured, skipping " << model->get_friendly_name());
        return false;
    }

    // Keep a SWA window in past for sliding layers when cache headroom is available.
    // In prefill-like static variants input_size can already occupy the full cache,
    // so past must stay zero to keep K-width and mask-width consistent.
    const uint32_t available_past = m_kvcache_size - m_input_size;
    OPENVINO_ASSERT(available_past == 0 || m_swa_layout.window_size <= available_past,
                    "[SWA] window_size (",
                    m_swa_layout.window_size,
                    ") must be <= available_past (kvcache_size - input_size = ",
                    available_past,
                    ").");
    const int64_t new_past = available_past == 0 ? 0 : static_cast<int64_t>(m_swa_layout.window_size);
    const int64_t new_kv_total = static_cast<int64_t>(m_input_size) + new_past;

    LOG_INFO("[SWA] PatchSlidingWindowKVLayout: model='"
              << model->get_friendly_name() << "' kvcache=" << m_kvcache_size << " input=" << m_input_size
              << " window=" << m_swa_layout.window_size << " new_past=" << new_past << " new_kv_total=" << new_kv_total
              << " sliding_layers="
              << std::count(m_swa_layout.layer_is_sliding.begin(), m_swa_layout.layer_is_sliding.end(), true));

    // Step 0: externalize sliding SDPA masks.
    externalize_sliding_sdpa_masks(model, m_swa_layout.layer_is_sliding);

    // Step 1a: if a sliding past_kv has fan-out, require extra users to be ShapeOf and
    // freeze those ShapeOf outputs to constants before any past_kv reshape.
    freeze_shared_shapeof_from_sliding_past_kv(model, m_swa_layout.layer_is_sliding);

    // Step 1b: patch K/V shape constants for sliding SDPAs (kvcache_size -> new_kv_total).
    // Runs before Step 1 so partial shapes in the graph still reflect kvcache_size,
    // enabling accurate kv_seq_axis tracking in privatize_sliding_sdpa_shapes().
    privatize_sliding_sdpa_shapes(model,
                                  m_swa_layout.layer_is_sliding,
                                  m_kv_axes_position.seq_len,
                                  static_cast<int64_t>(m_kvcache_size),
                                  new_kv_total);

    // Step 1: shrink past_key_values Parameter shapes for sliding layers only.
    // Strict contract: every sliding layer must have a corresponding past_key_values input.
    size_t num_params_reshaped = 0;
    std::vector<bool> sliding_layer_seen(m_swa_layout.layer_is_sliding.size(), false);
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        const auto maybe_layer_idx = ov::npuw::util::isPastKeyValuesContiguous(name);
        if (!maybe_layer_idx) {
            continue;
        }
        const size_t layer_idx = static_cast<size_t>(*maybe_layer_idx);
        if (layer_idx >= m_swa_layout.layer_is_sliding.size() || !m_swa_layout.layer_is_sliding[layer_idx]) {
            continue;
        }
        sliding_layer_seen[layer_idx] = true;

        const auto& pshape = input.get_partial_shape();
        OPENVINO_ASSERT(!pshape.rank().is_dynamic() && m_kv_axes_position.seq_len < pshape.size() &&
                            pshape[m_kv_axes_position.seq_len].is_static(),
                        "[SWA] Layer ",
                        layer_idx,
                        " past KV '",
                        name,
                        "' must have static rank with a static seq_len axis at position ",
                        m_kv_axes_position.seq_len,
                        ".");

        auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(input.get_node_shared_ptr());
        const int64_t old_past = pshape[m_kv_axes_position.seq_len].get_length();
        param->get_rt_info()[ov::npuw::util::NPUW_KV_CACHE_SLIDING_RT_KEY] = true;

        ov::PartialShape new_shape = pshape;
        new_shape[m_kv_axes_position.seq_len] = new_past;
        param->set_partial_shape(new_shape);
        ++num_params_reshaped;
        LOG_INFO("[SWA] Layer " << layer_idx << ": past KV '" << name << "' seq_len " << old_past << " -> " << new_past
                                 << " (window=" << m_swa_layout.window_size << ", kvcache_size=" << m_kvcache_size
                                 << ", input_size=" << m_input_size << ", post-concat total=" << new_kv_total << ")");
    }

    for (size_t i = 0; i < m_swa_layout.layer_is_sliding.size(); ++i) {
        OPENVINO_ASSERT(!m_swa_layout.layer_is_sliding[i] || sliding_layer_seen[i],
                        "[SWA] Layer ",
                        i,
                        " is marked as sliding but no past_key_values input was found.");
    }

    LOG_INFO("[SWA] Reshaped " << num_params_reshaped << " past_key_values parameter(s) in '"
                               << model->get_friendly_name() << "' for sliding-window layers.");

    // Step 2: shrink externalized mask width to the same `new_kv_total` for both prefill
    // and generate variants. Content generation stays runtime-side; this pass enforces shape.
    bool found_mask_param = false;
    for (const auto& input : model->inputs()) {
        if (input.get_any_name() != ov::npuw::util::kSlidingWindowAttentionMaskParamName) {
            continue;
        }

        found_mask_param = true;
        const auto& pshape = input.get_partial_shape();
        OPENVINO_ASSERT(!pshape.rank().is_dynamic() && pshape.size() > 0 && pshape[pshape.size() - 1].is_static(),
                        "[SWA] 'sliding_window_attention_mask' must have static non-zero rank with a static last "
                        "axis before PatchSlidingWindowKVLayout.");

        const size_t last_axis = pshape.size() - 1;
        const int64_t old_width = pshape[last_axis].get_length();
        auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(input.get_node_shared_ptr());

        ov::PartialShape new_shape = pshape;
        new_shape[last_axis] = new_kv_total;
        param->set_partial_shape(new_shape);
        LOG_INFO("[SWA] 'sliding_window_attention_mask' last axis " << old_width << " -> " << new_kv_total << " in '"
                                                                    << model->get_friendly_name() << "'.");
        break;
    }
    OPENVINO_ASSERT(found_mask_param, "[SWA] 'sliding_window_attention_mask' input is required when SWA is enabled.");

    model->validate_nodes_and_infer_types();
    return true;
}

}  // namespace ov::npuw
