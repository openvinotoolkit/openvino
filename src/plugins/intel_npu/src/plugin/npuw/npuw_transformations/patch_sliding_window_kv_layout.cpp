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
#include "detect_causal_mask.hpp"
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

// For sliding past_kv Parameters with fan-out > 1, require exactly one Concat
// consumer and allow only ShapeOf as additional consumers.
// Freeze those ShapeOf outputs so later past_kv reshapes do not leak into shared
// dynamic-shape chains; unsupported topologies fail via OPENVINO_ASSERT.
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
        OPENVINO_ASSERT(param, "[SWA] Sliding past_kv input '", name, "' must be a Parameter node.");

        std::vector<std::shared_ptr<ov::Node>> consumers;
        std::unordered_set<const ov::Node*> seen;
        for (const auto& target_input : param->output(0).get_target_inputs()) {
            auto consumer = target_input.get_node()->shared_from_this();
            if (seen.insert(consumer.get()).second) {
                consumers.push_back(consumer);
            }
        }

        if (consumers.size() == 1) {
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
            OPENVINO_ASSERT(folded, "[SWA] Failed to fold ShapeOf output for '", shapeof->get_friendly_name(), "'.");
            auto vals = folded->cast_vector<int64_t>();
            auto frozen = std::make_shared<ov::op::v0::Constant>(shapeof->output(0).get_element_type(),
                                                                 ov::Shape{vals.size()},
                                                                 vals);
            frozen->set_friendly_name(shapeof->get_friendly_name() + "/swa_shapeof_frozen");

            const auto shapeof_users = shapeof->output(0).get_target_inputs();
            for (const auto& user_input : shapeof_users) {
                user_input.replace_source_output(frozen);
            }
            ++num_shapeof_frozen;
            LOG_INFO("[SWA] Froze shared ShapeOf '" << shapeof->get_friendly_name() << "' from sliding past_kv '"
                                                    << name << "' (layer=" << layer_idx << ")");
        }

        OPENVINO_ASSERT(seen_concat,
                        "[SWA] Sliding past_kv '",
                        name,
                        "' has fan-out > 1 but no Concat consumer; unsupported topology.");
    }

    if (num_shapeof_frozen > 0) {
        LOG_INFO("[SWA] freeze_shared_shapeof_from_sliding_past_kv: froze " << num_shapeof_frozen
                                                                            << " shared ShapeOf node(s).");
    }
}

// Classify an SDPA as sliding or non-sliding using DetectAttentionMask rt_info,
// and extract layer_idx from SDPA friendly_name.
// Returns {is_sliding, layer_idx}; layer_idx is -1 if the layer cannot be identified.
// layer_is_sliding[] remains the authoritative per-layer contract and must agree
// with per-SDPA rt_info for in-range layer_idx values.
static std::pair<bool, int> classify_sdpa_layer(const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& sdpa,
                                                const std::vector<bool>& layer_is_sliding) {
    size_t layer_idx = 0;
    const bool has_layer_idx = ov::npuw::util::try_parse_self_attn_layer_idx(sdpa->get_friendly_name(), layer_idx);

    const auto& rt_info = sdpa->get_rt_info();
    const auto it = rt_info.find(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
    const bool is_sliding_by_rt = (it != rt_info.end()) && (it->second.as<int64_t>() >= 0);

    if (!has_layer_idx || layer_idx >= layer_is_sliding.size()) {
        OPENVINO_ASSERT(!is_sliding_by_rt,
                        "[SWA] Sliding SDPA '",
                        sdpa->get_friendly_name(),
                        "' has no valid layer index in its friendly_name; expected pattern 'layers.<idx>.self_attn'.");
        return {false, -1};
    }

    const bool is_sliding_by_layout = layer_is_sliding[layer_idx];
    OPENVINO_ASSERT(is_sliding_by_layout == is_sliding_by_rt,
                    "[SWA] Sliding classification mismatch for SDPA '",
                    sdpa->get_friendly_name(),
                    "' (layer=",
                    layer_idx,
                    "): layer_is_sliding=",
                    is_sliding_by_layout,
                    ", rt_info=",
                    is_sliding_by_rt,
                    ".");

    return {is_sliding_by_rt, static_cast<int>(layer_idx)};
}

// Externalize mask inputs of all sliding SDPAs into a single shared Parameter
// ('sliding_window_attention_mask') and resize its last axis to new_kv_total.
void externalize_sliding_sdpa_masks(const std::shared_ptr<ov::Model>& model,
                                    const std::vector<bool>& layer_is_sliding,
                                    int64_t new_kv_total) {
    constexpr size_t kSdpaMaskInputIndex = 3;

    std::shared_ptr<ov::op::v0::Parameter> mask_param;
    size_t num_externalized = 0;

    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
        if (!sdpa || !classify_sdpa_layer(sdpa, layer_is_sliding).first) {
            continue;
        }
        OPENVINO_ASSERT(sdpa->get_input_size() > kSdpaMaskInputIndex,
                        "[SWA] Sliding SDPA '",
                        sdpa->get_friendly_name(),
                        "' must have at least ",
                        kSdpaMaskInputIndex + 1,
                        " inputs (explicit mask required).");
        const ov::PartialShape mask_shape = sdpa->input(kSdpaMaskInputIndex).get_partial_shape();
        if (!mask_param) {
            OPENVINO_ASSERT(mask_shape.is_static(),
                            "[SWA] Sliding SDPA '",
                            sdpa->get_friendly_name(),
                            "' mask input must be fully static before PatchSlidingWindowKVLayout.");
            mask_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, mask_shape);
            mask_param->set_friendly_name(ov::npuw::util::kSlidingWindowAttentionMaskParamName);
            mask_param->get_output_tensor(0).set_names({ov::npuw::util::kSlidingWindowAttentionMaskParamName});
        } else {
            OPENVINO_ASSERT(mask_shape == mask_param->get_partial_shape(),
                            "[SWA] All sliding SDPA layers must share the same mask shape at '",
                            sdpa->get_friendly_name(),
                            "' — shape mismatch with previously seen mask shape.");
        }
        sdpa->input(kSdpaMaskInputIndex).replace_source_output(mask_param->output(0));
        ++num_externalized;
    }

    OPENVINO_ASSERT(mask_param, "[SWA] 'sliding_window_attention_mask' input is required when SWA is enabled.");

    const auto& pshape = mask_param->get_partial_shape();
    OPENVINO_ASSERT(!pshape.rank().is_dynamic() && pshape.size() > 0 && pshape[pshape.size() - 1].is_static(),
                    "[SWA] 'sliding_window_attention_mask' must have static non-zero rank with a static last "
                    "axis before PatchSlidingWindowKVLayout.");

    const size_t last_axis = pshape.size() - 1;
    const int64_t old_width = pshape[last_axis].get_length();
    ov::PartialShape new_shape = pshape;
    new_shape[last_axis] = new_kv_total;
    mask_param->set_partial_shape(new_shape);

    model->add_parameters({mask_param});
    LOG_INFO("[SWA] Externalized " << num_externalized << " sliding SDPA mask input(s) as '"
                                   << ov::npuw::util::kSlidingWindowAttentionMaskParamName << "' in '"
                                   << model->get_friendly_name() << "'.");
    LOG_INFO("[SWA] 'sliding_window_attention_mask' last axis " << old_width << " -> " << new_kv_total << " in '"
                                                                << model->get_friendly_name() << "'.");
}

// Patch sliding SDPA K/V shape constants from kvcache_size to new_kv_total.
//
// Expected backward path (from SDPA port 1/2):
//   SDPA(K or V)
//     <- Reshape (target_shape)
//     <- Broadcast (target_shape)
//     <- Unsqueeze
//     <- Concat(past_kv, cur_kv)   [stop boundary]
//
// Only Reshape/Broadcast/Unsqueeze are allowed before Concat; any other op fails
// via OPENVINO_ASSERT. The KV sequence dimension is strictly assumed to be the
// second-last axis of each target-shape constant.
void privatize_sliding_sdpa_shapes(const std::shared_ptr<ov::Model>& model,
                                   const std::vector<bool>& layer_is_sliding,
                                   int64_t kvcache_size,
                                   int64_t new_kv_total) {
    constexpr size_t kShapeInputIdx = 1;

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
        LOG_INFO("[SWA] privatize: sliding SDPA '" << sdpa->get_friendly_name() << "' (layer=" << layer_idx << ")");

        for (const size_t kv_port : {size_t{1}, size_t{2}}) {
            if (kv_port >= sdpa->get_input_size()) {
                continue;
            }
            auto cur = sdpa->input_value(kv_port).get_node_shared_ptr();

            while (cur && !ov::is_type<ov::op::v0::Concat>(cur)) {
                OPENVINO_ASSERT(
                    ov::is_type<ov::op::v1::Reshape>(cur) || ov::is_type<ov::op::v3::Broadcast>(cur) ||
                        ov::is_type<ov::op::v0::Unsqueeze>(cur),
                    "[SWA] Unexpected op '",
                    cur->get_type_name(),
                    "' in sliding SDPA '",
                    sdpa->get_friendly_name(),
                    "' KV path (port ",
                    kv_port,
                    "). Only Reshape, Broadcast, and Unsqueeze are allowed between SDPA and past_kv Concat.");

                if (ov::is_type<ov::op::v1::Reshape>(cur) || ov::is_type<ov::op::v3::Broadcast>(cur)) {
                    if (kShapeInputIdx < cur->get_input_size()) {
                        const auto& src = cur->input_value(kShapeInputIdx);
                        auto folded = ov::util::get_constant_from_source(src);
                        if (folded) {
                            auto vals = folded->cast_vector<int64_t>();
                            OPENVINO_ASSERT(vals.size() >= 2,
                                            "[SWA] ",
                                            cur->get_type_name(),
                                            " '",
                                            cur->get_friendly_name(),
                                            "' target shape rank must be >= 2 to use fixed KV axis.");

                            const size_t kv_axis = vals.size() - 2;
                            const int64_t old_val = vals[kv_axis];
                            OPENVINO_ASSERT(old_val == kvcache_size || old_val == -kvcache_size,
                                            "[SWA] ",
                                            cur->get_type_name(),
                                            " '",
                                            cur->get_friendly_name(),
                                            "' expected second-last target-shape value to be +/-",
                                            kvcache_size,
                                            ", got ",
                                            old_val,
                                            ".");

                            vals[kv_axis] = (old_val > 0) ? new_kv_total : -new_kv_total;
                            auto priv = std::make_shared<ov::op::v0::Constant>(src.get_element_type(),
                                                                               ov::Shape{vals.size()},
                                                                               vals);
                            priv->set_friendly_name(cur->get_friendly_name() + "/swa_kv_patched");
                            cur->input(kShapeInputIdx).replace_source_output(priv);
                            LOG_INFO("[SWA]   Patched " << cur->get_type_name() << " '" << cur->get_friendly_name()
                                                        << "' kv_axis=" << kv_axis << ": " << kvcache_size << " -> "
                                                        << new_kv_total);
                        }
                    }
                }
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

    // Step 0: externalize sliding SDPA mask input and resize its width.
    externalize_sliding_sdpa_masks(model, m_swa_layout.layer_is_sliding, new_kv_total);

    // Step 1: prepare sliding KV shape dependencies and patch SDPA K/V target shapes.
    // 1) freeze shared ShapeOf consumers of sliding past_kv fan-out,
    // 2) patch sliding SDPA K/V reshape/broadcast target shapes.
    freeze_shared_shapeof_from_sliding_past_kv(model, m_swa_layout.layer_is_sliding);
    privatize_sliding_sdpa_shapes(model,
                                  m_swa_layout.layer_is_sliding,
                                  static_cast<int64_t>(m_kvcache_size),
                                  new_kv_total);

    // Step 2: shrink past_key_values Parameter shapes for sliding layers only.
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

    model->validate_nodes_and_infer_types();
    return true;
}

}  // namespace ov::npuw
