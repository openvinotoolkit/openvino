// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "patch_sliding_window_kvcache.hpp"

#include <algorithm>
#include <regex>

#include "../llm_compiled_model_utils.hpp"
#include "../logging.hpp"
#include "../util.hpp"
#include "detect_causal_mask.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/slice.hpp"

namespace {

// SDPA input layout: query=0, key=1, value=2, [attn_mask=3, [scale=4, [sink=5]]] - see
// openvino/op/scaled_dot_product_attention.hpp.
constexpr size_t kMaskInputIdx = 3;
constexpr size_t kTargetShapeInputIdx = 1;  // Reshape/Broadcast target-shape input.
constexpr size_t kSliceBeginInputIdx = 1;
constexpr size_t kSliceEndInputIdx = 2;

void set_mask_param_name(const std::shared_ptr<ov::op::v0::Parameter>& param, const std::string& name) {
    param->set_friendly_name(name);
    param->get_output_tensor(0).set_names({name});
}

// Matches e.g. "past_key_values.3.key" / "past_key_values.12.value" -> layer index 3 / 12.
const std::regex& kv_param_regex() {
    static const std::regex re(R"(past_key_values\.(\d+)\.(?:key|value))");
    return re;
}

bool try_parse_layer_idx(const std::string& text, const std::regex& re, size_t& out_idx) {
    std::smatch m;
    if (!std::regex_search(text, m, re)) {
        return false;
    }
    out_idx = static_cast<size_t>(std::stoul(m[1].str()));
    return true;
}

struct TargetKVLengths {
    int64_t new_past = 0;
    int64_t new_kv_total = 0;
};

TargetKVLengths calculate_target_kv_lengths(uint32_t window_size, uint32_t kvcache_size, uint32_t input_size) {
    // Keep enough past for the earliest new query row in this call.
    // For one-shot prefill (input_size >= kvcache_size), the explicit past buffer is 0.
    const uint32_t new_past_u32 =
        (input_size >= kvcache_size) ? 0u : (window_size < kvcache_size ? window_size : (kvcache_size - input_size));
    TargetKVLengths lengths;
    lengths.new_past = static_cast<int64_t>(new_past_u32);
    lengths.new_kv_total = lengths.new_past + static_cast<int64_t>(input_size);
    return lengths;
}

bool is_sliding_sdpa(const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& sdpa) {
    const auto& rt_info = sdpa->get_rt_info();
    const auto it = rt_info.find(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
    return it != rt_info.end() && it->second.as<int64_t>() >= 0;
}

struct ShapeInputSnapshot {
    std::shared_ptr<ov::Node> consumer;
    size_t input_index = 0;
    size_t layer_idx = 0;
    bool layer_is_sliding = false;
    ov::element::Type elem_type;
    std::vector<int64_t> original_values;
};

void snapshot_kvcache_dependent_input(const std::shared_ptr<ov::Node>& node,
                                      size_t input_idx,
                                      const std::vector<bool>& layer_is_sliding,
                                      uint32_t kvcache_size,
                                      std::vector<ShapeInputSnapshot>& snapshots) {
    size_t layer_idx = 0;
    if (!ov::npuw::util::try_parse_self_attn_layer_idx(node->get_friendly_name(), layer_idx)) {
        return;
    }
    if (layer_idx >= layer_is_sliding.size() || input_idx >= node->get_input_size()) {
        return;
    }

    auto folded = ov::util::get_constant_from_source(node->input_value(input_idx));
    if (!folded) {
        LOG_WARN("[SWA] Layer " << layer_idx << ": " << node->get_type_name() << " '" << node->get_friendly_name()
                                << "' input(" << input_idx << ") is not constant-foldable, cannot verify/patch it.");
        return;
    }

    auto values = folded->cast_vector<int64_t>();
    const int64_t kv = static_cast<int64_t>(kvcache_size);
    const bool has_kvcache_dim = std::any_of(values.begin(), values.end(), [kv](int64_t v) {
        return v == kv || v == -kv;
    });
    if (!has_kvcache_dim) {
        return;
    }

    snapshots.push_back(ShapeInputSnapshot{node,
                                           input_idx,
                                           layer_idx,
                                           layer_is_sliding[layer_idx],
                                           folded->get_element_type(),
                                           std::move(values)});
}

std::vector<ShapeInputSnapshot> collect_shape_input_snapshots(const std::shared_ptr<ov::Model>& model,
                                                              const std::vector<bool>& layer_is_sliding,
                                                              uint32_t kvcache_size) {
    std::vector<ShapeInputSnapshot> snapshots;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v1::Reshape>(node) || ov::is_type<ov::op::v3::Broadcast>(node)) {
            snapshot_kvcache_dependent_input(node, kTargetShapeInputIdx, layer_is_sliding, kvcache_size, snapshots);
            continue;
        }
        if (ov::is_type<ov::op::v8::Slice>(node)) {
            snapshot_kvcache_dependent_input(node, kSliceBeginInputIdx, layer_is_sliding, kvcache_size, snapshots);
            snapshot_kvcache_dependent_input(node, kSliceEndInputIdx, layer_is_sliding, kvcache_size, snapshots);
        }
    }
    return snapshots;
}

size_t patch_shape_inputs_from_snapshots(const std::vector<ShapeInputSnapshot>& snapshots,
                                         uint32_t kvcache_size,
                                         int64_t new_kv_total,
                                         size_t& patched_count) {
    patched_count = 0;
    size_t privatized_count = 0;
    const int64_t kv = static_cast<int64_t>(kvcache_size);

    for (const auto& snapshot : snapshots) {
        auto values = snapshot.original_values;
        const int64_t corrected = snapshot.layer_is_sliding ? new_kv_total : kv;
        bool value_changed = false;
        for (auto& v : values) {
            if (v == kv && v != corrected) {
                v = corrected;
                value_changed = true;
            } else if (v == -kv && v != -corrected) {
                v = -corrected;
                value_changed = true;
            }
        }

        auto new_const = std::make_shared<ov::op::v0::Constant>(snapshot.elem_type, ov::Shape{values.size()}, values);
        new_const->set_friendly_name(snapshot.consumer->get_friendly_name() + "/swa_shape_patched_" +
                                     std::to_string(snapshot.input_index));
        snapshot.consumer->input(snapshot.input_index).replace_source_output(new_const);
        ++privatized_count;

        if (value_changed) {
            ++patched_count;
            LOG_INFO("[SWA] Layer " << snapshot.layer_idx << ": patched " << snapshot.consumer->get_type_name() << " '"
                                    << snapshot.consumer->get_friendly_name() << "' input(" << snapshot.input_index
                                    << ") constant " << kvcache_size << " -> " << new_kv_total);
        }
    }

    return privatized_count;
}

}  // namespace

namespace ov::npuw {

PatchSlidingWindowKVCache::PatchSlidingWindowKVCache(ov::npuw::util::SwaLayout swa_layout,
                                                     uint32_t kvcache_size,
                                                     uint32_t input_size,
                                                     const KVAxesPosition& kv_axes_position)
    : m_swa_layout(std::move(swa_layout)),
      m_kvcache_size(kvcache_size),
      m_input_size(input_size),
      m_kv_axes_position(kv_axes_position) {}

bool PatchSlidingWindowKVCache::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (!m_swa_layout.enabled() || m_swa_layout.layer_is_sliding.empty()) {
        LOG_INFO("[SWA] Sliding Window Attention is not configured, skipping " << model->get_friendly_name());
        return false;
    }

    // Compute target past/KV widths once and reuse in all subsequent steps.
    const auto kv_lengths = calculate_target_kv_lengths(m_swa_layout.window_size, m_kvcache_size, m_input_size);
    const int64_t new_past = kv_lengths.new_past;
    const int64_t new_kv_total = kv_lengths.new_kv_total;
    bool changed = false;

    // Step 0: externalize only sliding-layer SDPA masks to one shared input
    // (`sliding_window_attention_mask`). Full-attention layers stay untouched because this
    // pass does not shrink their KV shape.
    {
        std::shared_ptr<ov::op::v0::Parameter> sliding_mask_param;
        size_t num_cut = 0;
        for (const auto& node : model->get_ordered_ops()) {
            auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
            if (!sdpa) {
                continue;
            }
            if (!is_sliding_sdpa(sdpa)) {
                // Full-attention/causal SDPA - leave its mask input (or is_causal attribute)
                // exactly as the exporter produced it.
                continue;
            }
            // Every SlidingWindow-classified SDPA is guaranteed (by detect_causal_mask.cpp's
            // matchers) to already carry an explicit, statically-shaped mask input - none of them
            // are `is_causal`-only.
            NPUW_ASSERT(sdpa->get_input_size() > kMaskInputIdx);
            const ov::PartialShape mask_shape = sdpa->input(kMaskInputIdx).get_partial_shape();
            if (!sliding_mask_param) {
                NPUW_ASSERT(mask_shape.is_static());
                sliding_mask_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, mask_shape);
                set_mask_param_name(sliding_mask_param, ov::npuw::util::kSlidingWindowAttentionMaskParamName);
            } else {
                NPUW_ASSERT(mask_shape == sliding_mask_param->get_partial_shape());
            }
            sdpa->input(kMaskInputIdx).replace_source_output(sliding_mask_param->output(0));
            ++num_cut;
        }
        if (sliding_mask_param) {
            model->add_parameters({sliding_mask_param});
            changed = true;
            LOG_INFO("[SWA] Externalized " << num_cut
                                           << " sliding SDPA mask input(s) as 'sliding_window_attention_mask' in '"
                                           << model->get_friendly_name() << "'.");
        }
    }

    // Step 0b: snapshot all KV-size-dependent shape inputs (Broadcast/Reshape target-shape,
    // Slice begin/end). Step 2 will rebuild them as per-consumer Constants to break
    // cross-layer shared shape chains.
    const auto shape_snapshots = collect_shape_input_snapshots(model, m_swa_layout.layer_is_sliding, m_kvcache_size);

    // Step 1: shrink past_key_values Parameter shapes for sliding layers only.
    // Validation is deferred to the end, after Step 2 fixes dependent shape inputs.
    size_t num_params_reshaped = 0;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        size_t layer_idx = 0;
        if (!try_parse_layer_idx(name, kv_param_regex(), layer_idx)) {
            continue;
        }
        if (layer_idx >= m_swa_layout.layer_is_sliding.size() || !m_swa_layout.layer_is_sliding[layer_idx]) {
            continue;
        }
        const auto& pshape = input.get_partial_shape();
        if (pshape.rank().is_dynamic() || m_kv_axes_position.seq_len >= pshape.size() ||
            !pshape[m_kv_axes_position.seq_len].is_static()) {
            LOG_WARN("[SWA] Layer " << layer_idx << ": past KV parameter '" << name
                                    << "' has a non-static seq_len axis, skipping reshape. Was ReshapeToStatic "
                                       "applied before PatchSlidingWindowKVCache?");
            continue;
        }
        const int64_t old_past = pshape[m_kv_axes_position.seq_len].get_length();
        if (new_past == old_past) {
            continue;
        }
        auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(input.get_node_shared_ptr());
        if (!param) {
            LOG_WARN("[SWA] Layer " << layer_idx << ": past KV input '" << name
                                    << "' is not backed by a Parameter node, skipping.");
            continue;
        }
        ov::PartialShape new_shape = pshape;
        new_shape[m_kv_axes_position.seq_len] = new_past;
        param->set_partial_shape(new_shape);
        param->get_rt_info()[ov::npuw::util::NPUW_KV_CACHE_SLIDING_RT_KEY] = true;
        changed = true;
        ++num_params_reshaped;
        LOG_INFO("[SWA] Layer " << layer_idx << ": past KV '" << name << "' seq_len " << old_past << " -> " << new_past
                                << " (window=" << m_swa_layout.window_size << ", kvcache_size=" << m_kvcache_size
                                << ", input_size=" << m_input_size << ", post-concat total=" << new_kv_total << ")");
    }
    if (num_params_reshaped > 0) {
        LOG_INFO("[SWA] Reshaped " << num_params_reshaped << " past_key_values parameter(s) in '"
                                   << model->get_friendly_name() << "' for sliding-window layers.");
    } else {
        LOG_INFO("[SWA] No past_key_values parameters required a reshape in '" << model->get_friendly_name() << "'.");
    }

    // Step 1b: shrink externalized mask width to the same `new_kv_total` for both prefill
    // and generate variants. Content generation stays runtime-side; this pass enforces shape.
    for (const auto& input : model->inputs()) {
        if (input.get_any_name() != ov::npuw::util::kSlidingWindowAttentionMaskParamName) {
            continue;
        }
        const auto& pshape = input.get_partial_shape();
        if (pshape.rank().is_dynamic() || pshape.size() == 0 || !pshape[pshape.size() - 1].is_static()) {
            LOG_WARN("[SWA] 'sliding_window_attention_mask' has a dynamic last axis, skipping shrink. Was "
                     "ReshapeToStatic applied before PatchSlidingWindowKVCache?");
            break;
        }
        const size_t last_axis = pshape.size() - 1;
        const int64_t old_width = pshape[last_axis].get_length();
        if (new_kv_total == old_width) {
            break;  // already the right width (e.g. window_size >= kvcache_size for this variant).
        }
        auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(input.get_node_shared_ptr());
        if (!param) {
            LOG_WARN("[SWA] 'sliding_window_attention_mask' is not backed by a Parameter node, skipping.");
            break;
        }
        ov::PartialShape new_shape = pshape;
        new_shape[last_axis] = new_kv_total;
        param->set_partial_shape(new_shape);
        changed = true;
        LOG_INFO("[SWA] 'sliding_window_attention_mask' last axis " << old_width << " -> " << new_kv_total << " in '"
                                                                    << model->get_friendly_name() << "'.");
        break;
    }

    // Step 2: rebuild snapshotted shape inputs as private Constants.
    // Sliding layers get `m_kvcache_size -> new_kv_total`; full-attention layers are privatized
    // unchanged to avoid shared-shape side effects.
    {
        size_t patched_count = 0;
        const size_t privatized_count =
            patch_shape_inputs_from_snapshots(shape_snapshots, m_kvcache_size, new_kv_total, patched_count);
        if (privatized_count > 0) {
            changed = true;
            LOG_INFO("[SWA] Privatized " << privatized_count << " Broadcast/Reshape target-shape input(s) ("
                                         << patched_count << " value-corrected, " << (privatized_count - patched_count)
                                         << " guarded-only) feeding self_attn SDPA node(s) in '"
                                         << model->get_friendly_name() << "'.");
        }
    }

    if (changed) {
        model->validate_nodes_and_infer_types();
    }
    return changed;
}

}  // namespace ov::npuw
