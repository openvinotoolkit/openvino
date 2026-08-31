// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shrink_sliding_window_kv_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../llm_compiled_model_utils.hpp"
#include "../logging.hpp"
#include "../util.hpp"
#include "detect_causal_mask.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace {

using SDPA = ov::op::v13::ScaledDotProductAttention;

constexpr size_t kSdpaKeyInputIdx = 1;
constexpr size_t kSdpaValueInputIdx = 2;
constexpr size_t kSdpaMaskInputIdx = 3;
constexpr size_t kShapeInputIdx = 1;  // target_shape port of Reshape/Broadcast

// One Reshape/Broadcast target-shape replacement.
struct ShapePatch {
    ov::Input<ov::Node> port;
    std::shared_ptr<ov::op::v0::Constant> patched;
};

// One ShapeOf output to be replaced by its folded value.
struct ShapeOfFreeze {
    std::shared_ptr<ov::op::v3::ShapeOf> node;
    std::shared_ptr<ov::op::v0::Constant> frozen;
};

// Everything the transform phase needs, collected by a single model scan.
// All sliding layers of a model share the same topology, so no per-layer grouping is
// kept: the lists below are flat and their entries are applied independently.
struct SwaPatchPlan {
    ov::PartialShape mask_shape;                                         // common sliding SDPA mask shape
    std::vector<ov::Input<ov::Node>> sdpa_mask_ports;                    // to re-point to the shared mask Parameter
    std::vector<ShapePatch> kv_shape_patches;                            // sliding SDPA K/V target shapes
    std::vector<ShapeOfFreeze> shapeofs_to_freeze;                       // shared ShapeOf on sliding past_kv
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> past_kv_params;  // sliding past_kv to shrink
};

// past_kv Parameter feeding a KV Concat, plus the optional Convert in between.
struct PastKVSource {
    std::shared_ptr<ov::op::v0::Parameter> param;
    std::shared_ptr<ov::Node> convert;  // nullptr when Parameter -> Concat directly
};

// Classify an SDPA as sliding or non-sliding using DetectAttentionMask rt_info,
// and extract layer_idx from SDPA friendly_name.
// Returns {is_sliding, layer_idx}; layer_idx is -1 if the layer cannot be identified.
// layer_is_sliding[] remains the authoritative per-layer contract and must agree
// with per-SDPA rt_info for in-range layer_idx values.
std::pair<bool, int> classify_sdpa_layer(const std::shared_ptr<SDPA>& sdpa, const std::vector<bool>& layer_is_sliding) {
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

// Identify the past_kv Parameter feeding a KV Concat. The other Concat input carries the
// freshly computed KV, so elimination is enough; an optional Convert may sit in between.
PastKVSource find_past_kv_source(const std::shared_ptr<ov::op::v0::Concat>& concat) {
    for (size_t i = 0; i < concat->get_input_size(); ++i) {
        auto src = concat->input_value(i).get_node_shared_ptr();
        std::shared_ptr<ov::Node> convert;
        if (ov::is_type<ov::op::v0::Convert>(src)) {
            convert = src;
            src = src->input_value(0).get_node_shared_ptr();
        }
        if (auto param = ov::as_type_ptr<ov::op::v0::Parameter>(src)) {
            return {param, convert};
        }
    }
    return {nullptr, nullptr};
}

// Walk back from a sliding SDPA K/V input to the past_kv Concat and collect the
// target-shape patches on the way.
//
// Expected backward path (from SDPA port 1/2):
//   SDPA(K or V)
//     <- Reshape (target_shape)
//     <- Broadcast (target_shape)
//     <- Unsqueeze
//     <- Concat(past_kv, cur_kv)   [stop boundary]
//
// The KV sequence dimension is strictly assumed to be the second-last axis of each
// target-shape constant. Supported values on that axis are kvcache_size (patched to
// new_kv_total) and -1 (inferred by Reshape, left unchanged).
std::shared_ptr<ov::op::v0::Concat> scan_kv_path(const std::shared_ptr<SDPA>& sdpa,
                                                 size_t kv_port,
                                                 int64_t kvcache_size,
                                                 int64_t new_kv_total,
                                                 std::vector<ShapePatch>& patches) {
    auto cur = sdpa->input_value(kv_port).get_node_shared_ptr();

    while (cur && !ov::is_type<ov::op::v0::Concat>(cur)) {
        const bool has_target_shape = ov::is_type<ov::op::v1::Reshape>(cur) || ov::is_type<ov::op::v3::Broadcast>(cur);
        OPENVINO_ASSERT(has_target_shape || ov::is_type<ov::op::v0::Unsqueeze>(cur),
                        "[SWA] Unexpected op '",
                        cur->get_type_name(),
                        "' in sliding SDPA '",
                        sdpa->get_friendly_name(),
                        "' KV path (port ",
                        kv_port,
                        "). Only Reshape, Broadcast, and Unsqueeze are allowed between SDPA and past_kv Concat.");

        if (has_target_shape && kShapeInputIdx < cur->get_input_size()) {
            const auto& src = cur->input_value(kShapeInputIdx);
            if (auto folded = ov::util::get_constant_from_source(src)) {
                auto vals = folded->cast_vector<int64_t>();
                OPENVINO_ASSERT(vals.size() >= 2,
                                "[SWA] ",
                                cur->get_type_name(),
                                " '",
                                cur->get_friendly_name(),
                                "' target shape rank must be >= 2 to use fixed KV axis.");

                const size_t kv_axis = vals.size() - 2;
                const int64_t old_val = vals[kv_axis];
                OPENVINO_ASSERT(old_val == kvcache_size || old_val == -1,
                                "[SWA] ",
                                cur->get_type_name(),
                                " '",
                                cur->get_friendly_name(),
                                "' expected second-last target-shape value to be ",
                                kvcache_size,
                                " or -1 (inferred), got ",
                                old_val,
                                ".");

                if (old_val == -1) {
                    LOG_INFO("[SWA]   " << cur->get_type_name() << " '" << cur->get_friendly_name()
                                        << "' kv_axis=" << kv_axis << " uses inferred extent (-1); keep it unchanged.");
                } else {
                    vals[kv_axis] = new_kv_total;
                    auto priv =
                        std::make_shared<ov::op::v0::Constant>(src.get_element_type(), ov::Shape{vals.size()}, vals);
                    priv->set_friendly_name(cur->get_friendly_name() + "/swa_kv_patched");
                    patches.push_back({cur->input(kShapeInputIdx), std::move(priv)});
                    LOG_INFO("[SWA]   Patched " << cur->get_type_name() << " '" << cur->get_friendly_name()
                                                << "' kv_axis=" << kv_axis << ": " << kvcache_size << " -> "
                                                << new_kv_total);
                }
            }
        }
        cur = cur->input_value(0).get_node_shared_ptr();
    }

    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(cur);
    OPENVINO_ASSERT(concat,
                    "[SWA] Sliding SDPA '",
                    sdpa->get_friendly_name(),
                    "' KV path (port ",
                    kv_port,
                    ") did not reach the expected past_kv Concat boundary.");
    return concat;
}

// A sliding past_kv Parameter may only feed its own KV Concat (optionally via Convert)
// and ShapeOf. Fold those ShapeOf outputs now so that shrinking the Parameter later does
// not leak into shared dynamic-shape chains.
void collect_shapeofs(const PastKVSource& source,
                      const std::shared_ptr<ov::op::v0::Concat>& concat,
                      std::vector<ShapeOfFreeze>& out,
                      std::unordered_set<const ov::Node*>& seen) {
    for (const auto& target : source.param->output(0).get_target_inputs()) {
        auto consumer = target.get_node()->shared_from_this();
        if (consumer == concat || consumer == source.convert) {
            continue;
        }

        auto shapeof = ov::as_type_ptr<ov::op::v3::ShapeOf>(consumer);
        OPENVINO_ASSERT(shapeof,
                        "[SWA] Sliding past_kv '",
                        source.param->get_friendly_name(),
                        "' has unsupported consumer '",
                        consumer->get_type_name(),
                        "' (only its KV Concat and ShapeOf are allowed).");

        if (!seen.insert(shapeof.get()).second) {
            continue;
        }
        auto folded = ov::util::get_constant_from_source(shapeof->output(0));
        OPENVINO_ASSERT(folded, "[SWA] Failed to fold ShapeOf output for '", shapeof->get_friendly_name(), "'.");
        auto vals = folded->cast_vector<int64_t>();
        auto frozen =
            std::make_shared<ov::op::v0::Constant>(shapeof->output(0).get_element_type(), ov::Shape{vals.size()}, vals);
        frozen->set_friendly_name(shapeof->get_friendly_name() + "/swa_shapeof_frozen");
        out.push_back({std::move(shapeof), std::move(frozen)});
    }
}

// Scan phase: validate the model topology once and collect every node the transform
// phase touches. All OPENVINO_ASSERTs live here, so an unsupported topology fails
// before any node is modified.
SwaPatchPlan build_plan(const std::shared_ptr<ov::Model>& model,
                        const std::vector<bool>& layer_is_sliding,
                        int64_t kvcache_size,
                        int64_t new_kv_total,
                        size_t seq_len_axis) {
    SwaPatchPlan plan;
    std::vector<bool> sliding_layer_seen(layer_is_sliding.size(), false);
    std::unordered_set<const ov::Node*> seen_shapeof;
    std::unordered_set<const ov::Node*> seen_param;

    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<SDPA>(node);
        if (!sdpa) {
            continue;
        }
        const auto classification = classify_sdpa_layer(sdpa, layer_is_sliding);
        if (!classification.first || classification.second < 0) {
            continue;  // skip non-sliding and unclassified SDPAs
        }
        const size_t layer_idx = static_cast<size_t>(classification.second);
        sliding_layer_seen[layer_idx] = true;
        LOG_INFO("[SWA] Scanning sliding SDPA '" << sdpa->get_friendly_name() << "' (layer=" << layer_idx << ")");

        OPENVINO_ASSERT(sdpa->get_input_size() > kSdpaMaskInputIdx,
                        "[SWA] Sliding SDPA '",
                        sdpa->get_friendly_name(),
                        "' must have at least ",
                        kSdpaMaskInputIdx + 1,
                        " inputs (explicit mask required).");
        const ov::PartialShape mask_shape = sdpa->input(kSdpaMaskInputIdx).get_partial_shape();
        if (plan.sdpa_mask_ports.empty()) {
            OPENVINO_ASSERT(mask_shape.is_static() && mask_shape.size() > 0,
                            "[SWA] Sliding SDPA '",
                            sdpa->get_friendly_name(),
                            "' mask input must be fully static with non-zero rank before "
                            "ShrinkSlidingWindowKVCache.");
            plan.mask_shape = mask_shape;
        } else {
            OPENVINO_ASSERT(mask_shape == plan.mask_shape,
                            "[SWA] All sliding SDPA layers must share the same mask shape at '",
                            sdpa->get_friendly_name(),
                            "' — shape mismatch with previously seen mask shape.");
        }
        plan.sdpa_mask_ports.push_back(sdpa->input(kSdpaMaskInputIdx));

        for (const size_t kv_port : {kSdpaKeyInputIdx, kSdpaValueInputIdx}) {
            OPENVINO_ASSERT(kv_port < sdpa->get_input_size(),
                            "[SWA] Sliding SDPA '",
                            sdpa->get_friendly_name(),
                            "' has no K/V input at port ",
                            kv_port,
                            ".");
            auto concat = scan_kv_path(sdpa, kv_port, kvcache_size, new_kv_total, plan.kv_shape_patches);
            const auto source = find_past_kv_source(concat);
            OPENVINO_ASSERT(source.param,
                            "[SWA] Sliding SDPA '",
                            sdpa->get_friendly_name(),
                            "' KV path (port ",
                            kv_port,
                            ") reached Concat '",
                            concat->get_friendly_name(),
                            "' which has no past_kv Parameter input.");
            collect_shapeofs(source, concat, plan.shapeofs_to_freeze, seen_shapeof);

            if (!seen_param.insert(source.param.get()).second) {
                continue;
            }
            const auto& pshape = source.param->get_partial_shape();
            OPENVINO_ASSERT(
                pshape.rank().is_static() && seq_len_axis < pshape.size() && pshape[seq_len_axis].is_static(),
                "[SWA] Layer ",
                layer_idx,
                " past KV '",
                source.param->get_friendly_name(),
                "' must have static rank with a static seq_len axis at position ",
                seq_len_axis,
                ".");
            plan.past_kv_params.push_back(source.param);
        }
    }

    OPENVINO_ASSERT(!plan.sdpa_mask_ports.empty(),
                    "[SWA] '",
                    ov::npuw::util::kSlidingWindowAttentionMaskParamName,
                    "' input is required when SWA is enabled, but no sliding SDPA was found.");
    for (size_t i = 0; i < layer_is_sliding.size(); ++i) {
        OPENVINO_ASSERT(!layer_is_sliding[i] || sliding_layer_seen[i],
                        "[SWA] Layer ",
                        i,
                        " is marked as sliding but no matching SDPA was found.");
    }
    return plan;
}

// Transform phase: apply an already validated plan. No topology checks here.
void apply_plan(const std::shared_ptr<ov::Model>& model,
                const SwaPatchPlan& plan,
                int64_t new_past,
                int64_t new_kv_total,
                size_t seq_len_axis) {
    // Step 0: externalize sliding SDPA masks into one shared Parameter and resize its width.
    ov::PartialShape mask_shape = plan.mask_shape;
    const size_t last_axis = mask_shape.size() - 1;
    const int64_t old_mask_width = mask_shape[last_axis].get_length();
    mask_shape[last_axis] = new_kv_total;

    auto mask_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, mask_shape);
    mask_param->set_friendly_name(ov::npuw::util::kSlidingWindowAttentionMaskParamName);
    mask_param->get_output_tensor(0).set_names({ov::npuw::util::kSlidingWindowAttentionMaskParamName});
    for (const auto& port : plan.sdpa_mask_ports) {
        port.replace_source_output(mask_param->output(0));
    }
    model->add_parameters({mask_param});
    LOG_INFO("[SWA] Externalized " << plan.sdpa_mask_ports.size() << " sliding SDPA mask input(s) as '"
                                   << ov::npuw::util::kSlidingWindowAttentionMaskParamName << "' in '"
                                   << model->get_friendly_name() << "'; last axis " << old_mask_width << " -> "
                                   << new_kv_total << ".");

    // Step 1: privatize the sliding KV shape dependencies.
    for (const auto& freeze : plan.shapeofs_to_freeze) {
        const auto users = freeze.node->output(0).get_target_inputs();
        for (const auto& user : users) {
            user.replace_source_output(freeze.frozen);
        }
        LOG_INFO("[SWA] Froze shared ShapeOf '" << freeze.node->get_friendly_name() << "'.");
    }
    for (const auto& patch : plan.kv_shape_patches) {
        patch.port.replace_source_output(patch.patched);
    }

    // Step 2: shrink past_key_values Parameter shapes for sliding layers only.
    for (const auto& param : plan.past_kv_params) {
        ov::PartialShape new_shape = param->get_partial_shape();
        const int64_t old_past = new_shape[seq_len_axis].get_length();
        new_shape[seq_len_axis] = new_past;
        param->set_partial_shape(new_shape);
        param->get_rt_info()[ov::npuw::util::NPUW_KV_CACHE_SLIDING_RT_KEY] = true;
        LOG_INFO("[SWA] Past KV '" << param->get_friendly_name() << "' seq_len " << old_past << " -> " << new_past
                                   << " (post-concat total=" << new_kv_total << ")");
    }
    LOG_INFO("[SWA] Shrunk " << plan.past_kv_params.size() << " past_key_values parameter(s) and patched "
                             << plan.kv_shape_patches.size() << " KV shape constant(s) in '"
                             << model->get_friendly_name() << "'.");

    model->validate_nodes_and_infer_types();
}

}  // namespace

namespace ov::npuw {

ShrinkSlidingWindowKVCache::ShrinkSlidingWindowKVCache(ov::npuw::util::SwaLayout swa_layout,
                                                       uint32_t kvcache_size,
                                                       uint32_t input_size,
                                                       const KVAxesPosition& kv_axes_position)
    : m_swa_layout(std::move(swa_layout)),
      m_kvcache_size(kvcache_size),
      m_input_size(input_size),
      m_kv_axes_position(kv_axes_position) {}

bool ShrinkSlidingWindowKVCache::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (!m_swa_layout.enabled() || m_swa_layout.layer_is_sliding.empty()) {
        LOG_INFO("[SWA] Sliding Window Attention is not configured, skipping " << model->get_friendly_name());
        return false;
    }

    // available_past == 0 means the prompt is processed in a single shot (input_size ==
    // kvcache_size): there is no past region at all, so sliding layers keep no past KV.
    const uint32_t available_past = m_kvcache_size - m_input_size;
    OPENVINO_ASSERT(available_past == 0 || m_swa_layout.window_size <= available_past,
                    "[SWA] window_size (",
                    m_swa_layout.window_size,
                    ") must be <= available_past (kvcache_size - input_size = ",
                    available_past,
                    ").");
    const int64_t new_past = available_past == 0 ? 0 : static_cast<int64_t>(m_swa_layout.window_size);
    const int64_t new_kv_total = static_cast<int64_t>(m_input_size) + new_past;

    LOG_INFO("[SWA] ShrinkSlidingWindowKVCache: model='"
             << model->get_friendly_name() << "' kvcache=" << m_kvcache_size << " input=" << m_input_size
             << " window=" << m_swa_layout.window_size << " new_past=" << new_past << " new_kv_total=" << new_kv_total
             << " sliding_layers="
             << std::count(m_swa_layout.layer_is_sliding.begin(), m_swa_layout.layer_is_sliding.end(), true));

    const size_t seq_len_axis = static_cast<size_t>(m_kv_axes_position.seq_len);
    const auto plan = build_plan(model,
                                 m_swa_layout.layer_is_sliding,
                                 static_cast<int64_t>(m_kvcache_size),
                                 new_kv_total,
                                 seq_len_axis);
    apply_plan(model, plan, new_past, new_kv_total, seq_len_axis);
    return true;
}

}  // namespace ov::npuw
