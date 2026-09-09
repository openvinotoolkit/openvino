// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shrink_sliding_window_kv_cache.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_set>

#include "../kv_cache_sliding_window_manager.hpp"
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

// Returns the uniform SWA window size across all SDPA nodes annotated as sliding window,
// or 0 if no sliding window layer is found.
// Throws if sliding layers use different window sizes.
uint32_t detect_swa_window_size(const std::shared_ptr<ov::Model>& model) {
    int64_t detected_window = 0;
    bool has_detected_window = false;

    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<SDPA>(node);
        if (!sdpa) {
            continue;
        }

        const auto& rt_info = sdpa->get_rt_info();
        const auto it = rt_info.find(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
        if (it == rt_info.end()) {
            continue;
        }

        const int64_t encoded = it->second.as<int64_t>();
        if (encoded < 0) {
            continue;  // Causal (or unrecognized)
        }

        if (!has_detected_window) {
            detected_window = encoded;
            has_detected_window = true;
        } else {
            OPENVINO_ASSERT(detected_window == encoded,
                            "NPUW SWA: inconsistent window sizes across layers (",
                            detected_window,
                            " vs ",
                            encoded,
                            ").");
        }
    }

    if (!has_detected_window) {
        LOG_DEBUG("[SWA] No sliding-window layer detected; SWA support disabled.");
        return 0;
    }

    OPENVINO_ASSERT(
        detected_window > 0 && detected_window <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "NPUW SWA: invalid window size: ",
        detected_window);

    LOG_INFO("[SWA] Sliding Window Attention is ENABLED: window_size=" << detected_window);
    return static_cast<uint32_t>(detected_window);
}

constexpr size_t kSdpaKeyInputIdx = 1;
constexpr size_t kSdpaValueInputIdx = 2;
constexpr size_t kSdpaMaskInputIdx = 3;
constexpr size_t kShapeInputIdx = 1;  // target_shape port of Reshape/Broadcast

// past_kv Parameter feeding a KV Concat, plus the optional Convert in between.
struct PastKVSource {
    std::shared_ptr<ov::op::v0::Parameter> param;
    std::shared_ptr<ov::Node> convert;  // nullptr when Parameter -> Concat directly
};

// Classifies an SDPA node as SWA (sliding-window) or not, using attention mask
// classification information attached to the node earlier in the pipeline.
bool is_sliding_sdpa(const std::shared_ptr<SDPA>& sdpa) {
    const auto& rt_info = sdpa->get_rt_info();
    const auto it = rt_info.find(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
    return (it != rt_info.end()) && (it->second.as<int64_t>() >= 0);
}

// Identify the past_kv Parameter feeding a KV Concat. The other Concat input carries the
// freshly computed KV, so elimination is enough; an optional Convert may sit in between.
PastKVSource find_past_kv_source(const std::shared_ptr<ov::op::v0::Concat>& concat) {
    for (size_t i = 0; i < concat->get_input_size(); ++i) {
        auto src = concat->input_value(i).get_node_shared_ptr();
        std::shared_ptr<ov::op::v0::Convert> convert;
        if (auto as_convert = ov::as_type_ptr<ov::op::v0::Convert>(src)) {
            convert = as_convert;
            src = src->input_value(0).get_node_shared_ptr();
        }
        if (auto param = ov::as_type_ptr<ov::op::v0::Parameter>(src)) {
            return {param, convert};
        }
    }
    return {nullptr, nullptr};
}

// Walk back from an SWA SDPA K/V input to the past_kv Concat, patching target-shape
// constants (kvcache_size -> new_kv_total) in place as they are discovered.
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
                                                 int64_t new_kv_total) {
    auto cur = sdpa->input_value(kv_port).get_node_shared_ptr();

    while (cur && !ov::is_type<ov::op::v0::Concat>(cur)) {
        const bool has_target_shape =
            ov::is_type<ov::op::v1::Reshape>(cur) || ov::is_type<ov::op::util::BroadcastBase>(cur);
        OPENVINO_ASSERT(has_target_shape || ov::is_type<ov::op::v0::Unsqueeze>(cur),
                        "[SWA] Unexpected op '",
                        cur->get_type_name(),
                        "' between SDPA and past_kv Concat (only Reshape/Broadcast/Unsqueeze allowed).");

        if (has_target_shape) {
            const auto& src = cur->input_value(kShapeInputIdx);
            auto folded = ov::util::get_constant_from_source(src);
            OPENVINO_ASSERT(folded, "[SWA] ", cur->get_type_name(), " target shape must be constant-foldable.");

            auto vals = folded->cast_vector<int64_t>();
            OPENVINO_ASSERT(vals.size() >= 2, "[SWA] ", cur->get_type_name(), " target shape rank must be >= 2.");

            const size_t kv_axis = vals.size() - 2;
            const int64_t old_val = vals[kv_axis];
            OPENVINO_ASSERT(old_val == kvcache_size || old_val == -1,
                            "[SWA] ",
                            cur->get_type_name(),
                            " target shape kv_axis expected ",
                            kvcache_size,
                            " or -1, got ",
                            old_val,
                            ".");

            if (old_val == -1) {
                LOG_DEBUG("[SWA]   " << cur->get_type_name() << " '" << cur->get_friendly_name()
                                     << "' kv_axis=" << kv_axis << " uses inferred extent (-1); keep it unchanged.");
            } else {
                vals[kv_axis] = new_kv_total;
                auto priv =
                    std::make_shared<ov::op::v0::Constant>(src.get_element_type(), ov::Shape{vals.size()}, vals);
                priv->set_friendly_name(cur->get_friendly_name() + "/swa_kv_patched");
                cur->input(kShapeInputIdx).replace_source_output(priv);
                LOG_DEBUG("[SWA]   Patched " << cur->get_type_name() << " '" << cur->get_friendly_name() << "' kv_axis="
                                             << kv_axis << ": " << kvcache_size << " -> " << new_kv_total);
            }
        }
        cur = cur->input_value(0).get_node_shared_ptr();
    }

    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(cur);
    OPENVINO_ASSERT(concat, "[SWA] KV path (port ", kv_port, ") did not reach past_kv Concat.");
    return concat;
}

void freeze_shapeof_consumers(const ov::Output<ov::Node>& producer_output,
                              const std::shared_ptr<ov::Node>& skip_consumer,
                              const std::shared_ptr<ov::op::v0::Concat>& concat,
                              const char* via,
                              std::unordered_set<const ov::Node*>& seen) {
    for (const auto& target : producer_output.get_target_inputs()) {
        auto consumer = target.get_node()->shared_from_this();
        if (skip_consumer && consumer == skip_consumer) {
            continue;
        }
        if (consumer == concat) {
            continue;
        }

        auto shapeof = ov::as_type_ptr<ov::op::util::ShapeOfBase>(consumer);
        OPENVINO_ASSERT(shapeof,
                        "[SWA] past_kv has unsupported consumer '",
                        consumer->get_type_name(),
                        "' via ",
                        via,
                        " (only Concat/ShapeOf allowed).");

        if (!seen.insert(shapeof.get()).second) {
            continue;
        }
        auto folded = ov::util::get_constant_from_source(shapeof->output(0));
        OPENVINO_ASSERT(folded, "[SWA] Failed to fold ShapeOf output.");
        auto vals = folded->cast_vector<int64_t>();
        auto frozen =
            std::make_shared<ov::op::v0::Constant>(shapeof->output(0).get_element_type(), ov::Shape{vals.size()}, vals);
        frozen->set_friendly_name(shapeof->get_friendly_name() + "/swa_shapeof_frozen");

        const auto users = shapeof->output(0).get_target_inputs();
        for (const auto& user : users) {
            user.replace_source_output(frozen);
        }
        LOG_DEBUG("[SWA] Froze shared ShapeOf '" << shapeof->get_friendly_name() << "'.");
    }
}

// An SWA past_kv Parameter may only feed its own KV Concat and ShapeOf, optionally via an
// in-between Convert. Freeze each ShapeOf output to a constant before the Parameter is
// resized, so its consumers keep seeing the pre-shrink shape.
//
// Supported consumers of past_kv (Convert is optional):
//   Parameter -> [Convert] -> Concat(past_kv, cur_kv)   [KV path; left untouched]
//   Parameter -> [Convert] -> ShapeOf                   [frozen to a Constant]
void freeze_shapeofs(const PastKVSource& source,
                     const std::shared_ptr<ov::op::v0::Concat>& concat,
                     std::unordered_set<const ov::Node*>& seen) {
    freeze_shapeof_consumers(source.param->output(0), source.convert, concat, "Parameter", seen);

    if (source.convert) {
        freeze_shapeof_consumers(source.convert->output(0), nullptr, concat, "Convert", seen);
    }
}

// Scans every SDPA node once, in topological order. For each SWA one found, externalizes
// its mask into one shared Parameter (created on first encounter), patches its K/V shape
// dependencies, and shrinks its past_kv Parameter. Topology violations abort with an error
// as soon as they are found.
void scan_and_patch(const std::shared_ptr<ov::Model>& model,
                    int64_t kvcache_size,
                    int64_t new_past,
                    int64_t new_kv_total,
                    size_t seq_len_axis) {
    std::unordered_set<const ov::Node*> seen_shapeof;
    std::unordered_set<const ov::Node*> seen_param;
    std::shared_ptr<ov::op::v0::Parameter> mask_param;
    ov::PartialShape mask_shape;
    ov::element::Type mask_type;
    size_t num_masks_externalized = 0;
    size_t num_params_shrunk = 0;

    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<SDPA>(node);
        if (!sdpa) {
            continue;
        }
        if (!is_sliding_sdpa(sdpa)) {
            continue;  // skip non-SWA SDPAs
        }
        LOG_DEBUG("[SWA] Patching SWA SDPA '" << sdpa->get_friendly_name() << "'");

        // Mask: externalize into one shared Parameter, created on first SWA SDPA encountered.
        OPENVINO_ASSERT(sdpa->get_input_size() > kSdpaMaskInputIdx,
                        "[SWA] SDPA must have at least ",
                        kSdpaMaskInputIdx + 1,
                        " inputs (mask required).");
        auto mask_port = sdpa->input(kSdpaMaskInputIdx);
        const ov::PartialShape sdpa_mask_shape = mask_port.get_partial_shape();
        const ov::element::Type sdpa_mask_type = sdpa->input_value(kSdpaMaskInputIdx).get_element_type();
        if (!mask_param) {
            OPENVINO_ASSERT(sdpa_mask_shape.is_static() && sdpa_mask_shape.size() > 0,
                            "[SWA] SDPA mask input must be static with non-zero rank.");
            mask_shape = sdpa_mask_shape;
            mask_type = sdpa_mask_type;
            ov::PartialShape param_shape = mask_shape;
            const size_t last_axis = param_shape.size() - 1;
            const int64_t old_mask_width = param_shape[last_axis].get_length();
            param_shape[last_axis] = new_kv_total;

            mask_param = std::make_shared<ov::op::v0::Parameter>(mask_type, param_shape);
            mask_param->set_friendly_name(ov::npuw::util::kSlidingWindowAttentionMaskParamName);
            mask_param->get_output_tensor(0).set_names({ov::npuw::util::kSlidingWindowAttentionMaskParamName});
            model->add_parameters({mask_param});
            LOG_DEBUG("[SWA] Externalized SWA SDPA mask as '" << ov::npuw::util::kSlidingWindowAttentionMaskParamName
                                                              << "'; last axis " << old_mask_width << " -> "
                                                              << new_kv_total << ".");
        } else {
            OPENVINO_ASSERT(sdpa_mask_shape == mask_shape, "[SWA] mask shape mismatch across SWA layers.");
            OPENVINO_ASSERT(sdpa_mask_type == mask_type, "[SWA] mask type mismatch across SWA layers.");
        }
        mask_port.replace_source_output(mask_param->output(0));
        ++num_masks_externalized;

        // K/V: patch shape constants along the way, then shrink the past_kv Parameter (once).
        for (const size_t kv_port : {kSdpaKeyInputIdx, kSdpaValueInputIdx}) {
            auto concat = scan_kv_path(sdpa, kv_port, kvcache_size, new_kv_total);
            const auto source = find_past_kv_source(concat);
            OPENVINO_ASSERT(source.param, "[SWA] KV path (port ", kv_port, ") Concat has no past_kv Parameter input.");
            freeze_shapeofs(source, concat, seen_shapeof);

            if (!seen_param.insert(source.param.get()).second) {
                continue;
            }
            const auto& pshape = source.param->get_partial_shape();
            OPENVINO_ASSERT(
                pshape.rank().is_static() && seq_len_axis < pshape.size() && pshape[seq_len_axis].is_static(),
                "[SWA] Past KV '",
                source.param->get_friendly_name(),
                "' needs static rank and static seq_len axis (",
                seq_len_axis,
                ").");

            ov::PartialShape new_shape = pshape;
            const int64_t old_past = new_shape[seq_len_axis].get_length();
            new_shape[seq_len_axis] = new_past;
            source.param->set_partial_shape(new_shape);
            source.param->get_rt_info()[ov::npuw::util::NPUW_KV_CACHE_SLIDING_RT_KEY] = true;
            ++num_params_shrunk;
            LOG_DEBUG("[SWA] Past KV '" << source.param->get_friendly_name() << "' seq_len " << old_past << " -> "
                                        << new_past << " (post-concat total=" << new_kv_total << ")");
        }
    }

    LOG_INFO("[SWA] Externalized " << num_masks_externalized << " SWA SDPA mask input(s) as '"
                                   << ov::npuw::util::kSlidingWindowAttentionMaskParamName << "' and shrunk "
                                   << num_params_shrunk << " past_key_values parameter(s) in '"
                                   << model->get_friendly_name() << "'.");

    model->validate_nodes_and_infer_types();
}

}  // namespace

namespace ov::npuw {

ShrinkSlidingWindowKVCache::ShrinkSlidingWindowKVCache(uint32_t kvcache_size,
                                                       uint32_t input_size,
                                                       const KVAxesPosition& kv_axes_position)
    : m_kvcache_size(kvcache_size),
      m_input_size(input_size),
      m_kv_axes_position(kv_axes_position) {}

bool ShrinkSlidingWindowKVCache::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const uint32_t window_size = detect_swa_window_size(model);
    if (window_size == 0) {
        LOG_DEBUG("[SWA] Sliding Window Attention is not configured, skipping " << model->get_friendly_name());
        return false;
    }

    OPENVINO_ASSERT(m_input_size <= m_kvcache_size,
                    "[SWA] input_size (",
                    m_input_size,
                    ") exceeds kvcache_size (",
                    m_kvcache_size,
                    ").");

    // available_past == 0 means prefill consumes the whole KV budget in one shot.
    // Typical cases:
    // 1) chunk prefill is disabled; or
    // 2) prefill_chunk_size == max_prompt_len (for example, both are 1k).
    // Result: there is no past region, so SWA layers keep no past KV.
    const uint32_t available_past = m_kvcache_size - m_input_size;
    OPENVINO_ASSERT(available_past == 0 || window_size <= available_past,
                    "[SWA] window_size (",
                    window_size,
                    ") exceeds available_past (",
                    available_past,
                    ").");
    const int64_t new_past = available_past == 0 ? 0 : static_cast<int64_t>(window_size);
    const int64_t new_kv_total = static_cast<int64_t>(m_input_size) + new_past;

    LOG_INFO("[SWA] ShrinkSlidingWindowKVCache: model='"
             << model->get_friendly_name() << "' kvcache=" << m_kvcache_size << " input=" << m_input_size
             << " window=" << window_size << " new_past=" << new_past << " new_kv_total=" << new_kv_total);

    const size_t seq_len_axis = static_cast<size_t>(m_kv_axes_position.seq_len);
    scan_and_patch(model, static_cast<int64_t>(m_kvcache_size), new_past, new_kv_total, seq_len_axis);
    return true;
}

}  // namespace ov::npuw
