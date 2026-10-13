// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_to_static_gqa.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <unordered_set>

#include "../logging.hpp"
#include "../util.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"

namespace {

// Scans `model`'s past_key/past_value Parameters for a dynamic dimension and returns the
// axis to pin to the configured static capacity, keyed by Parameter friendly name. A
// KV-cache Parameter with more than one dynamic dimension is ambiguous and not something
// we can safely resolve.
std::unordered_map<std::string, size_t> find_dynamic_kv_cache_axes(const std::shared_ptr<ov::Model>& model) {
    std::unordered_map<std::string, size_t> result;

    const auto resolve_dynamic_axis =
        [](const std::shared_ptr<ov::op::v0::Parameter>& parameter) -> std::optional<size_t> {
        const auto& partial_shape = parameter->get_partial_shape();
        if (partial_shape.rank().is_dynamic()) {
            return std::nullopt;
        }
        std::optional<size_t> dynamic_axis;
        for (size_t i = 0; i < partial_shape.size(); ++i) {
            if (partial_shape[i].is_dynamic()) {
                OPENVINO_ASSERT(!dynamic_axis.has_value(),
                                "GQA parameter '",
                                parameter->get_friendly_name(),
                                "' has more than one dynamic dimension; can't resolve its max_seq_len axis");
                dynamic_axis = i;
            }
        }
        return dynamic_axis;
    };

    for (const auto& parameter : model->get_parameters()) {
        const auto& name = parameter->get_friendly_name();
        if (!ov::npuw::util::contains_ignore_case(name, "past_key") &&
            !ov::npuw::util::contains_ignore_case(name, "past_value")) {
            continue;
        }
        if (auto axis = resolve_dynamic_axis(parameter)) {
            result.emplace(name, *axis);
        }
    }

    // The attention bias/mask shares the KV-cache's max_seq_len dimension but isn't named
    // consistently across exporters, so it's located via the GQA op's ATTENTION_BIAS input
    // instead of by Parameter name.
    using ov::op::internal::GroupQueryAttention;
    using ov::op::internal::GroupQueryAttentionInputs;
    for (const auto& node : model->get_ordered_ops()) {
        auto gqa = ov::as_type_ptr<GroupQueryAttention>(node);
        if (!gqa || gqa->get_input_size() <= static_cast<size_t>(GroupQueryAttentionInputs::ATTENTION_BIAS)) {
            continue;
        }
        auto bias_parameter = ov::as_type_ptr<ov::op::v0::Parameter>(
            gqa->input_value(static_cast<size_t>(GroupQueryAttentionInputs::ATTENTION_BIAS)).get_node_shared_ptr());
        if (!bias_parameter) {
            continue;  // not fed directly by a Parameter; nothing we can reshape here
        }
        if (auto axis = resolve_dynamic_axis(bias_parameter)) {
            // GroupQueryAttentionDecomposition consumes this input as
            // [1, num_heads, curr_query_len, max_kv_len] (see its has_bias branch) -- the
            // KV-cache-capacity dimension is always the LAST axis (max_kv_len), never the
            // query-length axis. A naive "whichever single axis is dynamic" pick is wrong
            // if a model instead leaves the query-length axis dynamic (e.g. a
            // speculative-decode model with query_len > 1): pinning that axis to the full
            // KV-cache capacity would silently corrupt the bias shape instead of raising a
            // clear error.
            const auto last_axis = bias_parameter->get_partial_shape().size() - 1;
            OPENVINO_ASSERT(*axis == last_axis,
                            "GQA attention-bias parameter '",
                            bias_parameter->get_friendly_name(),
                            "' has its dynamic dimension at axis ",
                            *axis,
                            ", but GroupQueryAttentionDecomposition always reads the KV-cache capacity from "
                            "the last axis (",
                            last_axis,
                            ") of [1, num_heads, curr, max_kv]. Refusing to pin a non-capacity axis to the "
                            "static context length.");
            result.emplace(bias_parameter->get_friendly_name(), *axis);
        }
    }

    return result;
}

// Models with `sliding_window_cache=true` physically evict/repack their KV-cache ring
// buffer using its own runtime capacity (see GroupQueryAttentionDecomposition) -- this is
// fundamentally different from `local_window_size` alone, which only restricts the
// attention mask while the buffer keeps growing. Forcing such a KV-cache Parameter up to
// the model's full/global max_seq_len would silently corrupt the eviction math once total
// tokens exceed the local/sliding layer's real window (per review comment
// ankitm3k#4230466592 on PR openvinotoolkit/openvino#38304): such a layer needs its OWN,
// smaller static capacity, derived from its `local_window_size` rather than the model-wide
// max_seq_len.
//
// windowed_cache_end() (GroupQueryAttentionDecomposition) only requires capacity >=
// local_window_size; a tight capacity == local_window_size would still be correct but
// would roll the ring buffer on almost every appended token. Round up to the NPU's
// preferred 16-token KV granularity plus one extra alignment block of slack so evictions
// are amortized instead.
std::optional<size_t> sliding_window_target_capacity(const ov::op::internal::GroupQueryAttention& gqa) {
    if (!gqa.get_sliding_window_cache()) {
        return std::nullopt;
    }
    constexpr int64_t kAlign = 16;
    constexpr int64_t kHeadroom = 16;
    const int64_t window = gqa.get_local_window_size();
    OPENVINO_ASSERT(window >= 1,
                    "GQA op '",
                    gqa.get_friendly_name(),
                    "' has sliding_window_cache=true but an invalid local_window_size=",
                    window);
    const int64_t aligned = ((window + kAlign - 1) / kAlign) * kAlign;
    return static_cast<size_t>(aligned + kHeadroom);
}

// Finds the GQA op that directly reads `parameter` as its PAST_KEY, PAST_VALUE, or
// ATTENTION_BIAS input, or nullptr if none does -- used to look up a dynamic KV-cache/bias
// Parameter's own target capacity instead of blindly applying the model-wide max_seq_len
// to every layer regardless of whether it's a local/sliding-window one.
std::shared_ptr<ov::op::internal::GroupQueryAttention> find_gqa_consuming_parameter(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<ov::op::v0::Parameter>& parameter) {
    using ov::op::internal::GroupQueryAttention;
    using ov::op::internal::GroupQueryAttentionInputs;
    for (const auto& node : model->get_ordered_ops()) {
        auto gqa = ov::as_type_ptr<GroupQueryAttention>(node);
        if (!gqa) {
            continue;
        }
        for (auto input_idx : {GroupQueryAttentionInputs::PAST_KEY,
                               GroupQueryAttentionInputs::PAST_VALUE,
                               GroupQueryAttentionInputs::ATTENTION_BIAS}) {
            if (gqa->get_input_size() <= static_cast<size_t>(input_idx)) {
                continue;
            }
            if (gqa->input_value(static_cast<size_t>(input_idx)).get_node_shared_ptr() == parameter) {
                return gqa;
            }
        }
    }
    return nullptr;
}

// Maps every GQA op in `model` to a 0-based sequential layer index, in the order each is
// encountered while walking the model's topologically-sorted ops. This is how the model's
// own exporter numbers layers in the "full_attention_layers"/"sliding_window_layers"
// rt_info below (a standard sequential transformer stack), so op-encounter order is used
// as the layer index instead of parsing it out of a Parameter name -- the latter would
// assume a specific, exporter-dependent naming convention that attention-bias/mask
// Parameters (shared across layers, named inconsistently) don't necessarily follow anyway.
std::unordered_map<const ov::Node*, int64_t> index_gqa_layers(const std::shared_ptr<ov::Model>& model) {
    std::unordered_map<const ov::Node*, int64_t> result;
    int64_t index = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::internal::GroupQueryAttention>(node)) {
            result.emplace(node.get(), index++);
        }
    }
    return result;
}

// The authoritative source for which GQA layers use a smaller, host-managed sliding-window
// KV-cache when the GQA ops themselves carry no sliding_window_cache/local_window_size
// attribute at all (this app's actual models never set those -- windowing is instead
// communicated via ONNX metadata_props, preserved by the ONNX frontend as OV
// rt_info["framework"][...]; see GQA dispatch PR review comment ankitm3k#4230466592
// follow-up research). Pinning every dynamic KV-cache/bias Parameter to the model-wide
// max_seq_len regardless of this metadata is what silently collapsed the two distinct
// ONNX dynamic-shape symbols (max_seq_len vs. max_seq_len_sliding) into one.
struct RtInfoSlidingWindowMetadata {
    std::unordered_set<int64_t> sliding_layers;
    size_t sliding_window_size;
};

std::optional<RtInfoSlidingWindowMetadata> find_rt_info_sliding_window_metadata(
    const std::shared_ptr<ov::Model>& model) {
    if (!model->has_rt_info("framework", "sliding_window_layers") ||
        !model->has_rt_info("framework", "sliding_window_size")) {
        return std::nullopt;
    }
    auto sliding_layers =
        ov::npuw::util::parse_layer_index_list(model->get_rt_info<std::string>("framework", "sliding_window_layers"));
    if (sliding_layers.empty()) {
        return std::nullopt;
    }
    // This is a literal contract with the host app's own eviction logic (it allocates and
    // manages exactly this many KV-cache slots), not a value we're free to round up: an
    // exact match is also what lets GQAInferRequest::set_tensor() alias the host's tensor
    // directly instead of falling back to a copy.
    const auto declared_size = std::stoull(model->get_rt_info<std::string>("framework", "sliding_window_size"));
    return RtInfoSlidingWindowMetadata{std::move(sliding_layers), static_cast<size_t>(declared_size)};
}

// Last-resort fallback for models that carry NEITHER rt_info metadata NOR a
// sliding_window_cache=true op attribute -- i.e. no authoritative signal exists for either
// a layer's sliding-window membership or its target capacity. Even so, a real structural
// signal survives purely in the IR: this app's actual exporter gives every full-attention
// layer's GQA op the SAME shared SEQLENS_K/TOTAL_SEQUENCE_LENGTH input (e.g. one
// "past_seq_len_full" Parameter consumed by all of them), and every sliding-window layer's
// GQA op a DIFFERENT shared one (e.g. "past_seq_len_sliding") -- confirmed against the
// real model's actual ONNX graph. Unlike past_key/past_value Parameter names (which are
// already used, per-layer, to find the dynamic axis itself), this input is a true
// group-identity signal: its name doesn't need to be parsed for a layer index, just
// checked for a "sliding" hint, because group membership itself already falls out of
// which GQA ops share the exact same input node.
//
// What the IR can NOT tell us is the actual capacity number: these models do pure
// host-side eviction with no in-graph windowing math, so the real value (e.g. 2048) never
// appears anywhere in the graph. Without rt_info, that number can only come from an
// explicit, caller-supplied default (m_default_sliding_window_capacity) -- if the caller
// hasn't configured one, this fallback does nothing and every layer keeps the model-wide
// max_seq_len, exactly as before this whole investigation (a safe, conservative default).
bool has_sliding_window_seqlens_name(const ov::op::internal::GroupQueryAttention& gqa) {
    using ov::op::internal::GroupQueryAttentionInputs;
    const auto seqlens_input_name = [&](GroupQueryAttentionInputs input) -> std::string {
        if (gqa.get_input_size() <= static_cast<size_t>(input)) {
            return {};
        }
        return gqa.input_value(static_cast<size_t>(input)).get_node()->get_friendly_name();
    };
    return ov::npuw::util::contains_ignore_case(seqlens_input_name(GroupQueryAttentionInputs::SEQLENS_K), "sliding") ||
           ov::npuw::util::contains_ignore_case(seqlens_input_name(GroupQueryAttentionInputs::TOTAL_SEQUENCE_LENGTH),
                                                "sliding");
}

}  // namespace

ov::npuw::ReshapeToStaticGQA::ReshapeToStaticGQA(size_t max_seq_len,
                                                 std::optional<size_t> default_sliding_window_capacity)
    : m_max_seq_len(max_seq_len),
      m_default_sliding_window_capacity(default_sliding_window_capacity) {}

bool ov::npuw::ReshapeToStaticGQA::run_on_model(const std::shared_ptr<ov::Model>& model) {
    m_dynamic_kv_cache_axes = find_dynamic_kv_cache_axes(model);
    OPENVINO_ASSERT(!m_dynamic_kv_cache_axes.empty(),
                    "GQA model has a dynamic max_seq_len but no resolvable KV-cache Parameter was found");

    // Per-Parameter target capacity: the model-wide max_seq_len for ordinary (full/global-
    // attention) layers; a sliding-window layer gets its own, smaller capacity from
    // whichever of these independent mechanisms a model uses to declare it, in priority
    // order:
    //  1. rt_info["framework"]["sliding_window_layers"/"sliding_window_size"], derived from
    //     ONNX metadata_props -- this app's actual models' authoritative mechanism, since
    //     it's an exact-match contract with the host's own eviction logic;
    //  2. a GQA op's own sliding_window_cache=true + local_window_size attributes (the
    //     ONNX-standard, in-graph ring-buffer eviction case) -- see
    //     sliding_window_target_capacity() above;
    //  3. if neither is present, a best-effort fallback (only when the caller has
    //     configured m_default_sliding_window_capacity): detect sliding-window layers
    //     purely from a shared SEQLENS_K/TOTAL_SEQUENCE_LENGTH input name hint -- see
    //     has_sliding_window_seqlens_name() above. This can only ever supply a
    //     caller-configured default capacity, never a value read from the model itself,
    //     since these models carry no in-graph trace of the real number.
    auto rt_info_metadata = find_rt_info_sliding_window_metadata(model);
    auto gqa_layer_index = rt_info_metadata ? index_gqa_layers(model) : std::unordered_map<const ov::Node*, int64_t>{};
    LOG_INFO(
        "ReshapeToStaticGQA: resolving per-layer KV-cache capacity (model-wide max_seq_len="
        << m_max_seq_len << "); rt_info sliding-window metadata " << (rt_info_metadata ? "FOUND" : "NOT found")
        << (rt_info_metadata
                ? (", " + std::to_string(rt_info_metadata->sliding_layers.size()) +
                   " layer(s) declared sliding_window_size=" + std::to_string(rt_info_metadata->sliding_window_size))
                : std::string())
        << "; seqlens-name fallback default "
        << (m_default_sliding_window_capacity ? ("configured=" + std::to_string(*m_default_sliding_window_capacity))
                                              : std::string("not configured")));

    // Counters for the end-of-pass summary below -- lets a reader confirm, at a glance,
    // which of the three mechanisms (if any) actually fired for this model, rather than
    // having to tally up the per-Parameter LOG_INFO lines by hand.
    size_t resolved_via_rt_info = 0;
    size_t resolved_via_op_attribute = 0;
    size_t resolved_via_seqlens_name_fallback = 0;
    size_t resolved_via_default = 0;

    std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;
    for (const auto& kv : m_dynamic_kv_cache_axes) {
        // NB: structured bindings can't be captured by lambdas pre-C++20, so use
        // plain named locals for 'name'/'axis' instead of a [name, axis] binding here.
        const auto& name = kv.first;
        const auto axis = kv.second;
        const auto& params = model->get_parameters();
        auto it = std::find_if(params.begin(), params.end(), [&](const auto& parameter) {
            return parameter->get_friendly_name() == name;
        });
        OPENVINO_ASSERT(it != params.end(), "KV-cache parameter '", name, "' not found in the model");

        size_t target_capacity = m_max_seq_len;
        auto gqa = find_gqa_consuming_parameter(model, *it);
        bool resolved = false;
        if (gqa && rt_info_metadata) {
            auto layer_it = gqa_layer_index.find(gqa.get());
            if (layer_it != gqa_layer_index.end() && rt_info_metadata->sliding_layers.count(layer_it->second)) {
                target_capacity = std::min(rt_info_metadata->sliding_window_size, m_max_seq_len);
                resolved = true;
                resolved_via_rt_info++;
                LOG_INFO("GQA op '" << gqa->get_friendly_name() << "' (layer " << layer_it->second
                                    << ") is listed in rt_info sliding_window_layers: pinning '" << name
                                    << "' to the declared sliding_window_size of " << target_capacity
                                    << " tokens instead of the model-wide " << m_max_seq_len);
            }
        }
        if (!resolved && gqa) {
            if (auto window_capacity = sliding_window_target_capacity(*gqa)) {
                target_capacity = std::min(*window_capacity, m_max_seq_len);
                resolved = true;  // also skip the seqlens-name fallback below
                resolved_via_op_attribute++;
                LOG_INFO("GQA op '" << gqa->get_friendly_name() << "' has sliding_window_cache=true: pinning '" << name
                                    << "' to its own local capacity of " << target_capacity
                                    << " tokens instead of the model-wide " << m_max_seq_len);
            }
        }
        if (!resolved && gqa && m_default_sliding_window_capacity && has_sliding_window_seqlens_name(*gqa)) {
            target_capacity = std::min(*m_default_sliding_window_capacity, m_max_seq_len);
            resolved = true;
            resolved_via_seqlens_name_fallback++;
            LOG_INFO("GQA op '" << gqa->get_friendly_name()
                                << "' has no rt_info/op-attribute sliding-window signal, but its SEQLENS_K/"
                                   "TOTAL_SEQUENCE_LENGTH input name hints a sliding-window group: pinning '"
                                << name << "' to the configured default capacity of " << target_capacity
                                << " tokens instead of the model-wide " << m_max_seq_len
                                << " (best-effort fallback -- no authoritative source for this model)");
        }
        if (!resolved) {
            resolved_via_default++;
            LOG_INFO("'" << name << "' has no rt_info/op-attribute/seqlens-name sliding-window signal"
                         << (gqa ? "" : " (no owning GQA op found)") << ": pinning it to the model-wide max_seq_len of "
                         << target_capacity << " tokens");
        }

        auto new_shape = (*it)->get_partial_shape();
        new_shape[axis] = ov::Dimension(static_cast<int64_t>(target_capacity));
        new_shapes[(*it)->output(0)] = new_shape;
    }
    LOG_INFO("ReshapeToStaticGQA: resolved " << m_dynamic_kv_cache_axes.size() << " dynamic KV-cache/bias axis(es) -- "
                                             << "rt_info: " << resolved_via_rt_info
                                             << ", op_attribute: " << resolved_via_op_attribute
                                             << ", seqlens_name_fallback: " << resolved_via_seqlens_name_fallback
                                             << ", default(max_seq_len): " << resolved_via_default);
    model->reshape(new_shapes);

    for (const auto& kv : m_dynamic_kv_cache_axes) {
        const auto& name = kv.first;
        const auto& params = model->get_parameters();
        auto it = std::find_if(params.begin(), params.end(), [&](const auto& parameter) {
            return parameter->get_friendly_name() == name;
        });
        OPENVINO_ASSERT(it != params.end() && (*it)->get_partial_shape().is_static(),
                        "Reshaping the GQA KV-cache parameter '",
                        name,
                        "' to a static capacity did not make it fully static");
    }
    LOG_INFO("Reshaped dynamic GQA KV-cache to a static capacity (model-wide max of " << m_max_seq_len << " tokens)");
    return true;
}
