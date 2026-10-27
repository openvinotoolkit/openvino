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

// Scans `model`'s past_key/past_value Parameters for a dynamic dimension, keyed by
// Parameter name. More than one dynamic dim is ambiguous and not resolved.
std::unordered_map<std::string, size_t> find_dynamic_kv_cache_axes(const std::shared_ptr<ov::Model>& model) {
    std::unordered_map<std::string, size_t> result;

    const auto resolve_dynamic_axis =
        [](const std::shared_ptr<ov::op::v0::Parameter>& parameter) -> std::optional<size_t> {
        const auto& partial_shape = parameter->get_partial_shape();
        if (auto axis = ov::npuw::util::find_single_dynamic_axis(partial_shape)) {
            return axis;
        }
        if (partial_shape.rank().is_static()) {
            const auto dynamic_count = std::count_if(partial_shape.begin(), partial_shape.end(), [](const auto& dim) {
                return dim.is_dynamic();
            });
            OPENVINO_ASSERT(dynamic_count <= 1,
                            "GQA parameter '",
                            parameter->get_friendly_name(),
                            "' has more than one dynamic dimension; can't resolve its max_seq_len axis");
        }
        return std::nullopt;
    };

    // The capacity dim is axis 2 of [batch, kv_heads, capacity, head_size], or axis 3
    // under --transpose_v. Axis 0/1 are never the capacity dimension.
    for (const auto& parameter : model->get_parameters()) {
        const auto& name = parameter->get_friendly_name();
        if (!ov::npuw::util::contains_ignore_case(name, "past_key") &&
            !ov::npuw::util::contains_ignore_case(name, "past_value")) {
            continue;
        }
        if (auto axis = resolve_dynamic_axis(parameter)) {
            const auto rank = parameter->get_partial_shape().size();
            OPENVINO_ASSERT(ov::npuw::util::is_supported_kv_cache_capacity_axis(*axis, rank),
                            "GQA KV-cache parameter '",
                            name,
                            "' has its dynamic dimension at axis ",
                            *axis,
                            " of a rank-",
                            rank,
                            " shape, but GroupQueryAttentionDecomposition only ever reads/writes the "
                            "KV-cache capacity at axis 2 ([batch, kv_heads, capacity, head_size]) or axis 3 "
                            "(--transpose_v layout). Refusing to pin a non-capacity axis to the static "
                            "context length.");
            result.emplace(name, *axis);
        }
    }

    // The attention bias/mask shares max_seq_len but isn't named consistently; locate it
    // via the GQA op's ATTENTION_BIAS input instead.
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
            // The capacity dim is always the LAST axis ([1, heads, curr, max_kv]), never
            // the query-length axis -- important for speculative-decode (query_len > 1).
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

// `sliding_window_cache=true` layers physically evict/repack their KV-cache ring buffer,
// so they need their own, smaller static capacity derived from `local_window_size`
// instead of the model-wide max_seq_len.
//
// Round local_window_size up to the NPU's 16-token KV granularity plus one headroom
// block, so evictions are amortized rather than rolling the buffer on every token.
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

// Finds the GQA op that directly reads `parameter` as PAST_KEY/PAST_VALUE/ATTENTION_BIAS,
// used to resolve a Parameter's own target capacity rather than the model-wide max_seq_len.
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

// Maps each GQA op to a 0-based sequential layer index (op-encounter order), matching
// how rt_info's "full_attention_layers"/"sliding_window_layers" number layers.
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

// Authoritative source for sliding-window layers when GQA ops carry no
// sliding_window_cache/local_window_size attribute: ONNX metadata_props preserved as
// OV rt_info["framework"][...].
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
    // Exact match with the host's own eviction logic; lets GQAInferRequest::set_tensor()
    // alias the host's tensor directly instead of copying.
    const auto declared_size = std::stoull(model->get_rt_info<std::string>("framework", "sliding_window_size"));
    return RtInfoSlidingWindowMetadata{std::move(sliding_layers), static_cast<size_t>(declared_size)};
}

// Last-resort fallback for models with neither rt_info metadata nor a
// sliding_window_cache=true attribute: full-attention layers share one
// SEQLENS_K/TOTAL_SEQUENCE_LENGTH Parameter, sliding-window layers share a different one
// hinting "sliding" in its name -- group membership falls out of shared input identity,
// no index parsing needed.
//
// The actual capacity number isn't in the graph at all (pure host-side eviction), so it
// can only come from the caller-supplied m_default_sliding_window_capacity; if unset,
// every layer just keeps the model-wide max_seq_len.
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

    // Per-Parameter target capacity: model-wide max_seq_len normally; a sliding-window
    // layer gets its own smaller capacity, in priority order: (1) rt_info metadata, (2)
    // sliding_window_cache=true + local_window_size op attributes, (3) best-effort
    // seqlens-name fallback using the caller-configured default.
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

    // End-of-pass summary counters -- which mechanism(s) fired for this model.
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
