// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_to_static_gqa.hpp"

#include <algorithm>
#include <map>
#include <optional>

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
// attention mask while the buffer keeps growing. Forcing such a KV-cache Parameter to a
// capacity that doesn't match whatever buffer the caller actually uses at runtime would
// silently corrupt the eviction math (diverges from the reference once total tokens
// exceed the caller's real buffer + 1, per review comment ankitm3k#4230466592 on PR
// openvinotoolkit/openvino#38304). NPUW has no mechanism yet to give such a KV-cache a
// separate, caller-matched capacity (the reviewer's other suggested fix), so reject this
// case outright rather than risk a silent accuracy bug.
void reject_sliding_window_cache_with_dynamic_kv(const std::shared_ptr<ov::Model>& model,
                                                 const std::unordered_map<std::string, size_t>& dynamic_kv_cache_axes) {
    using ov::op::internal::GroupQueryAttention;
    using ov::op::internal::GroupQueryAttentionInputs;
    for (const auto& node : model->get_ordered_ops()) {
        auto gqa = ov::as_type_ptr<GroupQueryAttention>(node);
        if (!gqa || !gqa->get_sliding_window_cache()) {
            continue;
        }
        for (auto kv_input : {GroupQueryAttentionInputs::PAST_KEY, GroupQueryAttentionInputs::PAST_VALUE}) {
            if (gqa->get_input_size() <= static_cast<size_t>(kv_input)) {
                continue;
            }
            auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(
                gqa->input_value(static_cast<size_t>(kv_input)).get_node_shared_ptr());
            if (!parameter) {
                continue;
            }
            OPENVINO_ASSERT(dynamic_kv_cache_axes.count(parameter->get_friendly_name()) == 0,
                            "GQA op '",
                            gqa->get_friendly_name(),
                            "' has sliding_window_cache=true, whose physical eviction/repacking depends on its "
                            "own runtime KV-cache capacity. Forcing its dynamic KV-cache Parameter '",
                            parameter->get_friendly_name(),
                            "' to a single, model-wide static capacity would silently diverge from the caller's "
                            "actual window buffer. This configuration is not supported by GQA dispatch yet.");
        }
    }
}

}  // namespace

ov::npuw::ReshapeToStaticGQA::ReshapeToStaticGQA(size_t max_seq_len) : m_max_seq_len(max_seq_len) {}

bool ov::npuw::ReshapeToStaticGQA::run_on_model(const std::shared_ptr<ov::Model>& model) {
    m_dynamic_kv_cache_axes = find_dynamic_kv_cache_axes(model);
    OPENVINO_ASSERT(!m_dynamic_kv_cache_axes.empty(),
                    "GQA model has a dynamic max_seq_len but no resolvable KV-cache Parameter was found");
    reject_sliding_window_cache_with_dynamic_kv(model, m_dynamic_kv_cache_axes);

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
        auto new_shape = (*it)->get_partial_shape();
        new_shape[axis] = ov::Dimension(static_cast<int64_t>(m_max_seq_len));
        new_shapes[(*it)->output(0)] = new_shape;
    }
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
                        "' to a static capacity of ",
                        m_max_seq_len,
                        " did not make it fully static");
    }
    LOG_INFO("Reshaped dynamic GQA KV-cache to a static capacity of " << m_max_seq_len << " tokens");
    return true;
}
