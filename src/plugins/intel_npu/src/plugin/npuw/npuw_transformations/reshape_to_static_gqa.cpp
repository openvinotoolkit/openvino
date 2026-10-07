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
            result.emplace(bias_parameter->get_friendly_name(), *axis);
        }
    }

    return result;
}

}  // namespace

ov::npuw::ReshapeToStaticGQA::ReshapeToStaticGQA(size_t max_seq_len) : m_max_seq_len(max_seq_len) {}

bool ov::npuw::ReshapeToStaticGQA::run_on_model(const std::shared_ptr<ov::Model>& model) {
    m_dynamic_kv_cache_axes = find_dynamic_kv_cache_axes(model);
    OPENVINO_ASSERT(!m_dynamic_kv_cache_axes.empty(),
                    "GQA model has a dynamic max_seq_len but no resolvable KV-cache Parameter was found");

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
