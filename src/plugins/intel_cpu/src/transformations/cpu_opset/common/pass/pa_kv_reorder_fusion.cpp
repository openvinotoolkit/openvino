// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_reorder_fusion.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../op/pa_kv_reorder.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"

namespace ov::intel_cpu {
namespace {

struct CacheReorderPath {
    std::shared_ptr<ov::op::v3::ScatterUpdate> scatter;
    std::shared_ptr<ov::op::v8::Gather> gather;
    std::shared_ptr<ov::op::v0::Parameter> cache;
    ov::Output<ov::Node> block_indices;
    ov::Output<ov::Node> block_update_indices;
    bool is_key = false;
    std::string index;
};

bool same_output(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs) {
    return lhs.get_node() == rhs.get_node() && lhs.get_index() == rhs.get_index();
}

std::optional<int64_t> get_axis_value(const std::shared_ptr<ov::op::v0::Constant>& axis_const) {
    if (!axis_const) {
        return std::nullopt;
    }

    const auto axis_values = axis_const->cast_vector<int64_t>();
    if (axis_values.size() != 1) {
        return std::nullopt;
    }

    return axis_values[0];
}

std::optional<std::pair<bool, std::string>> parse_cache_role_and_index(const std::string& node_name) {
    // Match pattern: key_cache.N_clone_for_k_update or value_cache.N_clone_for_v_update
    static const std::regex cache_param_regex(R"((key|value)_cache\.(\d+)_clone_for_[kv]_update)");
    std::smatch cache_match;
    if (std::regex_match(node_name, cache_match, cache_param_regex)) {
        const bool is_key = cache_match[1].str() == "key";
        return std::make_pair(is_key, cache_match[2].str());
    }

    // Match pattern: updated_key_cache_N or updated_value_cache_N
    static const std::regex scatter_name_regex(R"(updated_(key|value)_cache_(\d+))");
    std::smatch scatter_match;
    if (std::regex_match(node_name, scatter_match, scatter_name_regex)) {
        const bool is_key = scatter_match[1].str() == "key";
        return std::make_pair(is_key, scatter_match[2].str());
    }

    return std::nullopt;
}

std::optional<CacheReorderPath> parse_cache_reorder_path(const std::shared_ptr<ov::op::v3::ScatterUpdate>& scatter) {
    if (!scatter || scatter->get_input_size() != 4) {
        return std::nullopt;
    }

    auto gather = ov::as_type_ptr<ov::op::v8::Gather>(scatter->input_value(2).get_node_shared_ptr());
    if (!gather || gather->get_input_size() != 3) {
        return std::nullopt;
    }

    auto cache = ov::as_type_ptr<ov::op::v0::Parameter>(scatter->input_value(0).get_node_shared_ptr());
    if (!cache) {
        return std::nullopt;
    }

    // Verify gather reads from the same cache that scatter updates
    if (!same_output(gather->input_value(0), scatter->input_value(0))) {
        return std::nullopt;
    }

    auto gather_axis_const = ov::as_type_ptr<ov::op::v0::Constant>(gather->input_value(2).get_node_shared_ptr());
    auto scatter_axis_const = ov::as_type_ptr<ov::op::v0::Constant>(scatter->input_value(3).get_node_shared_ptr());

    const auto gather_axis = get_axis_value(gather_axis_const);
    const auto scatter_axis = get_axis_value(scatter_axis_const);
    if (!gather_axis.has_value() || !scatter_axis.has_value() || gather_axis.value() != scatter_axis.value()) {
        return std::nullopt;
    }

    // Parse cache role (key/value) and index from node names
    auto role_and_index = parse_cache_role_and_index(cache->get_friendly_name());
    if (!role_and_index.has_value()) {
        role_and_index = parse_cache_role_and_index(scatter->get_friendly_name());
    }

    if (!role_and_index.has_value()) {
        return std::nullopt;
    }

    CacheReorderPath path;
    path.scatter = scatter;
    path.gather = gather;
    path.cache = cache;
    path.block_indices = scatter->input_value(1);
    path.block_update_indices = gather->input_value(1);
    path.is_key = role_and_index->first;
    path.index = role_and_index->second;
    return path;
}

std::shared_ptr<ov::op::v0::Parameter> get_parameter_by_name(const std::shared_ptr<ov::Model>& m,
                                                             const std::string& parameter_name) {
    for (const auto& parameter : m->get_parameters()) {
        if (parameter && parameter->get_friendly_name() == parameter_name) {
            return parameter;
        }
    }

    return nullptr;
}

std::vector<std::shared_ptr<ov::op::v0::Concat>> get_joint_concat_consumers(
    const ov::Output<ov::Node>& key_scatter_output,
    const ov::Output<ov::Node>& value_scatter_output) {
    std::vector<std::shared_ptr<ov::op::v0::Concat>> concats;
    for (const auto& target_input : key_scatter_output.get_target_inputs()) {
        auto concat = ov::as_type_ptr<ov::op::v0::Concat>(target_input.get_node()->shared_from_this());
        if (!concat) {
            continue;
        }

        bool has_key_input = false;
        bool has_value_input = false;
        for (const auto& input : concat->input_values()) {
            has_key_input = has_key_input || same_output(input, key_scatter_output);
            has_value_input = has_value_input || same_output(input, value_scatter_output);
        }

        if (!has_key_input || !has_value_input) {
            continue;
        }

        bool has_result_consumer = false;
        for (const auto& concat_target : concat->output(0).get_target_inputs()) {
            if (ov::as_type_ptr<ov::op::v0::Result>(concat_target.get_node()->shared_from_this())) {
                has_result_consumer = true;
                break;
            }
        }

        if (has_result_consumer) {
            concats.push_back(concat);
        }
    }

    return concats;
}

}  // namespace

bool PaKVReorderFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    auto block_indices_begins = get_parameter_by_name(m, "block_indices_begins");
    auto block_update_indices_begins = get_parameter_by_name(m, "block_update_indices_begins");
    if (!block_indices_begins || !block_update_indices_begins) {
        return false;
    }

    std::unordered_map<std::string, CacheReorderPath> key_paths;
    std::unordered_map<std::string, CacheReorderPath> value_paths;

    for (const auto& node : m->get_ordered_ops()) {
        auto scatter = ov::as_type_ptr<ov::op::v3::ScatterUpdate>(node);
        auto path = parse_cache_reorder_path(scatter);
        if (!path.has_value()) {
            continue;
        }

        if (path->is_key) {
            key_paths[path->index] = std::move(path.value());
        } else {
            value_paths[path->index] = std::move(path.value());
        }
    }

    bool rewritten = false;

    for (const auto& [index, key_path] : key_paths) {
        if (value_paths.count(index) == 0) {
            continue;
        }

        const auto& value_path = value_paths.at(index);
        if (!same_output(key_path.block_indices, value_path.block_indices) ||
            !same_output(key_path.block_update_indices, value_path.block_update_indices)) {
            continue;
        }

        const auto concat_consumers =
            get_joint_concat_consumers(key_path.scatter->output(0), value_path.scatter->output(0));
        if (concat_consumers.empty()) {
            continue;
        }

        auto pa_kv_reorder = std::make_shared<ov::intel_cpu::PaKVReorder>(key_path.cache->output(0),
                                                                          value_path.cache->output(0),
                                                                          key_path.block_indices,
                                                                          block_indices_begins->output(0),
                                                                          key_path.block_update_indices,
                                                                          block_update_indices_begins->output(0));

        // Format cache precision: use inference precision if cache is f16 and inference is bf16
        auto format_cache_precision = [](ov::element::Type cache_precision, ov::element::Type infer_precision) {
            return cache_precision == ov::element::f16 && infer_precision == ov::element::bf16 ? infer_precision
                                                                                               : cache_precision;
        };

        const auto key_cache_type = format_cache_precision(m_key_cache_precision, m_inference_precision);
        const auto value_cache_type = format_cache_precision(m_value_cache_precision, m_inference_precision);

        // Update key/value cache precision to match main paged attention configuration
        key_path.cache->set_element_type(key_cache_type);
        value_path.cache->set_element_type(value_cache_type);

        // Add runtime info for CPU-specific optimizations
        constexpr const char* rt_is_key_by_channel = "pa_kv_reorder.is_key_by_channel";
        constexpr const char* rt_key_dim_order = "pa_kv_reorder.key_cache_dim_order";
        constexpr const char* rt_value_dim_order = "pa_kv_reorder.value_cache_dim_order";

        const bool is_key_by_channel = m_key_cache_quant_by_channel;
        pa_kv_reorder->get_rt_info()[rt_is_key_by_channel] = is_key_by_channel;
        pa_kv_reorder->get_rt_info()[rt_key_dim_order] = m_key_cache_dim_order;
        pa_kv_reorder->get_rt_info()[rt_value_dim_order] = m_value_cache_dim_order;

        ov::copy_runtime_info(key_path.cache, pa_kv_reorder);
        pa_kv_reorder->set_friendly_name("pa_kv_reorder_" + index);
        ov::NodeVector runtime_info_nodes{key_path.scatter, key_path.gather, value_path.scatter, value_path.gather};
        for (const auto& concat : concat_consumers) {
            runtime_info_nodes.push_back(concat);
        }
        ov::copy_runtime_info(runtime_info_nodes, pa_kv_reorder);

        pa_kv_reorder->output(0).get_tensor().set_names(concat_consumers.front()->output(0).get_tensor().get_names());

        for (const auto& concat : concat_consumers) {
            for (const auto& target_input : concat->output(0).get_target_inputs()) {
                if (auto result = ov::as_type_ptr<ov::op::v0::Result>(target_input.get_node()->shared_from_this())) {
                    result->set_argument(0, pa_kv_reorder->output(0));
                }
            }
            ov::replace_output_update_name(concat->output(0), pa_kv_reorder->output(0));
        }
        rewritten = true;
    }

    return rewritten;
}

}  // namespace ov::intel_cpu
