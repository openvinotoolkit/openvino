// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split_kvcache_into_blocks.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

#include "logging.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
namespace npuw {
namespace pass {

namespace {

static constexpr const char* past_key_values = "past_key_values";

/**
 * @brief Structure to hold parameter and its associated Concat node
 */
struct KVCacheTransformInfo {
    std::shared_ptr<ov::op::v0::Parameter> param;
    std::shared_ptr<ov::op::v0::Concat> concat;
    ov::Output<ov::Node> present_kv_input;
    std::shared_ptr<ov::Node> convert_node;  // Convert node between param and concat (if exists)
};

/**
 * @brief Check if parameter is K (key) tensor
 * Matches: "past_key.0", "past_key_values.0.key", etc.
 */
bool is_key_parameter(const std::shared_ptr<ov::op::v0::Parameter>& param) {
    std::string name = param->get_friendly_name();
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    // Match "past_key_values.X.key" pattern
    if (name.find(past_key_values) != std::string::npos && name.find(".key") != std::string::npos) {
        return true;
    }
    // Match "past_key.X" pattern (but not "past_key_values")
    if (name.find("past_key") != std::string::npos && name.find(past_key_values) == std::string::npos) {
        return true;
    }
    return false;
}

/**
 * @brief Check if parameter is V (value) tensor
 * Matches: "past_value.0", "past_key_values.0.value", etc.
 */
bool is_value_parameter(const std::shared_ptr<ov::op::v0::Parameter>& param) {
    std::string name = param->get_friendly_name();
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    // Match "past_key_values.X.value" pattern
    if (name.find(past_key_values) != std::string::npos && name.find(".value") != std::string::npos) {
        return true;
    }
    // Match "past_value.X" pattern
    if (name.find("past_value") != std::string::npos) {
        return true;
    }
    return false;
}

/**
 * @brief Check if parameter name indicates KV cache
 *
 * KV cache parameters typically have names like:
 * - "past_key.0", "past_value.0" (standard naming)
 * - "past_key_values.0.key", "past_key_values.0.value" (alternative naming)
 */
bool is_kv_cache_parameter(const std::shared_ptr<ov::op::v0::Parameter>& param) {
    std::string name = param->get_friendly_name();
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    return (name.find("past_key") != std::string::npos || name.find("past_value") != std::string::npos ||
            name.find(past_key_values) != std::string::npos);
}

}  // namespace

SplitKVCacheIntoBlocks::SplitKVCacheIntoBlocks(uint32_t block_size, bool v_transposed)
    : m_block_size(block_size),
      m_v_transposed(v_transposed) {}

bool SplitKVCacheIntoBlocks::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool model_changed = false;

    // Collect all KV cache parameters and their associated Concat nodes
    std::vector<KVCacheTransformInfo> params_to_transform;
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> params_to_remove;

    for (const auto& param : model->get_parameters()) {
        if (!is_kv_cache_parameter(param)) {
            continue;
        }

        // Find the Concat that uses this parameter
        // Pattern: Parameter -> Concat or Parameter -> Convert -> Concat
        std::shared_ptr<ov::op::v0::Concat> concat = nullptr;
        std::shared_ptr<ov::Node> convert_node = nullptr;

        for (const auto& output : param->outputs()) {
            for (const auto& input : output.get_target_inputs()) {
                auto node = input.get_node()->shared_from_this();

                // Check if direct consumer is Concat
                if (auto concat_node = ov::as_type_ptr<ov::op::v0::Concat>(node)) {
                    concat = concat_node;
                    break;
                }

                // Check if consumer is Convert, then check Convert's consumers for Concat
                if (node->get_type_name() == std::string("Convert")) {
                    for (const auto& convert_output : node->outputs()) {
                        for (const auto& convert_target : convert_output.get_target_inputs()) {
                            auto next_node = convert_target.get_node()->shared_from_this();
                            if (auto concat_node = ov::as_type_ptr<ov::op::v0::Concat>(next_node)) {
                                concat = concat_node;
                                convert_node = node;  // Record the Convert node
                                break;
                            }
                        }
                        if (concat)
                            break;
                    }
                }
            }
            if (concat) {
                break;
            }
        }

        if (!concat) {
            continue;
        }

        // Find present_key/value input (the other input to concat)
        ov::Output<ov::Node> present_kv_input;
        bool found_present_kv = false;
        for (size_t i = 0; i < concat->get_input_size(); ++i) {
            const auto& input = concat->input(i);
            auto input_node = input.get_source_output().get_node_shared_ptr();

            // Check if input comes from our parameter (directly or through Convert)
            bool is_from_param = false;
            if (input_node == param) {
                is_from_param = true;
            } else if (input_node->get_type_name() == std::string("Convert")) {
                // Check if Convert's input is our parameter
                auto convert_input = input_node->input(0).get_source_output().get_node_shared_ptr();
                if (convert_input == param) {
                    is_from_param = true;
                }
            }

            if (!is_from_param) {
                present_kv_input = input.get_source_output();
                found_present_kv = true;
                break;
            }
        }

        if (found_present_kv) {
            params_to_transform.push_back({param, concat, present_kv_input, convert_node});
        }
    }

    // Transform each KV cache parameter
    for (auto& info : params_to_transform) {
        auto& param = info.param;
        auto& concat = info.concat;
        auto& present_kv_input = info.present_kv_input;
        auto& convert_node = info.convert_node;

        // Get original KV cache shape
        const auto& orig_shape = param->get_partial_shape();
        if (orig_shape.rank().is_dynamic() || orig_shape.rank().get_length() != 4) {
            LOG_WARN("SplitKVCacheIntoBlocks: Skipping parameter "
                     << param->get_friendly_name() << " - invalid rank (expected 4D, got " << orig_shape << ")");
            continue;
        }

        // Extract dimensions based on parameter type
        int64_t batch = orig_shape[0].is_static() ? orig_shape[0].get_length() : 1;
        int64_t num_heads = orig_shape[1].is_static() ? orig_shape[1].get_length() : -1;
        int64_t head_dim = -1;
        int64_t seq_len = -1;

        // For Key: always [B, H, S, D]
        // For Value transposed: [B, H, D, S]
        // For Value not transposed: [B, H, S, D]
        if (is_key_parameter(param)) {
            seq_len = orig_shape[2].is_static() ? orig_shape[2].get_length() : -1;
            head_dim = orig_shape[3].is_static() ? orig_shape[3].get_length() : -1;
        } else if (is_value_parameter(param) && m_v_transposed) {
            head_dim = orig_shape[2].is_static() ? orig_shape[2].get_length() : -1;
            seq_len = orig_shape[3].is_static() ? orig_shape[3].get_length() : -1;
        } else {
            seq_len = orig_shape[2].is_static() ? orig_shape[2].get_length() : -1;
            head_dim = orig_shape[3].is_static() ? orig_shape[3].get_length() : -1;
        }

        if (num_heads <= 0 || head_dim <= 0 || seq_len <= 0) {
            LOG_WARN("SplitKVCacheIntoBlocks: Skipping parameter " << param->get_friendly_name()
                                                                   << " - invalid or dynamic dimensions");
            continue;
        }

        LOG_INFO("SplitKVCacheIntoBlocks: Transforming " << param->get_friendly_name() << " from shape " << orig_shape);

        // Determine concat axis based on parameter type
        int64_t concat_axis = -1;
        if (is_key_parameter(param)) {
            concat_axis = 2;  // K: [B, H, S, D] concat on S
        } else if (is_value_parameter(param)) {
            concat_axis = m_v_transposed
                              ? 3
                              : 2;  // V transposed: [B, H, D, S] concat on S; not transposed: [B, H, S, D] concat on S
        } else {
            concat_axis = concat->get_axis();
        }

        // Calculate number of blocks
        uint32_t num_full_blocks = static_cast<uint32_t>(seq_len) / m_block_size;
        uint32_t tail_size = static_cast<uint32_t>(seq_len) % m_block_size;
        uint32_t total_blocks = num_full_blocks + (tail_size > 0 ? 1 : 0);

        LOG_INFO("SplitKVCacheIntoBlocks: block_size=" << m_block_size << ", num_full_blocks=" << num_full_blocks
                                                       << ", tail_size=" << tail_size
                                                       << ", total_blocks=" << total_blocks);

        // Create block parameters
        ov::OutputVector block_params;
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> new_params;
        block_params.reserve(total_blocks);
        new_params.reserve(total_blocks);

        // Create full blocks
        for (uint32_t i = 0; i < num_full_blocks; ++i) {
            ov::Shape block_shape;
            if (concat_axis == 2) {
                block_shape = {static_cast<size_t>(batch),
                               static_cast<size_t>(num_heads),
                               m_block_size,
                               static_cast<size_t>(head_dim)};
            } else {
                block_shape = {static_cast<size_t>(batch),
                               static_cast<size_t>(num_heads),
                               static_cast<size_t>(head_dim),
                               m_block_size};
            }
            auto block_param = std::make_shared<ov::op::v0::Parameter>(param->get_element_type(), block_shape);
            std::string block_name = param->get_friendly_name() + "_block_" + std::to_string(i);
            block_param->set_friendly_name(block_name);
            // CRITICAL: Set tensor names so compiled model ports have names
            block_param->output(0).set_names({block_name});
            block_params.push_back(block_param);
            new_params.push_back(block_param);
        }

        // Create tail block if needed
        if (tail_size > 0) {
            ov::Shape tail_shape;
            if (concat_axis == 2) {
                tail_shape = {static_cast<size_t>(batch),
                              static_cast<size_t>(num_heads),
                              tail_size,
                              static_cast<size_t>(head_dim)};
            } else {
                tail_shape = {static_cast<size_t>(batch),
                              static_cast<size_t>(num_heads),
                              static_cast<size_t>(head_dim),
                              tail_size};
            }
            auto tail_param = std::make_shared<ov::op::v0::Parameter>(param->get_element_type(), tail_shape);
            std::string tail_name = param->get_friendly_name() + "_block_tail";
            tail_param->set_friendly_name(tail_name);
            // CRITICAL: Set tensor names so compiled model ports have names
            tail_param->output(0).set_names({tail_name});
            block_params.push_back(tail_param);
            new_params.push_back(tail_param);
        }

        // Add new block parameters to model
        model->add_parameters(new_params);

        // If original path had Convert, create Convert for each block parameter
        ov::OutputVector inputs_for_concat;
        inputs_for_concat.reserve(block_params.size() + 1);

        if (convert_node) {
            // Get the target type from original Convert node
            auto target_type = convert_node->get_output_element_type(0);

            // Create Convert for each block parameter
            for (const auto& block_param : block_params) {
                auto convert = std::make_shared<ov::op::v0::Convert>(block_param, target_type);
                convert->set_friendly_name(block_param.get_node()->get_friendly_name() + "_convert");
                inputs_for_concat.push_back(convert);
            }
        } else {
            // No Convert needed, use block parameters directly
            inputs_for_concat = block_params;
        }

        // Add present_kv_input to concat inputs
        inputs_for_concat.push_back(present_kv_input);

        // Create new Concat
        auto new_concat = std::make_shared<ov::op::v0::Concat>(inputs_for_concat, concat_axis);
        new_concat->set_friendly_name(concat->get_friendly_name());

        LOG_DEBUG("SplitKVCacheIntoBlocks: Created new concat with output shape "
                  << new_concat->get_output_partial_shape(0));

        // Copy runtime info
        ov::NodeVector new_nodes{new_concat};
        for (const auto& p : new_params) {
            new_nodes.push_back(p);
        }
        copy_runtime_info({param, concat}, new_nodes);

        // Replace concat's output with new_concat
        concat->output(0).replace(new_concat->output(0));

        // Mark old parameter for removal
        params_to_remove.push_back(param);

        LOG_INFO("SplitKVCacheIntoBlocks: Successfully replaced " << param->get_friendly_name() << " with "
                                                                  << total_blocks << " block parameters");

        model_changed = true;
    }

    // Remove all old parameters after transformations are complete
    for (const auto& param : params_to_remove) {
        model->remove_parameter(param);
    }

    return model_changed;
}

}  // namespace pass
}  // namespace npuw
}  // namespace ov
