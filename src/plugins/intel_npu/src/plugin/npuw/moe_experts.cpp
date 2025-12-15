// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "moe_experts.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "logging.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/validate.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {
namespace function {

std::optional<MoEValidationResult> validate_and_setup_moe_expert(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("Validating MoE expert model...");
    LOG_BLOCK();

    MoEValidationResult result;

    // Helper: Trace back from node to Parameter
    auto trace_to_parameter = [&](std::shared_ptr<ov::Node> node) -> std::shared_ptr<ov::op::v0::Parameter> {
        std::function<std::shared_ptr<ov::op::v0::Parameter>(std::shared_ptr<ov::Node>)> trace;
        trace = [&](std::shared_ptr<ov::Node> n) -> std::shared_ptr<ov::op::v0::Parameter> {
            if (auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(n)) {
                return param;
            }
            if (n->get_input_size() == 1) {
                return trace(n->input_value(0).get_node_shared_ptr());
            }
            return nullptr;
        };
        return trace(node);
    };

    // Helper: Get parameter index
    auto get_param_index = [&](const std::shared_ptr<ov::op::v0::Parameter>& param) -> std::optional<size_t> {
        const auto& params = model->get_parameters();
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i] == param) {
                return i;
            }
        }
        return std::nullopt;
    };

    // Step 1: Find Tile operation and determine number of experts
    for (const auto& node : model->get_ordered_ops()) {
        if (auto tile = std::dynamic_pointer_cast<ov::op::v0::Tile>(node)) {
            auto repeats_input = tile->input_value(1);
            if (auto repeats_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(repeats_input.get_node_shared_ptr())) {
                auto repeats_data = repeats_const->cast_vector<int64_t>();
                if (!repeats_data.empty() && repeats_data[0] > 1) {
                    result.num_experts = static_cast<size_t>(repeats_data[0]);
                    result.tile_node = tile;
                    result.input_node = tile->input_value(0).get_node_shared_ptr();
                    LOG_DEBUG("Found Tile with " << result.num_experts << " experts");
                    break;
                }
            }
        }
    }

    if (!result.tile_node || result.num_experts == 0) {
        LOG_WARN("Could not find valid Tile operation");
        return std::nullopt;
    }

    // Step 2: Extract shape information
    auto tile_output = result.tile_node->output(0);
    auto tile_output_shape = tile_output.get_shape();
    if (tile_output_shape.empty() || tile_output_shape[0] % result.num_experts != 0) {
        LOG_WARN("Invalid Tile output shape");
        return std::nullopt;
    }
    result.expert_hidden_dim = tile_output_shape[0] / result.num_experts;
    result.input_batch_size = result.tile_node->input_value(0).get_shape()[0];
    LOG_DEBUG("Expert hidden dim: " << result.expert_hidden_dim << ", input batch: " << result.input_batch_size);

    // Step 3: Find MatMul consumer and capture attention parameter
    for (const auto& target_input : tile_output.get_target_inputs()) {
        auto consumer = target_input.get_node()->shared_from_this();

        if (auto reshape_op = std::dynamic_pointer_cast<ov::op::v1::Reshape>(consumer)) {
            LOG_DEBUG("Found Reshape after Tile");
            for (const auto& reshape_target : reshape_op->output(0).get_target_inputs()) {
                if (auto mm =
                        std::dynamic_pointer_cast<ov::op::v0::MatMul>(reshape_target.get_node()->shared_from_this())) {
                    result.matmul_node = mm;
                    result.node_before_matmul = reshape_op->output(0);
                    LOG_DEBUG("Found MatMul after Reshape");

                    // Capture attention parameter from MatMul input[0]
                    if (auto attention_param = trace_to_parameter(mm->input_value(0).get_node_shared_ptr())) {
                        result.attention_param_idx = get_param_index(attention_param);
                        LOG_DEBUG("  Attention param index: " << result.attention_param_idx.value());
                    }
                    break;
                }
            }
        } else if (auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(consumer)) {
            result.matmul_node = mm;
            result.node_before_matmul = tile_output;
            LOG_DEBUG("Found MatMul directly after Tile");

            // Capture attention parameter from MatMul input[0]
            if (auto attention_param = trace_to_parameter(mm->input_value(0).get_node_shared_ptr())) {
                result.attention_param_idx = get_param_index(attention_param);
                LOG_DEBUG("  Attention param index: " << result.attention_param_idx.value());
            }
            break;
        }

        if (result.matmul_node)
            break;
    }

    if (!result.matmul_node) {
        LOG_WARN("Could not find MatMul consumer of Tile");
        return std::nullopt;
    }

    // Step 4: Find output Reshape and ReduceSum nodes, capture router parameter
    for (const auto& result_node : model->get_results()) {
        auto result_input = result_node->input_value(0);
        auto result_input_node = result_input.get_node_shared_ptr();

        // Skip Convert if present
        if (auto convert_node = std::dynamic_pointer_cast<ov::op::v0::Convert>(result_input_node)) {
            result_input = convert_node->input_value(0);
            result_input_node = result_input.get_node_shared_ptr();
        }

        // Find ReduceSum
        if (auto reduce_sum = std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(result_input_node)) {
            result.reduce_sum_node = reduce_sum;
            LOG_DEBUG("Found ReduceSum node");

            // Find Multiply
            auto reduce_input = reduce_sum->input_value(0).get_node_shared_ptr();
            if (auto multiply_node = std::dynamic_pointer_cast<ov::op::v1::Multiply>(reduce_input)) {
                LOG_DEBUG("Found Multiply before ReduceSum");

                auto multiply_input0 = multiply_node->input_value(0).get_node_shared_ptr();
                auto multiply_input1 = multiply_node->input_value(1).get_node_shared_ptr();

                // Find Reshape and Router parameter
                std::shared_ptr<ov::Node> router_input_node;
                if (auto reshape0 = std::dynamic_pointer_cast<ov::op::v1::Reshape>(multiply_input0)) {
                    result.output_reshape_node = reshape0;
                    router_input_node = multiply_input1;
                } else if (auto reshape1 = std::dynamic_pointer_cast<ov::op::v1::Reshape>(multiply_input1)) {
                    result.output_reshape_node = reshape1;
                    router_input_node = multiply_input0;
                }

                // Capture router parameter
                if (router_input_node) {
                    if (auto router_param = trace_to_parameter(router_input_node)) {
                        result.router_param_idx = get_param_index(router_param);
                        LOG_DEBUG("  Router param index: " << result.router_param_idx.value());
                    }
                }
            }
            break;
        }
    }

    if (!result.output_reshape_node) {
        LOG_WARN("Could not find output Reshape node");
        return std::nullopt;
    }

    LOG_DEBUG("Validation complete - all required nodes found");
    return result;
}

std::shared_ptr<ov::Model> transform_to_single_expert(const std::shared_ptr<ov::Model>& original_model,
                                                      MoEValidationResult& validation_result) {
    LOG_DEBUG("Transforming model to single expert...");
    LOG_BLOCK();

    auto model = original_model->clone();
    const auto num_experts = validation_result.num_experts;

    // Helper: Find node in cloned model by friendly name
    auto find_node_by_name = [&](const std::string& name) -> std::shared_ptr<ov::Node> {
        for (const auto& node : model->get_ordered_ops()) {
            if (node->get_friendly_name() == name) {
                return node;
            }
        }
        return nullptr;
    };

    // Step 1: Find Tile node in cloned model using friendly name from validation
    auto tile_node_name = validation_result.tile_node->get_friendly_name();
    auto tile_node = std::dynamic_pointer_cast<ov::op::v0::Tile>(find_node_by_name(tile_node_name));
    if (!tile_node) {
        LOG_ERROR("Could not find Tile node '" << tile_node_name << "' in cloned model");
        return nullptr;
    }

    auto tile_output = tile_node->output(0);
    auto tile_input = tile_node->input_value(0);

    // Step 2: Find node before MatMul (either Tile output or Reshape output)
    ov::Output<ov::Node> node_before_matmul;
    for (const auto& target_input : tile_output.get_target_inputs()) {
        auto consumer = target_input.get_node()->shared_from_this();
        if (auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(consumer)) {
            node_before_matmul = reshape->output(0);
            break;
        } else if (std::dynamic_pointer_cast<ov::op::v0::MatMul>(consumer)) {
            node_before_matmul = tile_output;
            break;
        }
    }

    if (!node_before_matmul.get_node()) {
        LOG_ERROR("Could not find node before MatMul");
        return nullptr;
    }

    // Step 3: Replace Tile path with single-expert Reshape
    LOG_DEBUG("Replacing Tile with single-expert Reshape");
    auto matmul_input_shape = node_before_matmul.get_shape();
    ov::Shape single_expert_shape = matmul_input_shape;
    single_expert_shape[0] = 1;

    auto target_shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                     ov::Shape{single_expert_shape.size()},
                                                                     single_expert_shape);
    auto new_reshape = std::make_shared<ov::op::v1::Reshape>(tile_input, target_shape_const, false);
    new_reshape->set_friendly_name(tile_node->get_friendly_name() + "_single_expert");
    node_before_matmul.replace(new_reshape->output(0));

    // Step 4: Fix output Reshape (change shape from num_experts to 1)
    // Use the node name from validation result
    LOG_DEBUG("Fixing output Reshape node");
    auto output_reshape_name = validation_result.output_reshape_node->get_friendly_name();
    auto output_reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(find_node_by_name(output_reshape_name));

    if (output_reshape) {
        auto shape_input = output_reshape->input_value(1);
        if (auto shape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_input.get_node_shared_ptr())) {
            auto shape_data = shape_const->cast_vector<int64_t>();
            if (!shape_data.empty() && shape_data[0] == static_cast<int64_t>(num_experts)) {
                LOG_DEBUG("  Found output Reshape with shape[0]=" << shape_data[0]);

                shape_data[0] = 1;
                auto new_shape_const =
                    std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{shape_data.size()}, shape_data);
                auto new_output_reshape =
                    std::make_shared<ov::op::v1::Reshape>(output_reshape->input_value(0), new_shape_const, false);
                new_output_reshape->set_friendly_name(output_reshape->get_friendly_name() + "_single_expert");
                output_reshape->output(0).replace(new_output_reshape->output(0));
            }
        }
    } else {
        LOG_WARN("Could not find output Reshape node '" << output_reshape_name << "' in cloned model");
    }

    // Step 5: Replace ReduceSum with Add (add user-controlled aggregation weight parameter)
    // Use the node name from validation result
    LOG_DEBUG("Replacing ReduceSum with Add operation");
    if (validation_result.reduce_sum_node) {
        auto reduce_sum_name = validation_result.reduce_sum_node->get_friendly_name();
        auto reduce_sum = std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(find_node_by_name(reduce_sum_name));

        if (reduce_sum) {
            auto reduce_input = reduce_sum->input_value(0);
            auto reduce_input_shape = reduce_input.get_shape();

            // Create aggregation weight parameter for user control
            auto weight_param = std::make_shared<ov::op::v0::Parameter>(reduce_input.get_element_type(),
                                                                        ov::PartialShape(reduce_input_shape));
            weight_param->set_friendly_name("moe_expert_aggregation_weight");
            model->add_parameters({weight_param});

            LOG_DEBUG("  Added aggregation weight param with shape: " << reduce_input_shape);

            // Replace ReduceSum with Add
            auto add_op = std::make_shared<ov::op::v1::Add>(reduce_input, weight_param);
            add_op->set_friendly_name(reduce_sum->get_friendly_name() + "_replaced_with_add");
            reduce_sum->output(0).replace(add_op->output(0));

            LOG_DEBUG("  Replaced ReduceSum with Add");
        } else {
            LOG_WARN("Could not find ReduceSum node '" << reduce_sum_name << "' in cloned model");
        }
    } else {
        LOG_WARN("No ReduceSum node in validation result");
    }

    // Step 7: Ensure all tensors have names
    LOG_DEBUG("Ensuring tensor names");
    size_t in_idx = 0;
    for (auto& input : model->inputs()) {
        if (input.get_tensor().get_names().empty()) {
            input.get_tensor().set_names({"moe_in_tensor_" + std::to_string(in_idx)});
        }
        in_idx++;
    }
    size_t out_idx = 0;
    for (auto& output : model->outputs()) {
        if (output.get_tensor().get_names().empty()) {
            output.get_tensor().set_names({"moe_out_tensor_" + std::to_string(out_idx)});
        }
        out_idx++;
    }

    // Step 6: Fix parameters with num_experts dimension (change to 1)
    LOG_DEBUG("Fixing parameters with num_experts dimension");
    for (const auto& param : model->get_parameters()) {
        auto param_shape = param->get_partial_shape();
        if (param_shape.rank().is_static() && param_shape.rank().get_length() > 0) {
            auto shape = param_shape.to_shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (shape[i] == num_experts) {
                    LOG_DEBUG("  Fixing parameter '" << param->get_friendly_name() << "' shape[" << i << "]");
                    shape[i] = 1;
                    param->set_partial_shape(ov::PartialShape(shape));
                    param->validate_and_infer_types();
                    break;
                }
            }
        }
    }

    // Step 8: Validate and save
    model->validate_nodes_and_infer_types();
    LOG_DEBUG("Model validation passed");

    // Save debug model
    try {
        ov::serialize(model, "moe_single_expert_model.xml");
        LOG_INFO("Saved single expert model to: moe_single_expert_model.xml");
    } catch (const std::exception& e) {
        LOG_WARN("Failed to save debug model: " << e.what());
    }

    LOG_DEBUG("Transformation complete");
    return model;
}

std::optional<MoEExperts> MoEExperts::from(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("Creating MoEExperts from model: " << model->get_friendly_name());
    LOG_BLOCK();
    std::cout << "Creating MoEExperts from model: " << model->get_friendly_name() << std::endl;

    // Step 1: Validate the model and extract expert information
    auto validation_result = validate_and_setup_moe_expert(model);
    if (!validation_result || !validation_result->is_valid()) {
        LOG_WARN("Model validation failed for MoE expert pattern");
        return std::nullopt;
    }

    // Step 2: Transform the model to single expert
    auto single_expert_model = transform_to_single_expert(model, *validation_result);
    if (!single_expert_model) {
        LOG_WARN("Failed to transform model to single expert");
        return std::nullopt;
    }

    // Step 3: Populate MoEExperts structure
    MoEExperts moe_experts;
    moe_experts._num_experts = validation_result->num_experts;
    moe_experts._expert_hidden_dim = validation_result->expert_hidden_dim;
    moe_experts._input_batch_size = validation_result->input_batch_size;
    moe_experts._single_expert_model = single_expert_model;
    moe_experts._original_model = model;
    moe_experts._tile_op = validation_result->tile_node;
    moe_experts._original_tile_output_shape = validation_result->tile_node->output(0).get_shape();
    moe_experts._single_expert_shape = ov::Shape{validation_result->expert_hidden_dim};
    moe_experts._router_param_idx = validation_result->router_param_idx;

    // Step 4: Extract input/output information
    LOG_DEBUG("Extracting I/O information...");
    for (const auto& input : single_expert_model->inputs()) {
        ExpertIO io_info;
        io_info.name = input.get_any_name();
        io_info.element_type = input.get_element_type();
        io_info.shape = input.get_partial_shape();
        moe_experts._inputs.push_back(io_info);
        LOG_DEBUG("  Input: " << io_info.name << " [" << io_info.element_type << ", " << io_info.shape << "]");
    }

    for (const auto& output : single_expert_model->outputs()) {
        ExpertIO io_info;
        io_info.name = output.get_any_name();
        io_info.element_type = output.get_element_type();
        io_info.shape = output.get_partial_shape();
        moe_experts._outputs.push_back(io_info);
        LOG_DEBUG("  Output: " << io_info.name << " [" << io_info.element_type << ", " << io_info.shape << "]");
    }

    // Validation
    if (!moe_experts.is_valid()) {
        LOG_WARN("Created MoEExperts structure is invalid");
        return std::nullopt;
    }

    LOG_INFO("Successfully created MoEExperts:");
    LOG_INFO("  - Number of experts: " << moe_experts._num_experts);
    LOG_INFO("  - Expert hidden dim: " << moe_experts._expert_hidden_dim);
    LOG_INFO("  - Single expert model: " << single_expert_model->get_friendly_name());
    if (moe_experts._router_param_idx.has_value()) {
        LOG_INFO("  - Router parameter index: " << moe_experts._router_param_idx.value());
    }

    std::cout << "Successfully created MoEExperts with " << moe_experts._num_experts << " experts." << std::endl;
    if (moe_experts._router_param_idx.has_value()) {
        std::cout << "  Router parameter index: " << moe_experts._router_param_idx.value() << std::endl;
    }

    return moe_experts;
}

}  // namespace function

namespace compiled {

MoEExperts::MoEExperts(const function::MoEExperts& func_moe) {
    num_experts = func_moe._num_experts;
    expert_hidden_dim = func_moe._expert_hidden_dim;
    router_param_idx = func_moe._router_param_idx;
    _model_to_compile = func_moe._single_expert_model;

    LOG_DEBUG("Created compiled::MoEExperts with " << num_experts << " experts");
    if (router_param_idx.has_value()) {
        LOG_DEBUG("  Router parameter index: " << router_param_idx.value());
    }
}

void MoEExperts::set_compiled_model(ov::SoPtr<ov::ICompiledModel>&& compiled_model) {
    _compiled_model = std::move(compiled_model);
    _model_to_compile.reset();  // Free memory after compilation

    LOG_DEBUG("Set compiled model for MoE experts");
}

}  // namespace compiled

namespace runtime {
namespace moe_experts {

// TODO: Implement runtime execution logic

}  // namespace moe_experts
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
