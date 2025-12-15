// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "moe_experts.hpp"

#include <algorithm>
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

    // Step 1: Find the Tile operation to determine the number of experts
    for (const auto& node : model->get_ordered_ops()) {
        if (auto tile = std::dynamic_pointer_cast<ov::op::v0::Tile>(node)) {
            // Get the repeats input (second input to Tile)
            auto repeats_input = tile->input_value(1);

            // Try to get the constant value for repeats
            if (auto repeats_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(repeats_input.get_node_shared_ptr())) {
                auto repeats_data = repeats_const->cast_vector<int64_t>();

                // The number of experts is typically the first dimension being repeated
                if (!repeats_data.empty() && repeats_data[0] > 1) {
                    result.num_experts = static_cast<size_t>(repeats_data[0]);
                    result.tile_node = tile;
                    result.input_node = tile->input_value(0).get_node_shared_ptr();

                    LOG_DEBUG("Found Tile operation with " << result.num_experts << " experts");
                    break;
                }
            }
        }
    }

    if (!result.tile_node || result.num_experts == 0) {
        LOG_WARN("Could not find valid Tile operation or determine number of experts");
        return std::nullopt;
    }

    // Step 2: Extract shape information
    auto tile_input = result.tile_node->input_value(0);
    auto tile_output = result.tile_node->output(0);
    auto tile_output_shape = tile_output.get_shape();

    if (tile_output_shape.empty()) {
        LOG_WARN("Tile output shape is empty");
        return std::nullopt;
    }

    // Validate that the shape is evenly divisible by num_experts
    if (tile_output_shape[0] % result.num_experts != 0) {
        LOG_WARN("Tile output shape[0] (" << tile_output_shape[0] << ") is not evenly divisible by num_experts ("
                                          << result.num_experts << ")");
        return std::nullopt;
    }

    // Calculate expert hidden dimension (single expert size)
    result.expert_hidden_dim = tile_output_shape[0] / result.num_experts;
    result.input_batch_size = tile_input.get_shape()[0];

    LOG_DEBUG("Expert hidden dimension: " << result.expert_hidden_dim);
    LOG_DEBUG("Input batch size: " << result.input_batch_size);

    return result;
}

std::shared_ptr<ov::Model> transform_to_single_expert(const std::shared_ptr<ov::Model>& original_model,
                                                      const MoEValidationResult& validation_result) {
    LOG_DEBUG("Transforming model to single expert...");
    LOG_BLOCK();

    auto model = original_model->clone();
    const auto num_experts = validation_result.num_experts;

    // Lambda: Find Tile node in the cloned model
    auto find_tile_node = [&]() -> std::shared_ptr<ov::op::v0::Tile> {
        for (const auto& node : model->get_ordered_ops()) {
            if (auto tile = std::dynamic_pointer_cast<ov::op::v0::Tile>(node)) {
                auto repeats_input = tile->input_value(1);
                if (auto repeats_const =
                        std::dynamic_pointer_cast<ov::op::v0::Constant>(repeats_input.get_node_shared_ptr())) {
                    auto repeats_data = repeats_const->cast_vector<int64_t>();
                    if (!repeats_data.empty() && repeats_data[0] == static_cast<int64_t>(num_experts)) {
                        return tile;
                    }
                }
            }
        }
        return nullptr;
    };

    // Lambda: Find MatMul consumer of Tile (possibly through Reshape)
    auto find_matmul_and_predecessor =
        [&](ov::Output<ov::Node>& tile_output) -> std::pair<std::shared_ptr<ov::Node>, ov::Output<ov::Node>> {
        for (const auto& target_input : tile_output.get_target_inputs()) {
            auto consumer = target_input.get_node()->shared_from_this();

            if (auto reshape_op = std::dynamic_pointer_cast<ov::op::v1::Reshape>(consumer)) {
                LOG_DEBUG("Found Reshape after Tile: " << reshape_op->get_friendly_name());
                for (const auto& reshape_target : reshape_op->output(0).get_target_inputs()) {
                    if (auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(
                            reshape_target.get_node()->shared_from_this())) {
                        LOG_DEBUG("Found MatMul after Reshape: " << mm->get_friendly_name());
                        return {mm, reshape_op->output(0)};
                    }
                }
            } else if (auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(consumer)) {
                LOG_DEBUG("Found MatMul directly after Tile: " << mm->get_friendly_name());
                return {mm, tile_output};
            }
        }
        return {nullptr, ov::Output<ov::Node>()};
    };

    // Lambda: Replace Tile->MatMul path with new Reshape
    auto replace_tile_with_reshape = [&](const ov::Output<ov::Node>& tile_input,
                                         ov::Output<ov::Node>& node_before_matmul,
                                         const std::string& friendly_name) {
        auto matmul_input_shape = node_before_matmul.get_shape();
        ov::Shape single_expert_shape = matmul_input_shape;
        single_expert_shape[0] = 1;

        LOG_DEBUG("Original MatMul input shape: " << matmul_input_shape);
        LOG_DEBUG("Single expert shape: " << single_expert_shape);

        auto target_shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                         ov::Shape{single_expert_shape.size()},
                                                                         single_expert_shape);

        auto new_reshape = std::make_shared<ov::op::v1::Reshape>(tile_input, target_shape_const, false);
        new_reshape->set_friendly_name(friendly_name + "_single_expert");

        node_before_matmul.replace(new_reshape->output(0));
    };

    // Lambda: Fix output Reshape nodes
    // Output path: Reshape -> Multiply -> Convert(optional) -> Result
    auto fix_output_reshapes = [&]() {
        LOG_DEBUG("Fixing output Reshape nodes...");
        for (const auto& result : model->get_results()) {
            auto result_input = result->input_value(0);
            auto result_input_node = result_input.get_node_shared_ptr();

            // Skip Convert node if present (Result <- Convert)
            if (auto convert_node = std::dynamic_pointer_cast<ov::op::v0::Convert>(result_input_node)) {
                LOG_DEBUG("  Skipping Convert node before Result");
                result_input = convert_node->input_value(0);
                result_input_node = result_input.get_node_shared_ptr();
            }

            // Check for Multiply node (Result <- [Convert] <- Multiply)
            std::shared_ptr<ov::Node> reshape_node_ptr;
            if (auto multiply_node = std::dynamic_pointer_cast<ov::op::v1::Multiply>(result_input_node)) {
                LOG_DEBUG("  Found Multiply node before Result");
                // Multiply has two inputs, one should be from Reshape
                auto multiply_input0 = multiply_node->input_value(0).get_node_shared_ptr();
                auto multiply_input1 = multiply_node->input_value(1).get_node_shared_ptr();

                if (auto reshape0 = std::dynamic_pointer_cast<ov::op::v1::Reshape>(multiply_input0)) {
                    reshape_node_ptr = reshape0;
                } else if (auto reshape1 = std::dynamic_pointer_cast<ov::op::v1::Reshape>(multiply_input1)) {
                    reshape_node_ptr = reshape1;
                }
            } else if (auto reshape_direct = std::dynamic_pointer_cast<ov::op::v1::Reshape>(result_input_node)) {
                // Direct Reshape -> Result (fallback case)
                reshape_node_ptr = reshape_direct;
            }

            if (reshape_node_ptr) {
                auto reshape_node = std::dynamic_pointer_cast<ov::op::v1::Reshape>(reshape_node_ptr);
                auto shape_input = reshape_node->input_value(1);
                if (auto shape_const =
                        std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_input.get_node_shared_ptr())) {
                    auto shape_data = shape_const->cast_vector<int64_t>();

                    if (!shape_data.empty() && shape_data[0] == static_cast<int64_t>(num_experts)) {
                        LOG_DEBUG("  Found output Reshape '" << reshape_node->get_friendly_name() << "' with shape[0]="
                                                             << shape_data[0] << ", changing to 1");
                        shape_data[0] = 1;

                        auto new_shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                                      ov::Shape{shape_data.size()},
                                                                                      shape_data);
                        auto new_output_reshape =
                            std::make_shared<ov::op::v1::Reshape>(reshape_node->input_value(0), new_shape_const, false);
                        new_output_reshape->set_friendly_name(reshape_node->get_friendly_name() + "_single_expert");

                        reshape_node->output(0).replace(new_output_reshape->output(0));
                    }
                }
            }
        }
    };

    // Lambda: Ensure all tensors have names
    auto ensure_tensor_names = [&]() {
        LOG_DEBUG("Ensuring all tensors have names...");
        size_t in_tensor_idx = 0;
        for (auto& input : model->inputs()) {
            if (input.get_tensor().get_names().empty()) {
                std::string name = "moe_in_tensor_" + std::to_string(in_tensor_idx);
                input.get_tensor().set_names({name});
                LOG_DEBUG("  Added input tensor name: " << name);
            }
            in_tensor_idx++;
        }

        size_t out_tensor_idx = 0;
        for (auto& output : model->outputs()) {
            if (output.get_tensor().get_names().empty()) {
                std::string name = "moe_out_tensor_" + std::to_string(out_tensor_idx);
                output.get_tensor().set_names({name});
                LOG_DEBUG("  Added output tensor name: " << name);
            }
            out_tensor_idx++;
        }
    };

    // Lambda: Fix parameters with num_experts in their shape (manually, without model->reshape())
    auto fix_parameters_with_num_experts = [&]() {
        LOG_DEBUG("Fixing Parameter nodes with num_experts in shape...");

        for (const auto& param : model->get_parameters()) {
            auto param_shape = param->get_partial_shape();
            if (param_shape.rank().is_static() && param_shape.rank().get_length() > 0) {
                auto shape = param_shape.to_shape();
                bool needs_fix = false;
                size_t fix_dim = 0;

                for (size_t i = 0; i < shape.size(); ++i) {
                    if (shape[i] == num_experts) {
                        needs_fix = true;
                        fix_dim = i;
                        break;
                    }
                }

                if (needs_fix) {
                    LOG_DEBUG("  Found Parameter '" << param->get_friendly_name() << "' with shape[" << fix_dim
                                                    << "]=" << num_experts);

                    shape[fix_dim] = 1;

                    // Directly set the new shape on the parameter
                    param->set_partial_shape(ov::PartialShape(shape));
                    param->validate_and_infer_types();
                    LOG_DEBUG("    Successfully updated parameter shape");
                }
            }
        }
    };

    // Lambda: Save model for debugging
    auto save_debug_model = [&]() {
        try {
            std::string debug_path = "moe_single_expert_model.xml";
            ov::serialize(model, debug_path);
            LOG_INFO("Saved single expert model to: " << debug_path);
            std::cout << "Saved single expert model to: " << debug_path << std::endl;
        } catch (const std::exception& e) {
            LOG_WARN("Failed to save single expert model for debugging: " << e.what());
        }
    };

    // Main transformation flow
    auto tile_op = find_tile_node();
    if (!tile_op) {
        LOG_ERROR("Could not find Tile operation in cloned model");
        return nullptr;
    }

    auto tile_input = tile_op->input_value(0);
    auto tile_output = tile_op->output(0);

    auto [matmul_node, node_before_matmul] = find_matmul_and_predecessor(tile_output);
    if (!matmul_node) {
        LOG_ERROR("Could not find MatMul operation after Tile");
        return nullptr;
    }

    replace_tile_with_reshape(tile_input, node_before_matmul, tile_op->get_friendly_name());
    fix_output_reshapes();
    fix_parameters_with_num_experts();
    ensure_tensor_names();

    model->validate_nodes_and_infer_types();
    LOG_DEBUG("Successfully transformed to single expert model");

    save_debug_model();

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

    std::cout << "Successfully created MoEExperts with " << moe_experts._num_experts << " experts." << std::endl;

    return moe_experts;
}

}  // namespace function

namespace compiled {

MoEExperts::MoEExperts(const function::MoEExperts& func_moe) {
    num_experts = func_moe._num_experts;
    expert_hidden_dim = func_moe._expert_hidden_dim;
    _model_to_compile = func_moe._single_expert_model;

    LOG_DEBUG("Created compiled::MoEExperts with " << num_experts << " experts");
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
