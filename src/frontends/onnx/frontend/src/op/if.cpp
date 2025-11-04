// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/if.hpp"

#include "core/graph.hpp"
#include "core/operator_set.hpp"
#include "openvino/core/model.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "translate_session.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

namespace detail {
ov::OutputVector if_legacy(const ov::frontend::onnx::Node& node) {
    const auto& ng_inputs = node.get_ov_inputs();
    FRONT_END_GENERAL_CHECK(ng_inputs.size() >= 1, "If operator must have at least one input (condition)");

    const auto& subgraphs = node.get_subgraphs();
    FRONT_END_GENERAL_CHECK(subgraphs.count("then_branch") == 1, "Missing 'then_branch' attribute");
    auto then_subgraph = subgraphs.at("then_branch");
    const auto& then_params = then_subgraph->get_ng_parameters();
    auto then_outputs = then_subgraph->get_ov_outputs();

    FRONT_END_GENERAL_CHECK(subgraphs.count("else_branch") == 1, "Missing 'else_branch' attribute");
    auto else_subgraph = subgraphs.at("else_branch");
    const auto& else_params = else_subgraph->get_ng_parameters();
    auto else_outputs = else_subgraph->get_ov_outputs();

    // Align ranks if branches have different output ranks
    // This is necessary because OpenVINO If::resolve_shape() returns fully dynamic rank
    // when then/else branches have different ranks (which is correct per ONNX spec).
    // However, operations like Conv require static rank, so we align ranks by adding
    // Unsqueeze operations to make ranks equal. This is semantically equivalent
    // (adding dimensions of size 1 doesn't change the data).
    ov::OutputVector aligned_then_outputs;
    ov::OutputVector aligned_else_outputs;

    FRONT_END_GENERAL_CHECK(then_outputs.size() == else_outputs.size(),
                            "'then' and 'else' branches have to have the same number of outputs");

    for (size_t i = 0; i < then_outputs.size(); ++i) {
        auto then_shape = then_outputs[i].get_partial_shape();
        auto else_shape = else_outputs[i].get_partial_shape();

        if (then_shape.rank().is_static() && else_shape.rank().is_static()) {
            auto then_rank = then_shape.rank().get_length();
            auto else_rank = else_shape.rank().get_length();

            if (then_rank < else_rank) {
                // Add Unsqueeze operations to then_branch output to match else_branch rank
                // Add dimensions at the beginning (axis 0) to align with else_branch
                auto node = then_outputs[i].get_node_shared_ptr();
                for (int64_t j = 0; j < (else_rank - then_rank); ++j) {
                    // Always add at axis 0 (beginning)
                    auto axes = std::make_shared<v0::Constant>(
                        ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
                    node = std::make_shared<v0::Unsqueeze>(node, axes);
                }
                aligned_then_outputs.push_back(node);
                aligned_else_outputs.push_back(else_outputs[i]);
            } else if (else_rank < then_rank) {
                // Add Unsqueeze operations to else_branch output to match then_branch rank
                // Add dimensions at the beginning (axis 0) to align with then_branch
                auto node = else_outputs[i].get_node_shared_ptr();
                for (int64_t j = 0; j < (then_rank - else_rank); ++j) {
                    // Always add at axis 0 (beginning)
                    auto axes = std::make_shared<v0::Constant>(
                        ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
                    node = std::make_shared<v0::Unsqueeze>(node, axes);
                }
                aligned_then_outputs.push_back(then_outputs[i]);
                aligned_else_outputs.push_back(node);
            } else {
                // Ranks already equal, no alignment needed
                aligned_then_outputs.push_back(then_outputs[i]);
                aligned_else_outputs.push_back(else_outputs[i]);
            }
        } else {
            // At least one rank is dynamic, cannot align - pass through as-is
            aligned_then_outputs.push_back(then_outputs[i]);
            aligned_else_outputs.push_back(else_outputs[i]);
        }
    }

    // Create Models from original subgraph outputs first
    auto then_branch = std::make_shared<ov::Model>(then_outputs, then_params, then_subgraph->get_name());
    auto else_branch = std::make_shared<ov::Model>(else_outputs, else_params, else_subgraph->get_name());

    // Now update Results if we added alignment operations
    // This approach works because Unsqueeze nodes we added don't introduce new Parameters
    auto then_results = then_branch->get_results();
    auto else_results = else_branch->get_results();

    for (size_t i = 0; i < aligned_then_outputs.size(); ++i) {
        if (aligned_then_outputs[i].get_node_shared_ptr() != then_outputs[i].get_node_shared_ptr()) {
            // We modified this output, update the Result
            then_results[i]->input(0).replace_source_output(aligned_then_outputs[i]);
        }
        if (aligned_else_outputs[i].get_node_shared_ptr() != else_outputs[i].get_node_shared_ptr()) {
            // We modified this output, update the Result
            else_results[i]->input(0).replace_source_output(aligned_else_outputs[i]);
        }
    }

    auto if_node = std::make_shared<v8::If>(ng_inputs.at(0));

    if_node->set_then_body(then_branch);
    if_node->set_else_body(else_branch);

    const auto then_branch_inputs_from_parent = then_subgraph->get_inputs_from_parent();
    const auto else_branch_inputs_from_parent = else_subgraph->get_inputs_from_parent();
    const auto actual_then_params = then_branch->get_parameters();
    const auto actual_else_params = else_branch->get_parameters();

    // Two cases:
    // 1. Explicit inputs: ONNX If node has inputs, get_inputs_from_parent() returns them
    // 2. Implicit inputs: Branches use outer scope variables, ng_inputs has them

    if (then_branch_inputs_from_parent.size() > 0) {
        // Case 1: Explicit inputs from ONNX If node
        FRONT_END_GENERAL_CHECK(then_branch_inputs_from_parent.size() == actual_then_params.size(),
                                "inputs_from_parent.size() != then_params.size()");
        FRONT_END_GENERAL_CHECK(else_branch_inputs_from_parent.size() == actual_else_params.size(),
                                "inputs_from_parent.size() != else_params.size()");

        auto then_param = actual_then_params.cbegin();
        for (const auto& from_parent : then_branch_inputs_from_parent) {
            if_node->set_input(from_parent, *then_param, nullptr);
            then_param++;
        }

        auto else_param = actual_else_params.cbegin();
        for (const auto& from_parent : else_branch_inputs_from_parent) {
            if_node->set_input(from_parent, nullptr, *else_param);
            else_param++;
        }
    } else if (ng_inputs.size() > 1) {
        // Case 2: Implicit inputs collected by ONNX Frontend in ng_inputs
        size_t num_implicit = ng_inputs.size() - 1;  // Exclude condition

        FRONT_END_GENERAL_CHECK(num_implicit == actual_then_params.size(),
                                "num_implicit (" + std::to_string(num_implicit) +
                                    ") != then_params (" + std::to_string(actual_then_params.size()) + ")");
        FRONT_END_GENERAL_CHECK(num_implicit == actual_else_params.size(),
                                "num_implicit != else_params");

        for (size_t i = 0; i < num_implicit; ++i) {
            if_node->set_input(ng_inputs[i + 1], actual_then_params[i], actual_else_params[i]);
        }
    } else {
        // Case 3: No inputs (branches don't use external data)
        FRONT_END_GENERAL_CHECK(actual_then_params.size() == 0, "Expected 0 then_params");
        FRONT_END_GENERAL_CHECK(actual_else_params.size() == 0, "Expected 0 else_params");
    }

    FRONT_END_GENERAL_CHECK(then_branch->get_results().size() == else_branch->get_results().size(),
                            "'then' and 'else' branches have to have the same number of outputs");
    auto else_result = else_branch->get_results().cbegin();
    for (const auto& then_result : then_branch->get_results()) {
        if_node->set_output(then_result, *else_result);
        else_result++;
    }
    if_node->validate_and_infer_types();
    return if_node->outputs();
}

ov::OutputVector if_op(const ov::frontend::onnx::Node& node) {
    const auto& ng_inputs = node.get_ov_inputs();
    FRONT_END_GENERAL_CHECK(ng_inputs.size() == 1, "If operator takes only one input");

    auto if_node = std::make_shared<v8::If>(ng_inputs[0]);

    auto then_branch = node.get_attribute_value<std::shared_ptr<ov::Model>>("then_branch", nullptr);
    FRONT_END_GENERAL_CHECK(then_branch != nullptr, "Missing 'then_branch' attribute");
    auto else_branch = node.get_attribute_value<std::shared_ptr<ov::Model>>("else_branch", nullptr);
    FRONT_END_GENERAL_CHECK(else_branch != nullptr, "Missing 'else_branch' attribute");

    if_node->set_then_body(then_branch);
    if_node->set_else_body(else_branch);

    auto translate_session = node.get_translate_session();

    const auto& then_params = then_branch->get_parameters();
    const auto& else_params = else_branch->get_parameters();

    for (auto& input : then_params) {
        const auto& names = input->output(0).get_names();
        ov::Output<ov::Node> known_input;
        for (const auto& name : names) {
            known_input = translate_session->lookup_tensor(name);
            if (known_input.get_node() != nullptr) {
                break;
            }
        }
        if (known_input.get_node() != nullptr) {
            if_node->set_input(known_input, input, nullptr);
        } else {
            FRONT_END_THROW("Non-existent connection in then-branch to " + input->get_friendly_name());
        }
    }

    for (auto& input : else_params) {
        const auto& names = input->output(0).get_names();
        ov::Output<ov::Node> known_input;
        for (const auto& name : names) {
            known_input = translate_session->lookup_tensor(name);
            if (known_input.get_node() != nullptr) {
                break;
            }
        }
        if (known_input.get_node() != nullptr) {
            if_node->set_input(known_input, nullptr, input);
        } else {
            FRONT_END_THROW("Non-existent connection in else-branch to " + input->get_friendly_name());
        }
    }

    auto then_results = then_branch->get_results();
    auto else_results = else_branch->get_results();
    FRONT_END_GENERAL_CHECK(then_results.size() == else_results.size(),
                            "'then' and 'else' branches have to have the same number of outputs");

    // Align ranks if branches have different output ranks
    // Extract outputs from Results
    ov::OutputVector then_outputs;
    ov::OutputVector else_outputs;
    for (const auto& result : then_results) {
        then_outputs.push_back(result->input_value(0));
    }
    for (const auto& result : else_results) {
        else_outputs.push_back(result->input_value(0));
    }

    ov::OutputVector aligned_then_outputs;
    ov::OutputVector aligned_else_outputs;

    for (size_t i = 0; i < then_outputs.size(); ++i) {
        auto then_shape = then_outputs[i].get_partial_shape();
        auto else_shape = else_outputs[i].get_partial_shape();

        if (then_shape.rank().is_static() && else_shape.rank().is_static()) {
            auto then_rank = then_shape.rank().get_length();
            auto else_rank = else_shape.rank().get_length();

            if (then_rank < else_rank) {
                // Add dimensions at the beginning (axis 0) to align with else_branch
                auto node = then_outputs[i].get_node_shared_ptr();
                for (int64_t j = 0; j < (else_rank - then_rank); ++j) {
                    // Always add at axis 0 (beginning)
                    auto axes = std::make_shared<v0::Constant>(
                        ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
                    node = std::make_shared<v0::Unsqueeze>(node, axes);
                }
                aligned_then_outputs.push_back(node);
                aligned_else_outputs.push_back(else_outputs[i]);
            } else if (else_rank < then_rank) {
                // Add dimensions at the beginning (axis 0) to align with then_branch
                auto node = else_outputs[i].get_node_shared_ptr();
                for (int64_t j = 0; j < (then_rank - else_rank); ++j) {
                    // Always add at axis 0 (beginning)
                    auto axes = std::make_shared<v0::Constant>(
                        ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
                    node = std::make_shared<v0::Unsqueeze>(node, axes);
                }
                aligned_then_outputs.push_back(then_outputs[i]);
                aligned_else_outputs.push_back(node);
            } else {
                aligned_then_outputs.push_back(then_outputs[i]);
                aligned_else_outputs.push_back(else_outputs[i]);
            }
        } else {
            // At least one rank is dynamic, cannot align - pass through as-is
            aligned_then_outputs.push_back(then_outputs[i]);
            aligned_else_outputs.push_back(else_outputs[i]);
        }
    }

    // Update Results in the branches
    for (size_t i = 0; i < aligned_then_outputs.size(); ++i) {
        then_results[i]->input(0).replace_source_output(aligned_then_outputs[i]);
        else_results[i]->input(0).replace_source_output(aligned_else_outputs[i]);
    }

    int output_size = static_cast<int>(then_results.size());
    for (int ind = 0; ind < output_size; ++ind) {
        if_node->set_output(then_results[ind], else_results[ind]);
    }

    if_node->validate_and_infer_types();
    return if_node->outputs();
}
}  // namespace detail

ov::OutputVector if_op(const ov::frontend::onnx::Node& node) {
    if (!node.has_decoder()) {
        return detail::if_legacy(node);
    } else {
        return detail::if_op(node);
    }
}
ONNX_OP("If", OPSET_SINCE(1), ai_onnx::opset_1::if_op);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
