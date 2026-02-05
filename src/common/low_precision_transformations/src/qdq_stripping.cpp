// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/qdq_stripping.hpp"

#include <memory>
#include <queue>
#include <unordered_set>

#include "itt.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/lpt_itt.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {
namespace low_precision {

namespace {

// Helper to detect weight dequantization pattern: Convert->Subtract->Multiply
std::shared_ptr<ov::op::v1::Multiply> find_weight_dequant_multiply(const std::shared_ptr<Node>& node) {
    // Check if this is a Multiply that's part of dequant pattern
    auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(node);
    if (!multiply)
        return nullptr;

    // Dequantization Multiply has one input that is Subtract or Convert,
    // and another input that is a Constant scale factor
    bool has_subtract_or_convert = false;
    bool has_constant = false;

    for (size_t i = 0; i < multiply->get_input_size(); ++i) {
        auto input = multiply->get_input_node_shared_ptr(i);
        if (ov::is_type<ov::op::v1::Subtract>(input) || ov::is_type<ov::op::v0::Convert>(input)) {
            has_subtract_or_convert = true;
        }
        if (ov::is_type<ov::op::v0::Constant>(input)) {
            has_constant = true;
        }
    }

    // Valid dequant pattern needs both
    if (has_subtract_or_convert && has_constant) {
        return multiply;
    }
    return nullptr;
}

// Helper to apply scale directly to weight constant or DQ subgraph
void apply_scale_to_weight(const std::shared_ptr<Node>& weight_node, float scale) {
    std::cout << "  [ DEBUG ]     Applying scale " << scale << " to " << weight_node->get_friendly_name() << std::endl;
    
    // Check if this is a dequantization Multiply - modify its scale constant
    if (auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(weight_node)) {
        std::cout << "  [ DEBUG ]       Modifying existing DQ Multiply scale" << std::endl;
        // Find the scale constant (typically input 1, but check both)
        for (size_t i = 0; i < multiply->get_input_size(); ++i) {
            auto input = multiply->get_input_node_shared_ptr(i);
            if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input)) {
                // This is the scale constant - multiply it by scale
                auto old_values = constant->cast_vector<float>();
                std::vector<float> new_values;
                for (auto val : old_values) {
                    new_values.push_back(val * scale);
                }
                auto new_constant =
                    ov::op::v0::Constant::create(constant->get_element_type(), constant->get_shape(), new_values);
                multiply->input(i).replace_source_output(new_constant);
                std::cout << "  [ DEBUG ]       Updated DQ scale constant" << std::endl;
                return;
            }
        }
        std::cout << "  [ WARNING ]   Could not find scale constant in DQ Multiply" << std::endl;
        return;
    }

    // Otherwise it's a Constant weight - scale it directly
    if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(weight_node)) {
        std::cout << "  [ DEBUG ]       Scaling constant weight directly" << std::endl;
        auto old_values = constant->cast_vector<float>();
        std::vector<float> new_values;
        for (auto val : old_values) {
            new_values.push_back(val * scale);
        }
        auto new_constant = ov::op::v0::Constant::create(constant->get_element_type(), constant->get_shape(), new_values);
        
        // Replace the constant in the graph
        for (auto& output : weight_node->outputs()) {
            for (auto target_input : output.get_target_inputs()) {
                target_input.replace_source_output(new_constant);
            }
        }
        std::cout << "  [ DEBUG ]       Replaced constant weight" << std::endl;
    }
}

// Apply scale backward to find and scale weight operations
void apply_scale_backward(const std::shared_ptr<Node>& node,
                          float scale_adj,
                          std::unordered_set<std::shared_ptr<Node>>& visited) {
    if (!node || visited.count(node)) {
        if (!node) {
            std::cout << "  [ DEBUG ] apply_scale_backward: nullptr node, returning" << std::endl;
        } else {
            std::cout << "  [ DEBUG ] apply_scale_backward: already visited " << node->get_friendly_name() << ", returning" << std::endl;
        }
        return;
    }
    visited.insert(node);

    std::cout << "  [ DEBUG ] apply_scale_backward: node=" << node->get_friendly_name()
              << " type=" << node->get_type_name() << " scale_adj=" << scale_adj << std::endl;

    using namespace ov::pass::pattern;

    // Case 1: Convolution + Add (bias) - scale both Add's constant and Conv weights
    {
        auto conv_pattern = wrap_type<ov::op::v1::Convolution>();
        auto add_pattern = wrap_type<ov::op::v1::Add>({conv_pattern, any_input()});
        auto matcher = std::make_shared<Matcher>(add_pattern, "ConvAddPattern");
        
        if (matcher->match(node)) {
            std::cout << "  [ DEBUG ]   Matched Conv+Add(bias) pattern" << std::endl;
            auto pattern_map = matcher->get_pattern_value_map();
            auto conv_node = ov::as_type_ptr<ov::op::v1::Convolution>(pattern_map[conv_pattern].get_node_shared_ptr());
            auto add_node = ov::as_type_ptr<ov::op::v1::Add>(pattern_map[add_pattern].get_node_shared_ptr());
            
            OPENVINO_ASSERT(conv_node && add_node, "Matched Conv+Add pattern but failed to extract nodes");
            
            // Scale Conv weights (input 1)
            auto weight_input = conv_node->get_input_node_shared_ptr(1);
            std::cout << "  [ DEBUG ]     Conv weight: " << weight_input->get_friendly_name() << std::endl;
            
            auto dequant_multiply = find_weight_dequant_multiply(weight_input);
            if (dequant_multiply) {
                apply_scale_to_weight(dequant_multiply, scale_adj);
                std::cout << "  [ DEBUG ]     Scaled weight dequant Multiply" << std::endl;
            } else if (ov::is_type<ov::op::v0::Constant>(weight_input)) {
                apply_scale_to_weight(weight_input, scale_adj);
                std::cout << "  [ DEBUG ]     Scaled constant weight" << std::endl;
            }
            
            // Scale Add's constant (bias)
            for (size_t i = 0; i < add_node->get_input_size(); ++i) {
                auto input = add_node->get_input_node_shared_ptr(i);
                if (ov::is_type<ov::op::v0::Constant>(input)) {
                    apply_scale_to_weight(input, scale_adj);
                    std::cout << "  [ DEBUG ]     Scaled Add bias constant: " << input->get_friendly_name() << std::endl;
                    break;
                }
            }
            return;
        } else {
            std::cout << "  [ DEBUG ]   Conv+Add pattern did not match" << std::endl;
        }
    }

    // Case 2 & 3: MatMul with constant weights or MatMul with 2 activations
    if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node)) {
        std::cout << "  [ DEBUG ]   MatMul node" << std::endl;
        
        // Check both inputs for constants/weights
        bool found_weight = false;
        for (size_t i = 0; i < matmul->get_input_size(); ++i) {
            auto input_node = matmul->get_input_node_shared_ptr(i);
            std::cout << "  [ DEBUG ]     Input " << i << ": " << input_node->get_friendly_name() << std::endl;
            
            // Check for dequantization pattern or constant
            auto dequant_multiply = find_weight_dequant_multiply(input_node);
            if (dequant_multiply) {
                apply_scale_to_weight(dequant_multiply, scale_adj);
                std::cout << "  [ DEBUG ]     Found and scaled weight dequant Multiply" << std::endl;
                found_weight = true;
            } else if (ov::is_type<ov::op::v0::Constant>(input_node)) {
                apply_scale_to_weight(input_node, scale_adj);
                std::cout << "  [ DEBUG ]     Found and scaled constant weight" << std::endl;
                found_weight = true;
            }
        }
        
        // Case 2: Found weight(s) - done
        if (found_weight) {
            std::cout << "  [ DEBUG ]   MatMul: Found weights, done" << std::endl;
            return;
        }
        
        // Case 3: MatMul with 2 activations - propagate to input 1
        std::cout << "  [ DEBUG ]   MatMul with 2 activations, propagating to input(1)" << std::endl;
        if (matmul->get_input_size() > 1) {
            apply_scale_backward(matmul->get_input_node_shared_ptr(1), scale_adj, visited);
        }
        return;
    }

    // Case 4 & 5: Multiply with constant or Multiply with 2 activations
    if (auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(node)) {
        std::cout << "  [ DEBUG ]   Multiply node" << std::endl;
        
        // Check for constant input
        for (size_t i = 0; i < multiply->get_input_size(); ++i) {
            auto input = multiply->get_input_node_shared_ptr(i);
            if (ov::is_type<ov::op::v0::Constant>(input)) {
                // Case 4: Multiply with constant
                apply_scale_to_weight(input, scale_adj);
                std::cout << "  [ DEBUG ]     Found and scaled constant: " << input->get_friendly_name() << std::endl;
                std::cout << "  [ DEBUG ]   Multiply: Found constant, done" << std::endl;
                return;
            }
        }
        
        // Case 5: Multiply with 2 activations - propagate to input 1
        std::cout << "  [ DEBUG ]   Multiply with 2 activations (no constants found), propagating to input(1)" << std::endl;
        if (multiply->get_input_size() > 1) {
            apply_scale_backward(multiply->get_input_node_shared_ptr(1), scale_adj, visited);
        }
        return;
    }

    // Default: propagate to first input
    std::cout << "  [ DEBUG ]   Default case for node type " << node->get_type_name() << ", propagating to input(0)" << std::endl;
    if (node->get_input_size() > 0) {
        apply_scale_backward(node->get_input_node_shared_ptr(0), scale_adj, visited);
    } else {
        std::cout << "  [ DEBUG ]   No inputs available, stopping" << std::endl;
    }
}

// Helper to get scalar float value from a constant node
float get_const_float_value(const std::shared_ptr<Node>& node) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant || ov::shape_size(constant->get_shape()) != 1)
        return 0.0f;
    return constant->cast_vector<float>()[0];
}

}  // namespace

FQStrippingTransformation::FQStrippingTransformation(const std::set<size_t>& levels_to_strip)
    : levels_to_strip(levels_to_strip) {}

bool FQStrippingTransformation::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(FQStrippingTransformation);
    if (levels_to_strip.empty()) {
        return false;
    }

    auto check_fq_constants = [&](const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) -> bool {
        auto is_scalar_const = [](const std::shared_ptr<Node>& node) -> bool {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            if (!constant) {
                return false;
            }
            return ov::shape_size(constant->get_shape()) == 1;
        };

        if (!is_scalar_const(fq->get_input_node_shared_ptr(1)) || !is_scalar_const(fq->get_input_node_shared_ptr(2)) ||
            !is_scalar_const(fq->get_input_node_shared_ptr(3)) || !is_scalar_const(fq->get_input_node_shared_ptr(4))) {
            return false;
        }

        // Check if ranges are valid (not degenerate)
        float input_low = get_const_float_value(fq->get_input_node_shared_ptr(1));
        float input_high = get_const_float_value(fq->get_input_node_shared_ptr(2));
        float output_low = get_const_float_value(fq->get_input_node_shared_ptr(3));
        float output_high = get_const_float_value(fq->get_input_node_shared_ptr(4));
        return std::abs(input_high - input_low) > 1e-6f && std::abs(output_high - output_low) > 1e-6f;
    };

    bool model_changed = false;
    float max_q_scale = 0.0f;

    std::cout << "\n[ INFO ] === QDQ Stripping Pass ===" << std::endl;
    std::cout << "[ INFO ] Total nodes in graph: " << f->get_ops().size() << std::endl;
    std::cout << "[ INFO ] Levels to strip: ";
    for (auto level : levels_to_strip) {
        std::cout << level << " ";
    }
    std::cout << std::endl;

    NodeVector scale_invariant_nodes;
    
    // Process each FQ node
    for (const auto& node : f->get_ordered_ops()) {
        if (ov::is_type_any_of<ov::op::v0::MVN, ov::op::v6::MVN, ov::op::v1::Softmax, ov::op::v8::Softmax>(node)) {
            scale_invariant_nodes.push_back(node);
            continue;
        }

        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node);
        if (!fq || transformation_callback(node)) {
            continue;
        }

        if (!levels_to_strip.count(fq->get_levels())) {
            continue;
        }

        std::cout << "\n======== Processing FQ: " << fq->get_friendly_name() << " (levels=" << fq->get_levels() << ") ========" << std::endl;

        // Check if FQ has valid constants
        if (!check_fq_constants(fq)) {
            std::cout << "  [ DEBUG ] Skipped: invalid or degenerate FQ constants" << std::endl;
            continue;
        }

        // Compute q_scale for this FQ
        float input_low_val =
            ov::as_type_ptr<ov::op::v0::Constant>(fq->input_value(1).get_node_shared_ptr())->cast_vector<float>()[0];
        float input_high_val =
            ov::as_type_ptr<ov::op::v0::Constant>(fq->input_value(2).get_node_shared_ptr())->cast_vector<float>()[0];
        float output_low_val =
            ov::as_type_ptr<ov::op::v0::Constant>(fq->input_value(3).get_node_shared_ptr())->cast_vector<float>()[0];
        float output_high_val =
            ov::as_type_ptr<ov::op::v0::Constant>(fq->input_value(4).get_node_shared_ptr())->cast_vector<float>()[0];
        float input_range = input_high_val - input_low_val;
        float output_range = output_high_val - output_low_val;

        size_t levels = fq->get_levels();
        float q_scale = (levels - 1) / input_range;

        std::cout << "  [ DEBUG ] Input range: [" << input_low_val << ", " << input_high_val << "] = " << input_range << std::endl;
        std::cout << "  [ DEBUG ] Output range: [" << output_low_val << ", " << output_high_val << "] = " << output_range << std::endl;
        std::cout << "  [ DEBUG ] Q scale: " << q_scale << " (levels=" << levels << ")" << std::endl;

        // Track max q_scale
        if (q_scale > max_q_scale) {
            max_q_scale = q_scale;
        }

        // Remove the FQ
        std::cout << "  [ INFO ] Removing FQ: " << fq->get_friendly_name() << std::endl;
        OPENVINO_ASSERT(replace_output_update_name(fq->output(0), fq->input_value(0)), "FQ stripping failed");
        model_changed = true;
        std::cout << "========================================" << std::endl;
    }

    std::cout << "  [ INFO ] Max q_scale across model: " << max_q_scale << std::endl;
    const auto threshold = 1.f;
    if (max_q_scale <= threshold) {
        std::cout << "  [ INFO ] No scale adjustment needed, skipping" << std::endl;
    }

    if (max_q_scale > threshold && scale_invariant_nodes.empty()) {
        std::cout << "  [ INFO ] Scale adjustment is needed, but no scale-invariant nodes found, so this stage is skipped" << std::endl;
    }

    if (max_q_scale > threshold && !scale_invariant_nodes.empty()) {
        std::cout << "\n======== Applying backward scale adjustment ========" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    std::cout << "\n[ INFO ] === QDQ Stripping Pass Completed ===" << std::endl;
    return model_changed;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov