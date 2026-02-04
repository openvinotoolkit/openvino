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
#include "openvino/util/log.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {
namespace low_precision {

namespace {

// Helper functions for scale_factor management via rt_info
constexpr const char* SCALE_FACTOR_KEY = "qdq_scale_factor";

float get_scale_factor(const std::shared_ptr<Node>& node) {
    if (!node)
        return 1.0f;
    const auto& rt_info = node->get_rt_info();
    auto it = rt_info.find(SCALE_FACTOR_KEY);
    if (it != rt_info.end()) {
        return it->second.as<float>();
    }
    return 1.0f;
}

void set_scale_factor(const std::shared_ptr<Node>& node, float factor) {
    if (!node)
        return;
    node->get_rt_info()[SCALE_FACTOR_KEY] = factor;
}

void multiply_scale_factor(const std::shared_ptr<Node>& node, float multiplier) {
    float current = get_scale_factor(node);
    set_scale_factor(node, current * multiplier);
}

// Check if operation is scale-invariant (stops propagation)
bool is_scale_invariant_op(const std::shared_ptr<Node>& node) {
    return ov::is_type<ov::op::v0::MVN>(node) || ov::is_type<ov::op::v6::MVN>(node) ||
           ov::is_type<ov::op::v1::Softmax>(node) || ov::is_type<ov::op::v8::Softmax>(node);
}

// Check if node is a transparent scale-propagating op (Convert, etc.)
bool is_transparent_op(const std::shared_ptr<Node>& node) {
    return ov::is_type<ov::op::v0::Convert>(node);
}

// Count effective downstream consumers by traversing through transparent ops
size_t count_effective_consumers(const std::shared_ptr<Node>& node,
                                 std::unordered_set<std::shared_ptr<Node>>& visited,
                                 int depth = 0) {
    if (!node || visited.count(node))
        return 0;
    visited.insert(node);

    size_t count = 0;
    for (auto& output : node->outputs()) {
        auto target_inputs = output.get_target_inputs();
        if (target_inputs.empty())
            continue;

        for (auto& target_input : target_inputs) {
            auto consumer = target_input.get_node()->shared_from_this();
            if (is_transparent_op(consumer)) {
                // Recurse through transparent ops
                count += count_effective_consumers(consumer, visited, depth + 1);
            } else {
                // Found a real consumer
                count++;
            }
        }
    }

    return count;
}

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

// Apply scale backward to find operations to adjust
void apply_scale_backward(const std::shared_ptr<Node>& node,
                          float scale_adj,
                          std::unordered_set<std::shared_ptr<Node>>& visited,
                          std::vector<std::shared_ptr<Node>>& affected_weights) {
    if (!node || visited.count(node))
        return;
    visited.insert(node);

    std::cout << "[ DEBUG ] apply_scale_backward: node=" << node->get_friendly_name()
              << " type=" << node->get_type_name() << " scale_adj=" << scale_adj << std::endl;

    if (ov::is_type<ov::op::v1::Convolution>(node)) {
        std::cout << "[ DEBUG ]   Conv node, checking weight input" << std::endl;
        // For Conv, check weight input (input 1) for dequantization pattern
        if (node->get_input_size() > 1) {
            auto weight_input = node->get_input_node_shared_ptr(1);
            std::cout << "[ DEBUG ]     Weight input: " << weight_input->get_friendly_name()
                      << " type=" << weight_input->get_type_name() << std::endl;

            // Check if it's a dequantization Multiply
            auto dequant_multiply = find_weight_dequant_multiply(weight_input);
            if (dequant_multiply) {
                multiply_scale_factor(dequant_multiply, scale_adj);
                affected_weights.push_back(dequant_multiply);
                std::cout << "[ DEBUG ]     Found weight dequant Multiply: " << dequant_multiply->get_friendly_name()
                          << " new_scale_factor=" << get_scale_factor(dequant_multiply) << std::endl;
            } else if (ov::is_type<ov::op::v0::Constant>(weight_input)) {
                // Unquantized constant weight - will need a new Multiply
                multiply_scale_factor(weight_input, scale_adj);
                affected_weights.push_back(weight_input);
                std::cout << "[ DEBUG ]     Constant weight (unquantized) will get Multiply: "
                          << weight_input->get_friendly_name() << " new_scale_factor=" << get_scale_factor(weight_input)
                          << std::endl;
            }
        }
        return;
    }

    if (ov::is_type<ov::op::v0::MatMul>(node)) {
        std::cout << "[ DEBUG ]   MatMul node, checking weight inputs" << std::endl;
        // For MatMul, check both inputs for dequantization pattern
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            auto input_node = node->get_input_node_shared_ptr(i);
            std::cout << "[ DEBUG ]     Input " << i << ": " << input_node->get_friendly_name()
                      << " type=" << input_node->get_type_name() << std::endl;
            auto dequant_multiply = find_weight_dequant_multiply(input_node);
            if (dequant_multiply) {
                multiply_scale_factor(dequant_multiply, scale_adj);
                affected_weights.push_back(dequant_multiply);
                std::cout << "[ DEBUG ]     Found weight dequant Multiply at input " << i << ": "
                          << dequant_multiply->get_friendly_name()
                          << " new_scale_factor=" << get_scale_factor(dequant_multiply) << std::endl;
                // Don't return yet - check other input too
            }
        }
        // If weights were found, we're done
        if (!affected_weights.empty()) {
            return;
        }
        // Otherwise propagate to first input (activation path)
        if (node->get_input_size() > 0) {
            apply_scale_backward(node->get_input_node_shared_ptr(0), scale_adj, visited, affected_weights);
        }
        return;
    }

    if (ov::is_type<ov::op::v1::Add>(node)) {
        // Propagate to all inputs
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            apply_scale_backward(node->get_input_node_shared_ptr(i), scale_adj, visited, affected_weights);
        }
        return;
    }

    if (ov::is_type<ov::op::v1::Multiply>(node)) {
        // Could be dequantization Multiply or just a regular Multiply - propagate to inputs
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            apply_scale_backward(node->get_input_node_shared_ptr(i), scale_adj, visited, affected_weights);
        }
        return;
    }
}

// Propagate scale forward to collect affected FQ nodes
void propagate_scale_forward(const std::shared_ptr<Node>& node,
                             float scale_adj,
                             std::unordered_set<std::shared_ptr<Node>>& visited,
                             std::vector<std::shared_ptr<Node>>& affected_fqs,
                             const std::vector<std::shared_ptr<Node>>& source_weights) {
    if (!node || visited.count(node))
        return;
    visited.insert(node);

    std::cout << "[ DEBUG ] propagate_scale_forward: node=" << node->get_friendly_name()
              << " type=" << node->get_type_name() << std::endl;

    // Stop at scale-invariant operations
    if (is_scale_invariant_op(node)) {
        std::cout << "[ DEBUG ]   Stopped at scale-invariant op" << std::endl;
        return;
    }

    if (ov::is_type<ov::op::v0::FakeQuantize>(node)) {
        // FakeQuantize on activations - scale it and STOP
        // This FQ will be processed independently in its own iteration
        multiply_scale_factor(node, scale_adj);
        affected_fqs.push_back(node);
        std::cout << "[ DEBUG ]   Scaled downstream FQ: " << node->get_friendly_name()
                  << " new_scale_factor=" << get_scale_factor(node) << std::endl;
        std::cout << "[ DEBUG ]   Stopped propagation at downstream FQ" << std::endl;
        return;
    }

    if (ov::is_type<ov::op::v1::Add>(node)) {
        // For Add, need to propagate up through both branches first to ensure consistency
        // For simplicity, we'll just propagate forward
        for (auto& output : node->outputs()) {
            for (auto& target_input : output.get_target_inputs()) {
                propagate_scale_forward(target_input.get_node()->shared_from_this(),
                                        scale_adj,
                                        visited,
                                        affected_fqs,
                                        source_weights);
            }
        }
        return;
    }

    // Continue propagation forward
    for (auto& output : node->outputs()) {
        for (auto& target_input : output.get_target_inputs()) {
            propagate_scale_forward(target_input.get_node()->shared_from_this(),
                                    scale_adj,
                                    visited,
                                    affected_fqs,
                                    source_weights);
        }
    }
}

// Modify weight scale (for dequant Multiply) or insert new Multiply (for Constant)
void insert_multiply_on_weight(const std::shared_ptr<Node>& weight_node, float multiplier) {
    float scale_factor = get_scale_factor(weight_node);
    if (std::abs(scale_factor - 1.0f) < 1e-6f)
        return;

    std::cout << "[ DEBUG ] insert_multiply_on_weight: node=" << weight_node->get_friendly_name()
              << " type=" << weight_node->get_type_name() << " scale_factor=" << scale_factor << std::endl;

    // Check if this is a dequantization Multiply - modify its scale constant
    if (auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(weight_node)) {
        std::cout << "[ DEBUG ]   Modifying existing dequant Multiply" << std::endl;
        // Find the scale constant (typically input 1, but check both)
        for (size_t i = 0; i < multiply->get_input_size(); ++i) {
            auto input = multiply->get_input_node_shared_ptr(i);
            if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input)) {
                // This is the scale constant - multiply it by scale_factor
                auto old_values = constant->cast_vector<float>();
                std::vector<float> new_values;
                for (auto val : old_values) {
                    new_values.push_back(val * scale_factor);
                }
                auto new_constant =
                    ov::op::v0::Constant::create(constant->get_element_type(), constant->get_shape(), new_values);
                multiply->input(i).replace_source_output(new_constant);
                std::cout << "[ DEBUG ]   Updated dequant scale constant by factor " << scale_factor << std::endl;
                set_scale_factor(weight_node, 1.0f);
                return;
            }
        }
        std::cout << "[ WARNING ] Could not find scale constant in dequant Multiply" << std::endl;
        return;
    }

    // Otherwise it's an unquantized Constant weight - insert new Multiply
    if (ov::is_type<ov::op::v0::Constant>(weight_node)) {
        std::cout << "[ DEBUG ]   Inserting new Multiply for Constant weight" << std::endl;
        auto mult_const = ov::op::v0::Constant::create(weight_node->get_element_type(), ov::Shape{}, {scale_factor});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(weight_node, mult_const);

        // Replace all uses of weight_node (except the multiply we just created) with multiply
        for (auto& output : weight_node->outputs()) {
            for (auto target_input : output.get_target_inputs()) {
                if (target_input.get_node()->shared_from_this() != multiply) {
                    target_input.replace_source_output(multiply);
                }
            }
        }
        std::cout << "[ DEBUG ]   Inserted new Multiply with factor " << scale_factor << std::endl;
    }

    // Reset scale factor
    set_scale_factor(weight_node, 1.0f);
}

// Helper to get scalar float value from a constant node
float get_const_float_value(const std::shared_ptr<Node>& node) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant || ov::shape_size(constant->get_shape()) != 1)
        return 0.0f;
    return constant->cast_vector<float>()[0];
}

// Adjust FakeQuantize intervals by dividing by scale_factor
void adjust_fq_intervals(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) {
    float scale_factor = get_scale_factor(fq);
    if (std::abs(scale_factor - 1.0f) < 1e-6f)
        return;

    float input_low = get_const_float_value(fq->get_input_node_shared_ptr(1));
    float input_high = get_const_float_value(fq->get_input_node_shared_ptr(2));

    std::cout << "[ DEBUG ] adjust_fq_intervals: FQ=" << fq->get_friendly_name() << " scale_factor=" << scale_factor
              << " old_input_range=[" << input_low << ", " << input_high << "]" << std::endl;

    // Adjust input range (divide by scale_factor)
    input_low /= scale_factor;
    input_high /= scale_factor;

    // Create new constants
    auto new_input_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {input_low});
    auto new_input_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {input_high});

    // Replace inputs
    fq->input(1).replace_source_output(new_input_low);
    fq->input(2).replace_source_output(new_input_high);

    std::cout << "[ DEBUG ]   new_input_range=[" << input_low << ", " << input_high << "]" << std::endl;

    // Reset scale factor
    set_scale_factor(fq, 1.0f);
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
    const float threshold = 1.0f;
    const float ratio = 10.0f;

    std::cout << "\n[ INFO ] === QDQ Stripping Pass (threshold=" << threshold << ", ratio=" << ratio
              << ") ===" << std::endl;
    std::cout << "[ INFO ] Total nodes in graph: " << f->get_ops().size() << std::endl;
    std::cout << "[ INFO ] Levels to strip: ";
    for (auto level : levels_to_strip) {
        std::cout << level << " ";
    }
    std::cout << std::endl;

    // Process each FQ node
    for (const auto& node : f->get_ordered_ops()) {
        if (transformation_callback(node)) {
            continue;
        }
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node);
        if (!fq) {
            continue;
        }

        std::cout << "[ DEBUG ] Found FQ: " << fq->get_friendly_name() << " levels=" << fq->get_levels() << std::endl;

        if (!levels_to_strip.count(fq->get_levels())) {
            std::cout << "[ DEBUG ]   Skipped: level not in levels_to_strip" << std::endl;
            continue;
        }

        // Step 1: Check if FQ has valid constants (non-degenerate ranges)
        if (!check_fq_constants(fq)) {
            std::cout << "[ DEBUG ]   Skipped: invalid or degenerate FQ constants" << std::endl;
            continue;
        }

        // Step 2: Compute effective scale for this FQ
        // Get FQ parameters
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

        // Check if this is an identity FQ (QDQ-like where input_range ≈ output_range)
        bool is_identity_fq = std::abs(input_range - output_range) < 1e-4f;

        // Compute actual quantization scale using FQ formula
        // For QDQ-like FQ (where input_range ≈ output_range), this gives the real quantization scale
        size_t levels = fq->get_levels();
        float q_scale = (levels - 1) / input_range;

        float current_scale_factor = get_scale_factor(fq);
        float effective_scale = q_scale / current_scale_factor;

        std::cout << "[ DEBUG ]   Input range: [" << input_low_val << ", " << input_high_val << "] = " << input_range
                  << std::endl;
        std::cout << "[ DEBUG ]   Output range: [" << output_low_val << ", " << output_high_val
                  << "] = " << output_range << std::endl;
        std::cout << "[ DEBUG ]   Q scale: " << q_scale << " (levels=" << levels
                  << "), effective_scale: " << effective_scale << ", identity_fq: " << is_identity_fq << std::endl;

        // Check if this FQ has multiple effective consumers (traversing through transparent ops like Convert)
        std::unordered_set<std::shared_ptr<Node>> visited_consumers;
        size_t num_consumers = count_effective_consumers(fq, visited_consumers);
        bool has_multiple_consumers = num_consumers > 1;

        if (has_multiple_consumers) {
            std::cout << "[ DEBUG ]   FQ has " << num_consumers << " effective consumers" << std::endl;
        }

        // Step 3: Decision based on effective_scale vs threshold
        if (effective_scale <= threshold) {
            // For FQs with multiple effective consumers, skip removal to avoid inconsistent scale adjustments
            if (has_multiple_consumers) {
                std::cout << "[ WARNING ] Skipping FQ with multiple effective consumers (effective_scale="
                          << effective_scale << ", consumers=" << num_consumers << "): " << fq->get_friendly_name()
                          << std::endl;
                continue;
            }

            // Remove FQ if scale is at or below threshold and it only has one consumer path
            std::cout << "[ INFO ] Removing FQ (effective_scale=" << effective_scale << " <= threshold=" << threshold
                      << "): " << fq->get_friendly_name() << std::endl;
            OPENVINO_ASSERT(replace_output_update_name(fq->output(0), fq->input_value(0)), "FQ stripping failed");
            model_changed = true;
        } else if (is_identity_fq) {
            // For identity FQs (QDQ-like), always remove them regardless of scale
            std::cout << "[ INFO ] Removing identity FQ (input_range ≈ output_range): " << fq->get_friendly_name()
                      << std::endl;
            OPENVINO_ASSERT(replace_output_update_name(fq->output(0), fq->input_value(0)), "FQ stripping failed");
            model_changed = true;
            continue;
        }

        // Step 4: Adjust scales via propagation
        std::cout << "\n[ INFO ] --- Adjusting FQ (effective_scale=" << effective_scale << " >= threshold=" << threshold
                  << "): " << fq->get_friendly_name() << " ---" << std::endl;

        // Calculate scale adjustment
        float scale_adj = effective_scale / threshold * ratio;
        std::cout << "[ INFO ] Scale adjustment: " << scale_adj << std::endl;

        // Apply scale backward to find weight nodes to adjust
        std::cout << "[ INFO ] Backward propagation:" << std::endl;
        std::unordered_set<std::shared_ptr<Node>> visited_backward;
        std::vector<std::shared_ptr<Node>> affected_weights;

        if (fq->get_input_size() > 0) {
            auto input_node = fq->get_input_node_shared_ptr(0);
            apply_scale_backward(input_node, scale_adj, visited_backward, affected_weights);
        }

        std::cout << "[ INFO ] Affected weights: " << affected_weights.size() << std::endl;

        // Propagate scale forward to collect affected FQs
        std::cout << "[ INFO ] Forward propagation:" << std::endl;
        std::unordered_set<std::shared_ptr<Node>> visited_forward;
        std::vector<std::shared_ptr<Node>> affected_fqs;

        for (auto& weight_node : affected_weights) {
            visited_forward.clear();
            propagate_scale_forward(weight_node, scale_adj, visited_forward, affected_fqs, affected_weights);
        }

        std::cout << "[ INFO ] Affected downstream FQs: " << affected_fqs.size() << std::endl;

        // If no weights/FQs were affected, we can't compensate - keep the FQ unchanged
        if (affected_weights.empty() && affected_fqs.empty()) {
            std::cout << "[ WARNING ] No compensation possible - keeping FQ unchanged" << std::endl;
            continue;
        }

        // Apply physical modifications
        std::cout << "[ INFO ] Applying physical modifications:" << std::endl;
        // 1. Insert Multiply on weights (both dequant Multiply and Constant)
        for (auto& weight_node : affected_weights) {
            insert_multiply_on_weight(weight_node, scale_adj);
        }

        // 2. Adjust FQ intervals for downstream FQs
        for (auto& affected_node : affected_fqs) {
            if (auto affected_fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(affected_node)) {
                adjust_fq_intervals(affected_fq);
            }
        }

        // After successfully adjusting scales, remove the FQ since its effect is now distributed
        std::cout << "[ INFO ] Removing FQ after scale adjustment (distributed to " << affected_weights.size()
                  << " weights, " << affected_fqs.size() << " FQs)" << std::endl;
        OPENVINO_ASSERT(replace_output_update_name(fq->output(0), fq->input_value(0)),
                        "FQ stripping failed after adjustment");

        model_changed = true;
        std::cout << "[ INFO ] QDQ Stripping: adjusted and removed FakeQuantize " << fq->get_friendly_name()
                  << " (effective_scale=" << effective_scale << ", adjustment=" << scale_adj
                  << ", affected_weights=" << affected_weights.size() << ", affected_fqs=" << affected_fqs.size() << ")"
                  << std::endl;
    }

    return model_changed;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov