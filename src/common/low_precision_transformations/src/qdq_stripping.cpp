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
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class WeightsDequantizationBlock : public ov::pass::pattern::op::Block {
public:
    WeightsDequantizationBlock() : Block({}, {}, "WeightsDequantizationBlock") {
        using namespace ov::pass::pattern;

        // Const (quantized weights) -> Convert -> [optional Subtract] -> Multiply (with scale const)
        auto weights = wrap_type<ov::op::v0::Constant>();
        auto convert = wrap_type<ov::op::v0::Convert>({weights});

        auto sub_const = wrap_type<ov::op::v0::Constant>();
        auto subtract = wrap_type<ov::op::v1::Subtract>({convert, sub_const});

        // Multiply can have either (subtract, const) or (convert, const) as inputs
        auto mul_input = subtract | convert;
        auto mul_const = wrap_type<ov::op::v0::Constant>();
        auto multiply = wrap_type<ov::op::v1::Multiply>({mul_input, mul_const});

        m_inputs = ov::OutputVector{weights};
        m_outputs = ov::OutputVector{multiply};
        REGISTER_ANCHORS(this, weights, convert, sub_const, subtract, mul_input, mul_const, multiply);
    }
};

namespace {

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
        std::unordered_set<ov::Node*> visited;
        auto skip_node_predicate = [](ov::Node* n) {
            return false;
        };
        const auto scale_adj = 1 / max_q_scale;

        // Lambda to apply scale to weight DQ subgraph
        auto apply_scale_to_weight = [&](const ov::pass::pattern::PatternValueMap& pattern_map,
                                         const std::shared_ptr<WeightsDequantizationBlock>& dq_block) {
            auto original_constant = dq_block->get_anchor("mul_const", pattern_map).value().get_node_shared_ptr();
            auto old_multiply = dq_block->get_anchor("multiply", pattern_map).value().get_node_shared_ptr();
            auto mul_input = dq_block->get_anchor("mul_input", pattern_map).value().get_node_shared_ptr();
            if (visited.find(old_multiply.get()) != visited.end()) {
                std::cout << "        [ DEBUG ]   Node " << mul_input->get_friendly_name()
                          << " already visited, skipping scale adjustment" << std::endl;
                return;
            }

            std::cout << "        [ DEBUG ]     Scaling multiply " << old_multiply->get_friendly_name() << " by "
                      << scale_adj << std::endl;

            // Create new scaled constant
            auto scale_const =
                ov::op::v0::Constant::create(original_constant->get_output_element_type(0), {}, {scale_adj});
            auto new_constant = ov::op::util::make_try_fold<ov::op::v1::Multiply>(original_constant, scale_const);

            // Create new multiply with the scaled constant
            auto new_multiply = old_multiply->clone_with_new_inputs({mul_input, new_constant});

            ov::replace_node(old_multiply, new_multiply);
            visited.insert(new_multiply.get());
        };

        auto adjust_weights_scale = [&](ov::Node* node) {
            std::cout << "    [ INFO ] adjust_weights_scale called for node: " << node->get_friendly_name() << std::endl;
            using namespace ov::pass::pattern;
            const auto node_shared = node->shared_from_this();
            // Case 1: Convolution + Add (bias) - scale both Add's constant and Conv weights
            {
                auto conv_weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
                auto conv_pattern = wrap_type<ov::op::v1::Convolution>({any_input(), conv_weights_dq_block});
                auto bias_dq_block = std::make_shared<WeightsDequantizationBlock>();
                auto add_pattern = wrap_type<ov::op::v1::Add>({conv_pattern, bias_dq_block});
                auto matcher = std::make_shared<Matcher>(add_pattern, "ConvAddPattern");

                if (matcher->match(node_shared)) {
                    std::cout << "        [ INFO ]   Matched Conv+Add(bias) pattern" << std::endl;
                    auto pattern_map = matcher->get_pattern_value_map();
                    apply_scale_to_weight(pattern_map, conv_weights_dq_block);
                    apply_scale_to_weight(pattern_map, bias_dq_block);
                    return;
                }
            }

            // Case 2: MatMul with weights
            {
                auto weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
                auto matmul_pattern = wrap_type<ov::op::v0::MatMul>({any_input(), weights_dq_block});
                auto matcher = std::make_shared<Matcher>(matmul_pattern, "MatMulPattern");

                if (matcher->match(node_shared)) {
                    std::cout << "        [ INFO ]   Matched MatMul with weights pattern" << std::endl;
                    auto pattern_map = matcher->get_pattern_value_map();
                    apply_scale_to_weight(pattern_map, weights_dq_block);
                    return;
                }
            }

            // Case 3: Multiply with weights
            {
                auto weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
                auto multiply_pattern = wrap_type<ov::op::v1::Multiply>({any_input(), weights_dq_block});
                auto matcher = std::make_shared<Matcher>(multiply_pattern, "MultiplyPattern");

                if (matcher->match(node_shared)) {
                    std::cout << "        [ INFO ]   Matched Multiply with weights pattern" << std::endl;
                    auto pattern_map = matcher->get_pattern_value_map();
                    apply_scale_to_weight(pattern_map, weights_dq_block);
                    return;
                }
            }
        };
        for (const auto& node : scale_invariant_nodes) {
            std::cout << "  [ INFO ] Processing scale-invariant node: " << node->get_friendly_name()
                      << " type=" << node->get_type_name() << std::endl;
            ov::op::util::visit_path(node->get_input_node_ptr(0), visited, adjust_weights_scale, skip_node_predicate);
        }
        std::cout << "========================================" << std::endl;
    }

    std::cout << "\n[ INFO ] === QDQ Stripping Pass Completed ===" << std::endl;
    return model_changed;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov