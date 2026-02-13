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
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class WeightsDequantizationBlock : public ov::pass::pattern::op::Block {
public:
    WeightsDequantizationBlock() : Block({}, {}, "WeightsDequantizationBlock") {
        using namespace ov::pass::pattern;

        auto weights = wrap_type<ov::op::v0::Constant>();
        auto convert = wrap_type<ov::op::v0::Convert>({weights});

        auto sub_const = wrap_type<ov::op::v0::Constant>();
        auto sub_const_convert = optional<ov::op::v0::Convert>({sub_const});
        auto subtract = wrap_type<ov::op::v1::Subtract>({convert, sub_const_convert});

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

FQStrippingTransformation::FQStrippingTransformation(const std::set<size_t>& levels_to_strip,
                                                     bool need_weights_adjustment)
    : levels_to_strip(levels_to_strip),
      need_weights_adjustment(need_weights_adjustment) {}

bool FQStrippingTransformation::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(FQStrippingTransformation);
    if (levels_to_strip.empty()) {
        return false;
    }

    auto fq_ranges_are_the_same = [&](const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) -> bool {
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

        // Check if input and output ranges are the same — only such FQs should be stripped
        float input_low = get_const_float_value(fq->get_input_node_shared_ptr(1));
        float input_high = get_const_float_value(fq->get_input_node_shared_ptr(2));
        float output_low = get_const_float_value(fq->get_input_node_shared_ptr(3));
        float output_high = get_const_float_value(fq->get_input_node_shared_ptr(4));
        return std::abs(input_low - output_low) <= 1e-6f && std::abs(input_high - output_high) <= 1e-6f;
    };

    bool model_changed = false;
    const float ratio = 10.0f;
    const float threshold = 1.0f;

    // Scale adjustment infrastructure
    std::unordered_set<ov::Node*> visited;
    float current_scale_divisor = 1.0f;
    auto backward_skip_predicate = [](ov::Node* n) {
        return ov::is_type<op::v0::ShapeOf>(n) || ov::is_type<op::v3::ShapeOf>(n);
    };

    // Deferred adjustment storage: during forward propagation we collect adjustments
    // instead of applying them immediately. After the forward pass, if no Result node
    // was encountered, we apply all collected adjustments. If Result was reached, the
    // FQ output feeds directly into a model output without a scale-invariant consumer,
    // so weight scaling would change the model's numerical output — we skip adjustments.

    // Pending weight scale adjustment: stores {old_multiply, original_constant, mul_input}
    struct PendingWeightScale {
        std::shared_ptr<ov::Node> old_multiply;
        std::shared_ptr<ov::Node> original_constant;
        std::shared_ptr<ov::Node> mul_input;
        std::vector<ov::Node*> extra_visited;  // match root inputs to mark visited
    };
    std::vector<PendingWeightScale> pending_weight_scales;

    // Pending FQ range adjustment: stores the FQ node pointer
    std::vector<ov::Node*> pending_fq_adjustments;

    auto collect_scale_to_weight = [&](const ov::pass::pattern::PatternValueMap& pattern_map,
                                       const std::shared_ptr<WeightsDequantizationBlock>& dq_block,
                                       const std::vector<ov::Node*>& extra_visited_nodes) {
        auto original_constant = dq_block->get_anchor("mul_const", pattern_map).value().get_node_shared_ptr();
        auto old_multiply = dq_block->get_anchor("multiply", pattern_map).value().get_node_shared_ptr();
        auto mul_input = dq_block->get_anchor("mul_input", pattern_map).value().get_node_shared_ptr();

        if (visited.find(old_multiply.get()) != visited.end()) {
            return;
        }

        pending_weight_scales.push_back({old_multiply, original_constant, mul_input, extra_visited_nodes});
    };

    auto apply_pending_weight_scale = [&](const PendingWeightScale& pending) {
        auto divisor_const = ov::op::v0::Constant::create(
            pending.original_constant->get_output_element_type(0), {}, {current_scale_divisor});
        auto new_constant = ov::op::util::make_try_fold<ov::op::v1::Divide>(pending.original_constant, divisor_const);
        auto new_multiply = pending.old_multiply->clone_with_new_inputs({pending.mul_input, new_constant});

        ov::replace_node(pending.old_multiply, new_multiply);
        visited.insert(new_multiply.get());
        for (auto* n : pending.extra_visited) {
            visited.insert(n);
        }
    };

    // Helper to adjust FQ range constants (input_low, input_high, output_low, output_high)
    // by dividing them by current_scale_divisor. This is needed when scale propagation passes
    // through an un-stripped FQ — the quantization grid must shift to match the new value range.
    // Uses a separate tracking set (adjusted_fqs) instead of `visited` because visit_path adds
    // nodes to `visited` before calling the callback, which would cause the adjustment to be
    // skipped when reached via backward propagation.
    std::unordered_set<ov::Node*> adjusted_fqs;
    auto collect_fq_adjustment = [&](ov::Node* node) {
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node->shared_from_this());
        if (!fq)
            return;

        if (adjusted_fqs.count(node))
            return;

        pending_fq_adjustments.push_back(node);
        adjusted_fqs.insert(node);
    };

    auto apply_fq_adjustment = [&](ov::Node* node) {
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node->shared_from_this());
        if (!fq)
            return;

        auto divisor_const = ov::op::v0::Constant::create(ov::element::f32, {}, {current_scale_divisor});

        // Adjust all 4 range constants: input_low(1), input_high(2), output_low(3), output_high(4)
        for (size_t idx = 1; idx <= 4; ++idx) {
            auto original_const = fq->get_input_node_shared_ptr(idx);
            auto new_const = ov::op::util::make_try_fold<ov::op::v1::Divide>(original_const, divisor_const);
            fq->input(idx).replace_source_output(new_const);
        }
    };

    auto collect_weights_scale = [&](ov::Node* node) {
        using namespace ov::pass::pattern;
        const auto node_shared = node->shared_from_this();
        // Case 1: Convolution + Add (bias) - scale both Add's constant and Conv weights
        // The bias pattern came from ONNX FE
        {
            // Conv with weights DQ
            auto conv_weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto conv_pattern = wrap_type<ov::op::v1::Convolution>({any_input(), conv_weights_dq_block});

            // Bias DQ and reshape with shape computation
            // Shape computation: ShapeOf(conv) -> ShapeOf -> Subtract -> Broadcast, ShapeOf(bias) -> Concat
            auto conv_shape = wrap_type<ov::op::v3::ShapeOf>({conv_pattern});
            auto conv_rank = wrap_type<ov::op::v3::ShapeOf>({conv_shape});
            auto rank_minus_2 = wrap_type<ov::op::v1::Subtract>({conv_rank, any_input()});
            auto tail = wrap_type<ov::op::v3::Broadcast>({any_input(), rank_minus_2});

            auto bias_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto c_dim = wrap_type<ov::op::v3::ShapeOf>({bias_dq_block});
            auto target_shape = wrap_type<ov::op::v0::Concat>({any_input(), c_dim, tail});
            auto reshape_pattern = wrap_type<ov::op::v1::Reshape>({bias_dq_block, target_shape});

            auto add_pattern = wrap_type<ov::op::v1::Add>({conv_pattern, reshape_pattern});
            auto matcher = std::make_shared<Matcher>(add_pattern, "ConvAddPattern");

            if (matcher->match(node_shared)) {
                auto pattern_map = matcher->get_pattern_value_map();
                std::vector<ov::Node*> extra;
                for (const auto& in : matcher->get_match_root()->input_values()) {
                    extra.push_back(in.get_node());
                }
                collect_scale_to_weight(pattern_map, conv_weights_dq_block, extra);
                collect_scale_to_weight(pattern_map, bias_dq_block, {});
                return;
            }
        }

        // Case 2: MatMul + Add (bias) - scale both MatMul weights and bias
        {
            auto weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto matmul_pattern = wrap_type<ov::op::v0::MatMul>({any_input(), weights_dq_block});

            auto bias_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto add_pattern = wrap_type<ov::op::v1::Add>({matmul_pattern, bias_dq_block});
            auto matcher = std::make_shared<Matcher>(add_pattern, "MatMulAddPattern");

            if (matcher->match(node_shared)) {
                auto pattern_map = matcher->get_pattern_value_map();
                std::vector<ov::Node*> extra;
                for (const auto& in : matcher->get_match_root()->input_values()) {
                    extra.push_back(in.get_node());
                }
                collect_scale_to_weight(pattern_map, weights_dq_block, extra);
                collect_scale_to_weight(pattern_map, bias_dq_block, {});
                return;
            }
        }

        // Case 3: MatMul with weights (no bias)
        {
            auto weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto matmul_pattern = wrap_type<ov::op::v0::MatMul>({any_input(), weights_dq_block});
            auto matcher = std::make_shared<Matcher>(matmul_pattern, "MatMulPattern");

            if (matcher->match(node_shared)) {
                auto pattern_map = matcher->get_pattern_value_map();
                std::vector<ov::Node*> extra;
                for (const auto& in : matcher->get_match_root()->input_values()) {
                    extra.push_back(in.get_node());
                }
                collect_scale_to_weight(pattern_map, weights_dq_block, extra);
                return;
            }
        }

        // Case 4: Multiply with weights
        {
            auto weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto multiply_pattern = wrap_type<ov::op::v1::Multiply>({any_input(), weights_dq_block});
            auto matcher = std::make_shared<Matcher>(multiply_pattern, "MultiplyPattern");

            if (matcher->match(node_shared)) {
                auto pattern_map = matcher->get_pattern_value_map();
                std::vector<ov::Node*> extra;
                for (const auto& in : matcher->get_match_root()->input_values()) {
                    extra.push_back(in.get_node());
                }
                collect_scale_to_weight(pattern_map, weights_dq_block, extra);
                return;
            }
        }

        // Case 5: FakeQuantize (un-stripped) — collect its range constants for adjustment
        if (ov::is_type<ov::op::v0::FakeQuantize>(node)) {
            collect_fq_adjustment(node);
            return;
        }
    };

    // Forward visited set: tracks nodes reached during forward propagation.
    // Declared here (outside the loop) so the forward_propagate_callback can capture it.
    // Cleared before each forward propagation pass.
    std::unordered_set<ov::Node*> forward_visited;

    // Track FQs that have already been stripped so forward propagation skips them.
    // After replace_output_update_name, the stripped FQ node is still connected as a consumer
    // of its input node (only its output consumers were redirected), so forward traversal
    // would still reach it.
    std::unordered_set<ov::Node*> stripped_fqs;

    // Set to true if forward propagation reaches a Result node, meaning the FQ output
    // feeds directly into a model output without a scale-invariant consumer.
    bool result_reachable = false;

    // Forward propagation callback: collect FQ adjustments, run backward propagation
    // from Add branches, and detect Result nodes. Weight scale and FQ adjustments are
    // collected (not applied) until the forward pass completes — they are applied only
    // if no Result was encountered.
    auto forward_propagate_callback = [&](ov::Node* node) {
        // Detect Result node: forward path reaches model output without scale-invariant consumer
        if (ov::is_type<ov::op::v0::Result>(node)) {
            result_reachable = true;
            return;
        }

        // Handle un-stripped FakeQuantize: collect its range constants for adjustment
        if (ov::is_type<ov::op::v0::FakeQuantize>(node)) {
            if (stripped_fqs.count(node))
                return;
            collect_fq_adjustment(node);
            return;
        }

        if (!ov::is_type<ov::op::v1::Add>(node))
            return;

        // For each input of the Add, run backward propagation for weight adjustment
        // on branches that haven't been visited yet (the "other" branch of the residual).
        // This collects weight scales immediately (into pending_weight_scales) rather than
        // deferring the backward pass itself.
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            auto input_node = node->get_input_node_ptr(i);
            // Skip inputs already on the forward path (e.g., skip connections in residual blocks
            // that already carry scaled values) or already backward-visited
            if (visited.count(input_node) || forward_visited.count(input_node))
                continue;

            ov::op::util::visit_path(input_node, visited, collect_weights_scale, backward_skip_predicate);
        }
    };

    // Forward propagation skip predicate: stop at scale-invariant nodes (MVN, Softmax)
    auto forward_skip_predicate = [](ov::Node* n) {
        return ov::is_type_any_of<ov::op::v0::MVN, ov::op::v6::MVN, ov::op::v1::Softmax, ov::op::v8::Softmax>(
            n->shared_from_this());
    };

    // Single-pass: strip each FQ and immediately propagate scale before processing the next FQ.
    // This ensures that forward propagation from FQ_A adjusts downstream FQ_B's ranges,
    // so when FQ_B is processed its y_scale is recomputed from the adjusted (smaller) ranges.
    // Without this, cascaded FQs would each independently compute y_scale from original ranges,
    // potentially double-scaling weights.
    for (const auto& node : f->get_ordered_ops()) {
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node);
        if (!fq || transformation_callback(node)) {
            continue;
        }

        if (!levels_to_strip.count(fq->get_levels())) {
            continue;
        }

        // Skip FQs with non-scalar constants or different input/output ranges
        if (!fq_ranges_are_the_same(fq)) {
            continue;
        }

        // Compute y_scale (dequantization scale) for this FQ:
        //   y_scale = (input_high - input_low) / (levels - 1)
        // Note: these ranges may have been adjusted by forward propagation from a preceding FQ,
        // so y_scale reflects the current (post-adjustment) scale rather than the original.
        const auto& input_low = fq->input_value(1);
        const auto& input_high = fq->input_value(2);

        auto levels_minus_one_node = ov::op::v0::Constant::create(input_high.get_element_type(),
                                                                  ov::Shape{},
                                                                  {static_cast<float>(fq->get_levels() - 1)});
        auto input_range_node = ov::op::util::make_try_fold<ov::op::v1::Subtract>(input_high, input_low);
        auto y_scale_node = ov::op::util::make_try_fold<ov::op::v1::Divide>(input_range_node, levels_minus_one_node);

        // Fold the subgraph to a constant and extract the scalar value
        auto y_scale_const = ov::as_type_ptr<ov::op::v0::Constant>(y_scale_node);
        if (!y_scale_const) {
            continue;
        }
        float y_scale = y_scale_const->cast_vector<float>()[0];

        // Remember the FQ's input node before stripping (this is the node that feeds into the FQ)
        auto propagation_root = fq->get_input_node_shared_ptr(0);

        OPENVINO_ASSERT(replace_output_update_name(fq->output(0), fq->input_value(0)), "FQ stripping failed");
        stripped_fqs.insert(fq.get());
        model_changed = true;

        if (need_weights_adjustment && y_scale > threshold) {
            current_scale_divisor = y_scale * ratio;

            // Clear deferred collections for this FQ's propagation pass
            pending_weight_scales.clear();
            pending_fq_adjustments.clear();
            result_reachable = false;
            // Clear adjusted_fqs so the same FQ can be re-collected for a new propagation
            // (each propagation pass is independent)
            adjusted_fqs.clear();

            // Step 1: Backward propagation from FQ position — collect weight adjustments
            ov::op::util::visit_path(propagation_root.get(),
                                     visited,
                                     collect_weights_scale,
                                     backward_skip_predicate);

            // Step 2: Forward propagation from FQ position — collect FQ adjustments,
            // backward propagation roots from Add branches, and detect Result nodes.
            forward_visited.clear();
            ov::op::util::visit_path_forward(propagation_root.get(),
                                             forward_visited,
                                             forward_propagate_callback,
                                             forward_skip_predicate);

            // Step 3: Apply all collected adjustments only if Result was NOT reached.
            // If forward propagation reached a Result node, the FQ output feeds into a
            // model output without a scale-invariant consumer — weight scaling would
            // change the model's numerical output, so we skip all adjustments.
            if (result_reachable) {
                // Undo adjusted_fqs tracking since we didn't actually modify anything
                adjusted_fqs.clear();
            } else {
                // Apply all collected weight scale adjustments (from initial backward
                // propagation and backward propagation triggered from Add branches)
                for (const auto& pending : pending_weight_scales) {
                    apply_pending_weight_scale(pending);
                }

                // Apply collected FQ range adjustments
                for (auto* fq_node : pending_fq_adjustments) {
                    apply_fq_adjustment(fq_node);
                }
            }
        }
    }

    return model_changed;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov