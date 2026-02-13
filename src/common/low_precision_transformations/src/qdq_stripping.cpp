// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/qdq_stripping.hpp"

#include <algorithm>
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
#include "openvino/op/abs.hpp"
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
#include "openvino/op/less.hpp"
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
namespace {
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

bool fq_ranges_are_the_same(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) {
    auto equal_with_threshold = [](const ov::Output<ov::Node>& val1, const ov::Output<ov::Node>& val2) {
        auto diff = std::make_shared<ov::op::v1::Subtract>(val1, val2);
        auto abs_diff = std::make_shared<ov::op::v0::Abs>(diff);
        auto eps = ov::op::v0::Constant::create(val1.get_element_type(), {}, {1e-6f});
        auto is_less = ov::util::get_constant_from_source(std::make_shared<ov::op::v1::Less>(abs_diff, eps));

        auto all_true = [](const std::shared_ptr<ov::op::v0::Constant>& c) {
            auto v = c->get_vector<bool>();
            return std::all_of(v.begin(), v.end(), [](bool b) {
                return b;
            });
        };
        return is_less && all_true(is_less);
    };

    return equal_with_threshold(fq->input_value(1), fq->input_value(3)) &&
           equal_with_threshold(fq->input_value(2), fq->input_value(4));
}

// Adjusts weight DQ scales and FQ range constants to compensate for FQ stripping.
// Walks backward from the FQ to find weight DQ blocks, then forward to find
// downstream FQs and detect Result nodes. If forward propagation reaches a Result
// (model output without a scale-invariant consumer), all adjustments are discarded.
class ScaleAdjuster {
public:
    ScaleAdjuster(float scale_divisor, const std::shared_ptr<ov::Node>& fq)
        : m_scale_divisor(scale_divisor),
          m_fq(fq.get()) {}

    void adjust() {
        propagate_backward(m_fq);
        propagate_forward(m_fq);

        if (!m_result_reachable)
            apply_collected_adjustments();
    }

private:
    struct PendingWeightScale {
        std::shared_ptr<ov::Node> old_multiply;
        std::shared_ptr<ov::Node> original_constant;
        std::shared_ptr<ov::Node> mul_input;
        std::vector<ov::Node*> extra_visited;
    };

    float m_scale_divisor;
    ov::Node* m_fq;
    bool m_result_reachable = false;

    std::unordered_set<ov::Node*> m_visited;
    std::unordered_set<ov::Node*> m_forward_visited;
    std::unordered_set<ov::Node*> m_adjusted_fqs;

    std::vector<PendingWeightScale> m_pending_weight_scales;
    std::vector<ov::Node*> m_pending_fq_adjustments;

    void collect_fq_adjustment(ov::Node* node) {
        if (!ov::is_type<ov::op::v0::FakeQuantize>(node) || m_adjusted_fqs.count(node))
            return;
        m_pending_fq_adjustments.push_back(node);
        m_adjusted_fqs.insert(node);
    }

    void propagate_backward(ov::Node* root) {
        using namespace ov::pass::pattern;

        auto collect_scale_to_weight = [&](const PatternValueMap& pattern_map,
                                           const std::shared_ptr<WeightsDequantizationBlock>& dq_block,
                                           const std::vector<ov::Node*>& extra_visited_nodes) {
            auto original_constant = dq_block->get_anchor("mul_const", pattern_map).value().get_node_shared_ptr();
            auto old_multiply = dq_block->get_anchor("multiply", pattern_map).value().get_node_shared_ptr();
            auto mul_input = dq_block->get_anchor("mul_input", pattern_map).value().get_node_shared_ptr();

            if (m_visited.count(old_multiply.get()))
                return;

            m_pending_weight_scales.push_back({old_multiply, original_constant, mul_input, extra_visited_nodes});
        };

        auto collect_weights_scale = [&](ov::Node* node) {
            const auto node_shared = node->shared_from_this();

            // Case 1: Convolution + Add (bias) — scale both Conv weights and bias
            {
                auto conv_weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
                auto conv_pattern = wrap_type<ov::op::v1::Convolution>({any_input(), conv_weights_dq_block});

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

            // Case 2: MatMul + Add (bias) — scale both MatMul weights and bias
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

            // Case 5: FakeQuantize (un-stripped) — collect for range adjustment
            if (ov::is_type<ov::op::v0::FakeQuantize>(node)) {
                collect_fq_adjustment(node);
            }
        };

        auto skip_predicate = [](ov::Node* n) {
            return ov::is_type<ov::op::v0::ShapeOf>(n) || ov::is_type<ov::op::v3::ShapeOf>(n);
        };

        ov::op::util::visit_path(root, m_visited, collect_weights_scale, skip_predicate);
    }

    void propagate_forward(ov::Node* root) {
        auto skip_predicate = [](ov::Node* n) {
            return ov::is_type_any_of<ov::op::v0::MVN, ov::op::v6::MVN, ov::op::v1::Softmax, ov::op::v8::Softmax>(
                n->shared_from_this());
        };

        ov::op::util::visit_path_forward(root, m_forward_visited, [&](ov::Node* node) {
            if (ov::is_type<ov::op::v0::Result>(node)) {
                m_result_reachable = true;
                return;
            }

            if (ov::is_type<ov::op::v0::FakeQuantize>(node) && node != m_fq && !node->get_users().empty()) {
                collect_fq_adjustment(node);
                return;
            }

            if (!ov::is_type<ov::op::v1::Add>(node))
                return;

            for (size_t i = 0; i < node->get_input_size(); ++i) {
                auto input_node = node->get_input_node_ptr(i);
                if (m_visited.count(input_node) || m_forward_visited.count(input_node))
                    continue;
                propagate_backward(input_node);
            }
        }, skip_predicate);
    }

    void apply_collected_adjustments() {
        for (const auto& pending : m_pending_weight_scales) {
            auto divisor_const = ov::op::v0::Constant::create(
                pending.original_constant->get_output_element_type(0), {}, {m_scale_divisor});
            auto new_constant =
                ov::op::util::make_try_fold<ov::op::v1::Divide>(pending.original_constant, divisor_const);
            auto new_multiply = pending.old_multiply->clone_with_new_inputs({pending.mul_input, new_constant});

            ov::replace_node(pending.old_multiply, new_multiply);
            m_visited.insert(new_multiply.get());
            for (auto* n : pending.extra_visited) {
                m_visited.insert(n);
            }
        }

        for (auto* node : m_pending_fq_adjustments) {
            auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node->shared_from_this());
            if (!fq)
                continue;

            auto divisor_const = ov::op::v0::Constant::create(ov::element::f32, {}, {m_scale_divisor});
            for (size_t idx = 1; idx <= 4; ++idx) {
                auto original_const = fq->get_input_node_shared_ptr(idx);
                auto new_const = ov::op::util::make_try_fold<ov::op::v1::Divide>(original_const, divisor_const);
                fq->input(idx).replace_source_output(new_const);
            }
        }
    }
};
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

    bool model_changed = false;

    for (const auto& node : f->get_ordered_ops()) {
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node);
        if (!fq || transformation_callback(node)) {
            continue;
        }

        if (!levels_to_strip.count(fq->get_levels()) || !fq_ranges_are_the_same(fq)) {
            continue;
        }

        // Compute dq_scale = |input_high - input_low| / (levels - 1)
        auto levels_minus_one_node = ov::op::v0::Constant::create(fq->input_value(2).get_element_type(),
                                      ov::Shape{},
                                      {static_cast<float>(fq->get_levels() - 1)});
        auto input_range_node = std::make_shared<ov::op::v1::Subtract>(fq->input_value(2), fq->input_value(1));
        auto abs_input_range = std::make_shared<ov::op::v0::Abs>(input_range_node);
        auto dq_scale_per_elem = ov::util::get_constant_from_source(
            std::make_shared<ov::op::v1::Divide>(abs_input_range, levels_minus_one_node));
        if (!dq_scale_per_elem) {
            continue;
        }

        const auto dq_scale_values = dq_scale_per_elem->cast_vector<float>();
        const auto max_dq_scale = *std::max_element(dq_scale_values.begin(), dq_scale_values.end());
        constexpr auto threshold = 1.0f;

        if (need_weights_adjustment && max_dq_scale > threshold) {
            constexpr auto ratio = 10.0f;
            ScaleAdjuster adjuster(max_dq_scale * ratio, fq);
            adjuster.adjust();
        }

        OPENVINO_ASSERT(replace_output_update_name(fq->output(0), fq->input_value(0)), "FQ stripping failed");
        model_changed = true;
    }

    return model_changed;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov