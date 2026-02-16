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
#include "openvino/op/transpose.hpp"
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

    std::shared_ptr<ov::Node> get_multiply(const ov::pass::pattern::PatternValueMap& pm) const {
        return get_anchor("multiply", pm).value().get_node_shared_ptr();
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

        if (scale_adjustment_possible()) {
            for (auto& input : m_pending_adjustments) {
                auto original_const = input.get_source_output();
                auto divisor_const =
                    ov::op::v0::Constant::create(original_const.get_element_type(), {}, {m_scale_divisor});
                auto new_const = ov::op::util::make_try_fold<ov::op::v1::Divide>(original_const, divisor_const);
                OPENVINO_ASSERT(new_const, "Adjusted scale must be constant");
                ov::copy_runtime_info(original_const.get_node_shared_ptr(), new_const);
                input.replace_source_output(new_const);
            }
        }
    }

private:
    float m_scale_divisor;
    ov::Node* m_fq;
    bool m_scale_adjustment_possible = true;

    std::unordered_set<ov::Node*> m_visited;
    std::vector<ov::Input<ov::Node>> m_pending_adjustments;

    bool scale_adjustment_possible() const {
        return m_scale_adjustment_possible;
    }

    static bool is_allowed_node(ov::Node* n) {
        return ov::is_type_any_of<ov::op::v1::Add,
                                  ov::op::v0::Constant,
                                  ov::op::v0::Convert,
                                  ov::op::v1::Convolution,
                                  ov::op::v0::FakeQuantize,
                                  ov::op::v0::MatMul,
                                  ov::op::v1::Multiply,
                                  ov::op::v1::Reshape,
                                  ov::op::v1::Transpose,
                                  ov::op::v0::MVN,
                                  ov::op::v6::MVN,
                                  ov::op::v1::Softmax,
                                  ov::op::v8::Softmax>(n);
    }

    void collect_dq_multiply(const std::shared_ptr<ov::Node>& multiply) {
        bool const_is_in1 = ov::is_type<ov::op::v0::Constant>(multiply->input_value(1).get_node());
        m_pending_adjustments.push_back(multiply->input(const_is_in1 ? 1 : 0));
    }

    void collect_fq_ranges(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) {
        for (size_t i = 1; i < fq->get_input_size(); ++i) {
            m_pending_adjustments.push_back(fq->input(i));
        }
    }

    void propagate_backward(ov::Node* root) {
        using namespace ov::pass::pattern;

        auto collect_nodes_to_scale = [&](ov::Node* node) {
            const auto node_shared = node->shared_from_this();
            // Case 1: DQ block on constant path — collect for scale adjustment
            auto weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto matcher = std::make_shared<Matcher>(weights_dq_block, "WeightsDQPattern");
            if (matcher->match(node_shared)) {
                collect_dq_multiply(weights_dq_block->get_multiply(matcher->get_pattern_value_map()));
                for (const auto& in : matcher->get_match_root()->input_values()) {
                    m_visited.insert(in.get_node());
                }
                return;
            }

            // Case 2: FakeQuantize (un-stripped) — collect for ranges adjustment
            if (auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node_shared)) {
                collect_fq_ranges(fq);
                for (size_t i = 1; i < fq->get_input_size(); ++i) {
                    m_visited.insert(fq->input_value(i).get_node());
                }
                return;
            }

            // Case 3: Layers with weights: Backward propagation goes only by 2nd input
            if (ov::is_type_any_of<ov::op::v0::MatMul, ov::op::v1::Multiply, ov::op::v1::Convolution>(node)) {
                m_visited.insert(node->get_input_node_ptr(0));
                return;
            }
        };

        auto skip_predicate = [&](ov::Node* n) {
            if (!is_allowed_node(n)) {
                m_scale_adjustment_possible = false;
            }

            const auto& out_precision = n->get_output_element_type(0);
            // Scale adjustment shouldn't be propagated via ShapeOf subgraphs
            const bool shapeof_subgraph = ov::is_type<ov::op::v0::ShapeOf>(n) || ov::is_type<ov::op::v3::ShapeOf>(n) ||
                                          out_precision == ov::element::i32 || out_precision == ov::element::i64;
            return !scale_adjustment_possible() || shapeof_subgraph;
        };

        ov::op::util::visit_path(root, m_visited, collect_nodes_to_scale, skip_predicate);
    }

    void propagate_forward(ov::Node* root) {
        using namespace ov::pass::pattern;

        auto collect_nodes_to_scale = [&](ov::Node* node) {
            const auto node_shared = node->shared_from_this();

            auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node_shared);
            if (fq && node != m_fq && !node->get_users().empty()) {
                collect_fq_ranges(fq);
                return;
            }

            if (ov::is_type<ov::op::v1::Add>(node)) {
                for (size_t i = 0; i < node->get_input_size(); ++i) {
                    auto input_node = node->get_input_node_ptr(i);
                    if (m_visited.count(input_node))
                        continue;
                    propagate_backward(input_node);
                }
            }
        };

        auto skip_predicate = [&](ov::Node* n) {
            if (!is_allowed_node(n)) {
                m_scale_adjustment_possible = false;
            }

            // Propagation stops at scale-invariant nodes
            const bool scale_invariant_nodes =
                ov::is_type_any_of<ov::op::v0::MVN, ov::op::v6::MVN, ov::op::v1::Softmax, ov::op::v8::Softmax>(
                    n->shared_from_this());
            const auto& out_precision = n->get_output_element_type(0);
            // Scale adjustment shouldn't be propagated via ShapeOf subgraphs
            const bool shapeof_subgraph = ov::is_type<ov::op::v0::ShapeOf>(n) || ov::is_type<ov::op::v3::ShapeOf>(n) ||
                                          out_precision == ov::element::i32 || out_precision == ov::element::i64;
            return !scale_adjustment_possible() || scale_invariant_nodes || shapeof_subgraph;
        };

        ov::op::util::visit_path_forward(root, m_visited, collect_nodes_to_scale, skip_predicate);
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