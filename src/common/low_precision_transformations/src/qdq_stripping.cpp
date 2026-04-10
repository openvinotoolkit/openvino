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
// Motivation: if the stripped FQ's dequantization scale (y_scale) is large,
// the original activation values flowing through the stripped FQ path can exceed f16 range,
// causing overflow and corrupting inference results.
//
// ScaleAdjuster reduces the magnitude of activations, keeping them within f16 range, by dividing weight DQ constants by
// `scale_divisor = y_scale × ratio` (where `ratio = 10.0`).
// Note: Such scaling is possible only if all downstream paths go to scale-invariant nodes (such as MVN or Softmax).
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

    static bool is_scale_invariant(ov::Node* n) {
        return ov::is_type_any_of<ov::op::v0::MVN, ov::op::v6::MVN, ov::op::v1::Softmax, ov::op::v8::Softmax>(n);
    }

    void validate_activations_flow_node(ov::Node* n, bool is_forward) {
        if (is_forward && is_scale_invariant(n)) {
            return;
        }
        // Note: the set of supported nodes is intentionally limited to avoid overcomplicating the adjuster logic and make it safer.
        // The current set is enough for covering all existing models which require scale adjustment.
        if (!ov::is_type_any_of<ov::op::v1::Add,
                                ov::op::v0::Constant,
                                ov::op::v0::Convert,
                                ov::op::v1::Convolution,
                                ov::op::v0::FakeQuantize,
                                ov::op::v0::MatMul,
                                ov::op::v1::Multiply,
                                ov::op::v1::Reshape,
                                ov::op::v1::Transpose>(n)) {
            m_scale_adjustment_possible = false;
        }
    }

    auto make_skip_predicate(bool is_forward) {
        return [this, is_forward](ov::Node* n) {
            const auto& out_precision = n->get_output_element_type(0);
            const bool shapeof_subgraph = ov::is_type<ov::op::v0::ShapeOf>(n) || ov::is_type<ov::op::v3::ShapeOf>(n) ||
                                          out_precision == ov::element::i32 || out_precision == ov::element::i64;
            // Both forward/backward propagation should not visit shape related paths
            if (shapeof_subgraph) {
                return true;
            }

            validate_activations_flow_node(n, is_forward);
            return !scale_adjustment_possible() || (is_forward && is_scale_invariant(n));
        };
    }

    void propagate_backward(ov::Node* root) {
        auto collect_nodes_to_scale = [&](ov::Node* node) {
            using namespace ov::pass::pattern;
            auto convert = wrap_type<ov::op::v0::Convert>({wrap_const()});
            auto sub_const_convert = optional<ov::op::v0::Convert>({wrap_const()});
            auto subtract = optional<ov::op::v1::Subtract>({convert, sub_const_convert});
            auto multiply = wrap_type<ov::op::v1::Multiply>({subtract, wrap_const()});
            auto matcher = std::make_shared<Matcher>(multiply, "WeightsDQPattern");

            // Case 1: DQ block on constant path — collect for scale adjustment
            if (matcher->match(node->shared_from_this())) {
                const auto mul = matcher->get_pattern_value_map().at(multiply).get_node_shared_ptr();
                const bool const_is_in1 = ov::is_type<ov::op::v0::Constant>(mul->get_input_node_shared_ptr(1));

                m_pending_adjustments.push_back(mul->input(const_is_in1 ? 1 : 0));
                // Stop backward propagation since adjustement is done for this branch
                for (const auto& in : matcher->get_match_root()->input_values()) {
                    m_visited.insert(in.get_node());
                }
                return;
            }

            // Case 2: FakeQuantize (un-stripped) — collect for ranges adjustment
            if (ov::is_type<ov::op::v0::FakeQuantize>(node)) {
                for (size_t i = 1; i < node->get_input_size(); ++i) {
                    m_pending_adjustments.push_back(node->input(i));
                    m_visited.insert(node->get_input_node_ptr(i));
                }
                return;
            }

            // Case 3: Layers with weights: backward propagation goes only by 2nd input
            if (ov::is_type_any_of<ov::op::v0::MatMul, ov::op::v1::Multiply, ov::op::v1::Convolution>(node)) {
                m_visited.insert(node->get_input_node_ptr(0));
                return;
            }
        };
        ov::op::util::visit_path(root, m_visited, collect_nodes_to_scale, make_skip_predicate(false));
    }

    void propagate_forward(ov::Node* root) {
        auto collect_nodes_to_scale = [&](ov::Node* node) {
            // Case 1: FakeQuantize (un-stripped) — collect for ranges adjustment
            if (ov::is_type<ov::op::v0::FakeQuantize>(node) && node != m_fq && !node->get_users().empty()) {
                for (size_t i = 1; i < node->get_input_size(); ++i) {
                    m_pending_adjustments.push_back(node->input(i));
                }
                return;
            }

            // Case 2: Commutative ops: backward propagation should be called for all non visited inputs
            if (ov::is_type<ov::op::v1::Add>(node)) {
                for (size_t i = 0; i < node->get_input_size(); ++i) {
                    auto input_node = node->get_input_node_ptr(i);
                    if (m_visited.count(input_node))
                        continue;
                    propagate_backward(input_node);
                }
            }
        };
        ov::op::util::visit_path_forward(root, m_visited, collect_nodes_to_scale, make_skip_predicate(true));
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

    auto fq_ranges_are_the_same = [](const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) {
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
    };

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