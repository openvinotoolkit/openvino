// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fold_const.hpp"

#include "../../logging.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace ov {
namespace npuw {
namespace patterns {
namespace util {

namespace {
// Guard used ONLY by FoldEltwiseOfConsts. Shape-compute arithmetic (e.g. a split
// size expressed as `total - other`) operates on individual dimension sizes,
// i.e. scalar integer values, which are later Unsqueeze'd and Concat'ed into the
// split_lengths vector. Restricting the folder to scalar integer operands means
// it can never touch a (multi-element) weight quantization/decompression
// constant - so no size heuristic is needed and weightless caching / DCOFF stay
// intact.
bool is_scalar_integer_const(const ov::Output<ov::Node>& in) {
    if (!in.get_element_type().is_integral())
        return false;
    const auto& pshape = in.get_partial_shape();
    return pshape.is_static() && ov::shape_size(pshape.to_shape()) == 1;
}

// If every input to `node` is an ov::op::v0::Constant and every output shape
// is statically known, evaluate the node and return one folded Constant per
// output port.  Returns true on success; `replacements` is populated.
bool fold_if_all_const(const std::shared_ptr<ov::Node>& node, ov::OutputVector& replacements) {
    ov::TensorVector inputs;
    inputs.reserve(node->get_input_size());
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        auto c = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(i));
        if (!c)
            return false;
        inputs.emplace_back(c->get_element_type(), c->get_shape(), const_cast<void*>(c->get_data_ptr()));
    }
    ov::TensorVector outputs;
    outputs.reserve(node->get_output_size());
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        const auto& pshape = node->get_output_partial_shape(i);
        if (pshape.is_dynamic())
            return false;
        outputs.emplace_back(node->get_output_element_type(i), pshape.to_shape());
    }
    if (!node->evaluate(outputs, inputs))
        return false;
    replacements.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto new_c = std::make_shared<ov::op::v0::Constant>(outputs[i]);
        new_c->set_friendly_name("NPUW/Folded/" + node->get_friendly_name());
        replacements.push_back(new_c->output(0));
    }
    return true;
}
}  // namespace

FoldShapeOf::FoldShapeOf() {
    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});

    register_matcher(std::make_shared<opp::Matcher>(shape_of, "FoldShapeOf"), [](opp::Matcher& m) {
        auto matched_out = m.get_match_root()->output(0);
        auto& tensor = matched_out.get_tensor();
        if (!tensor.has_and_set_bound())
            return false;
        auto new_c = std::make_shared<ov::op::v0::Constant>(tensor.get_upper_value());
        new_c->set_friendly_name("NPUW/Folded/" + m.get_match_root()->get_friendly_name());
        for (auto& input : matched_out.get_target_inputs()) {
            input.replace_source_output(new_c);
        }
        return false;  // root itself not replaced, only consumers redirected
    });
}

FoldGatherOfConst::FoldGatherOfConst() {
    auto const_data = opp::wrap_type<ov::op::v0::Constant>();
    auto const_idx = opp::wrap_type<ov::op::v0::Constant>();
    auto const_axis = opp::wrap_type<ov::op::v0::Constant>();
    auto gather = opp::wrap_type<ov::op::v8::Gather>({const_data, const_idx, const_axis});

    register_matcher(std::make_shared<opp::Matcher>(gather, "FoldGatherOfConst"), [](opp::Matcher& m) {
        ov::OutputVector replacements;
        if (!fold_if_all_const(m.get_match_root(), replacements))
            return false;
        ov::replace_node(m.get_match_root(), replacements);
        return true;
    });
}

FoldUnsqueezeOfConst::FoldUnsqueezeOfConst() {
    auto const_data = opp::wrap_type<ov::op::v0::Constant>();
    auto const_axes = opp::wrap_type<ov::op::v0::Constant>();
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({const_data, const_axes});

    register_matcher(std::make_shared<opp::Matcher>(unsqueeze, "FoldUnsqueezeOfConst"), [](opp::Matcher& m) {
        ov::OutputVector replacements;
        if (!fold_if_all_const(m.get_match_root(), replacements))
            return false;
        ov::replace_node(m.get_match_root(), replacements);
        return true;
    });
}

FoldConcatOfConsts::FoldConcatOfConsts() {
    auto concat = opp::wrap_type<ov::op::v0::Concat>();

    register_matcher(std::make_shared<opp::Matcher>(concat, "FoldConcatOfConsts"), [](opp::Matcher& m) {
        ov::OutputVector replacements;
        if (!fold_if_all_const(m.get_match_root(), replacements))
            return false;
        ov::replace_node(m.get_match_root(), replacements);
        return true;
    });
}

FoldEltwiseOfConsts::FoldEltwiseOfConsts() {
    auto const_lhs = opp::wrap_type<ov::op::v0::Constant>();
    auto const_rhs = opp::wrap_type<ov::op::v0::Constant>();
    auto eltwise = opp::wrap_type<ov::op::v1::Subtract, ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Divide>(
        {const_lhs, const_rhs});

    register_matcher(std::make_shared<opp::Matcher>(eltwise, "FoldEltwiseOfConsts"), [](opp::Matcher& m) {
        auto node = m.get_match_root();
        // Self-contained weight-safety guard (not shared with the other folders):
        // only fold when both operands are scalar integer constants.
        if (!is_scalar_integer_const(node->input_value(0)) || !is_scalar_integer_const(node->input_value(1)))
            return false;
        ov::OutputVector replacements;
        if (!fold_if_all_const(node, replacements))
            return false;
        ov::replace_node(node, replacements);
        return true;
    });
}

bool FoldShapeComputeChain::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<FoldShapeOf>();
    rewr.add_matcher<FoldGatherOfConst>();
    rewr.add_matcher<FoldUnsqueezeOfConst>();
    rewr.add_matcher<FoldConcatOfConsts>();
    return rewr.run_on_model(model);
}

void foldShapeComputeChainsForConstAttrs(const std::shared_ptr<ov::Model>& model) {
    const bool needs_shape_fold = [&] {
        for (const auto& node : model->get_ordered_ops()) {
            if (!ov::is_type<ov::op::v1::VariadicSplit>(node)) {
                continue;
            }
            if (!ov::is_type<ov::op::v0::Constant>(node->input_value(2).get_node_shared_ptr())) {
                return true;
            }
        }
        return false;
    }();
    if (!needs_shape_fold) {
        return;
    }
    LOG_INFO("Found VariadicSplit with non-constant split_lengths; folding shape-compute "
             "chains before online partitioning.");
    // Local rewrite (NOT the shared FoldShapeComputeChain, which the MoE path in
    // llm_compiled_model.cpp also uses): the shape-compute-chain matchers plus the
    // opt-in FoldEltwiseOfConsts so a Subtract/Add step (split size = total - other)
    // also collapses. Scoping it here keeps every other constant-folding caller
    // untouched.
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<FoldShapeOf>();
    rewr.add_matcher<FoldGatherOfConst>();
    rewr.add_matcher<FoldUnsqueezeOfConst>();
    rewr.add_matcher<FoldEltwiseOfConsts>();
    rewr.add_matcher<FoldConcatOfConsts>();
    rewr.run_on_model(model);
}

}  // namespace util
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
