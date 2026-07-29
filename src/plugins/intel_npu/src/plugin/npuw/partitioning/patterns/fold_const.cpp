// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fold_const.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace ov {
namespace npuw {
namespace patterns {
namespace util {

namespace {
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

bool FoldShapeComputeChain::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<FoldShapeOf>();
    rewr.add_matcher<FoldGatherOfConst>();
    rewr.add_matcher<FoldUnsqueezeOfConst>();
    rewr.add_matcher<FoldConcatOfConsts>();
    return rewr.run_on_model(model);
}

}  // namespace util
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
