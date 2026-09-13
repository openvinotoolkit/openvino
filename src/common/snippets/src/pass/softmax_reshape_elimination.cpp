// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/softmax_reshape_elimination.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/symbol.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"

bool ov::snippets::pass::SoftmaxReshapeElimination::eliminate(const std::shared_ptr<ov::op::v1::Reshape>& reshape0,
                                                              const std::shared_ptr<ov::Node>& softmax,
                                                              const std::shared_ptr<ov::op::v1::Reshape>& reshape1) {
    const auto input_shape = reshape0->get_input_partial_shape(0);
    const auto output_shape = reshape1->get_output_partial_shape(0);
    const auto softmax_shape = softmax->get_input_partial_shape(0);
    if (input_shape.rank().is_dynamic() || output_shape.rank().is_dynamic() || softmax_shape.rank().is_dynamic() ||
        input_shape.rank().get_length() == 0 || softmax_shape.rank().get_length() == 0 ||
        input_shape.rank() != output_shape.rank()) {
        return false;
    }

    const auto dims_equal = [](const ov::Dimension& lhs, const ov::Dimension& rhs) {
        return (lhs.is_static() && rhs.is_static() && lhs == rhs) ||
               ov::symbol::are_equal(lhs.get_symbol(), rhs.get_symbol());
    };
    const auto shape_of = ov::as_type_ptr<ov::op::util::ShapeOfBase>(reshape1->input_value(1).get_node_shared_ptr());
    const bool shape_of_input = shape_of && shape_of->input_value(0) == reshape0->input_value(0);
    const bool constant_patterns = ov::is_type<ov::op::v0::Constant>(reshape0->get_input_node_ptr(1)) &&
                                   ov::is_type<ov::op::v0::Constant>(reshape1->get_input_node_ptr(1));
    if (!shape_of_input && !(constant_patterns && input_shape == output_shape) &&
        !std::equal(input_shape.begin(), input_shape.end(), output_shape.begin(), dims_equal)) {
        return false;
    }

    const auto axis = ov::snippets::utils::get_softmax_axis(softmax);
    // Supports only the last axis.
    if (!axis || *axis != softmax_shape.rank().get_length() - 1) {
        return false;
    }

    // An unknown reduction dimension is not proof that the reshape preserves it.
    const auto& in_last_dim = *input_shape.crbegin();
    if (!dims_equal(in_last_dim, *softmax_shape.crbegin()) ||
        (!shape_of_input && !dims_equal(in_last_dim, *output_shape.crbegin()))) {
        return false;
    }

    // Eliminate Reshape before Softmax.
    replace_output_update_name(reshape0->output(0), reshape0->input_value(0));

    // Eliminate Reshape after Softmax with name saving.
    replace_output_update_name(reshape1->output(0), reshape1->input_value(0));

    // Update axis.
    const auto new_axis = input_shape.rank().get_length() - 1;
    if (auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(softmax)) {
        softmax_v8->set_axis(new_axis);
    } else if (auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(softmax)) {
        softmax_v1->set_axis(new_axis);
    }

    return true;
}

ov::snippets::pass::SoftmaxReshapeElimination::SoftmaxReshapeElimination() {
    MATCHER_SCOPE(SoftmaxReshapeElimination);
    const auto m_reshape0 = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>(
        {ov::pass::pattern::any_input(), ov::pass::pattern::wrap_type<ov::op::v0::Constant>()},
        ov::pass::pattern::consumers_count(1));
    const auto m_softmax =
        ov::pass::pattern::wrap_type<ov::op::v1::Softmax, ov::op::v8::Softmax>({m_reshape0},
                                                                               ov::pass::pattern::consumers_count(1));
    const auto m_reshape1 = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>(
        {m_softmax, ov::pass::pattern::wrap_type<ov::op::v0::Constant>()});

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_reshape1, matcher_name),
                     [=](ov::pass::pattern::Matcher& m) {
                         OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform,
                                            "Snippets::op::SoftmaxReshapeElimination")
                         auto& pattern_to_output = m.get_pattern_value_map();
                         auto reshape0 = pattern_to_output[m_reshape0].get_node_shared_ptr();
                         auto softmax = pattern_to_output[m_softmax].get_node_shared_ptr();
                         auto reshape1 = pattern_to_output[m_reshape1].get_node_shared_ptr();
                         return eliminate(ov::as_type_ptr<ov::op::v1::Reshape>(reshape0),
                                          softmax,
                                          ov::as_type_ptr<ov::op::v1::Reshape>(reshape1));
                     });
}
