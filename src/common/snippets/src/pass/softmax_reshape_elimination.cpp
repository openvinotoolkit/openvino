// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/pass/softmax_reshape_elimination.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/validation_util.hpp>

ngraph::snippets::pass::SoftmaxReshapeElimination::SoftmaxReshapeElimination() {
    MATCHER_SCOPE(SoftmaxReshapeElimination);
    const auto m_reshape0 = pattern::wrap_type<opset1::Reshape>(pattern::has_static_shape());
    const auto m_softmax = pattern::wrap_type<ngraph::op::v1::Softmax, ngraph::op::v8::Softmax>({m_reshape0});
    const auto m_reshape1 = pattern::wrap_type<opset1::Reshape>({m_softmax, pattern::wrap_type<opset1::Constant>()});

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(m_reshape1, matcher_name),
            [=](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::SoftmaxReshapeElimination")
            auto& pattern_to_output = m.get_pattern_value_map();
            auto reshape0 = pattern_to_output[m_reshape0].get_node_shared_ptr();
            auto softmax = pattern_to_output[m_softmax].get_node_shared_ptr();
            auto reshape1 = pattern_to_output[m_reshape1].get_node_shared_ptr();

            const auto input_shape = reshape0->get_input_partial_shape(0);
            const auto output_shape = reshape1->get_output_partial_shape(0);
            if (input_shape.is_dynamic() || output_shape.is_dynamic() || input_shape.get_shape() != output_shape.get_shape())
                return false;

            const auto softmax_rank = softmax->get_input_partial_shape(0).rank();
            int64_t axis = 0;
            if (const auto softmax_v8 = ngraph::as_type_ptr<const ov::op::v8::Softmax>(softmax)) {
                OPENVINO_SUPPRESS_DEPRECATED_START
                axis = ngraph::normalize_axis(softmax->get_friendly_name(), softmax_v8->get_axis(), softmax_rank);
                OPENVINO_SUPPRESS_DEPRECATED_END
            } else if (const auto softmax_v1 = ngraph::as_type_ptr<const ov::op::v1::Softmax>(softmax)) {
                axis = softmax_v1->get_axis();
            } else {
                return false;
            }

            // Supports only last axis
            if (axis != softmax_rank.get_length() - 1)
                return false;

            // Dimensions by reduction axis should be equal
            if (input_shape.get_shape().back() != softmax->get_input_shape(0).back())
                return false;

            // Eliminate Reshape before Softmax
            reshape0->output(0).replace(reshape0->input_value(0));
            copy_runtime_info({reshape0->input_value(0).get_node_shared_ptr(), reshape0->output(0).get_node_shared_ptr()},
                reshape0->input_value(0).get_node_shared_ptr());

            // Eliminate Reshape after Softmax with name saving
            replace_output_update_name(reshape1->output(0), reshape1->input_value(0));

            // update axis
            const auto new_axis = input_shape.rank().get_length() - 1;
            if (auto softmax_v8 = ngraph::as_type_ptr<ov::op::v8::Softmax>(softmax)) {
                softmax_v8->set_axis(new_axis);
            } else if (auto softmax_v1 = ngraph::as_type_ptr<ov::op::v1::Softmax>(softmax)) {
                softmax_v1->set_axis(new_axis);
            }

            return true;
        });
}
